import argparse
import glob
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import d4rl

import wandb
from tqdm import trange

# === MLP and Model Definitions: Beta-VAE, Diffusion, QNet, VNet ===

# ---------- Base Modules ----------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.GELU(), nn.LayerNorm(dims[i+1])]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- Beta-VAE ----------
class BetaVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super().__init__()
        self.encoder = MLP(state_dim + action_dim, [128, 128], latent_dim * 2)
        self.policy_decoder = MLP(state_dim + latent_dim, [128, 128], action_dim)
        self.state_decoder = MLP(state_dim + latent_dim, [128, 128], state_dim)
        self.state_prior = MLP(state_dim, [128], latent_dim * 2)

    def encode(self, s, a):
        h = self.encoder(torch.cat([s, a], dim=-1))
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode_action(self, s, z):
        return self.policy_decoder(torch.cat([s, z], dim=-1))

    def decode_state(self, s, z):
        return self.state_decoder(torch.cat([s, z], dim=-1))

    def prior(self, s):
        h = self.state_prior(s)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def elbo_loss(self, s, a, beta=0.1, lambda_state=1.0, gamma=1.0, diffusion_model=None, steps=200):
        z, mu, logvar = self.encode(s, a)
        recon_a = self.decode_action(s, z)
        recon_s = self.decode_state(s, z)
        prior_mu, prior_logvar = self.prior(s)
        kl = -0.5 * torch.sum(1 + logvar - prior_logvar - ((mu - prior_mu)**2 + torch.exp(logvar)) / torch.exp(prior_logvar), dim=-1)
        loss_action = F.mse_loss(recon_a, a, reduction='none').sum(-1)
        loss_state = F.mse_loss(recon_s, s, reduction='none').sum(-1)
        if diffusion_model is not None:
            z_t = z.clone()
            for j in reversed(range(1, steps + 1)):
                t = torch.full((z_t.shape[0],), j, device=z_t.device)
                z_0_pred = diffusion_model(z_t, s, t)
                beta_t = 1.0 / steps
                alpha_t = 1.0 - beta_t
                z_t = alpha_t * z_0_pred + (1 - alpha_t) * z_t
            l4 = F.mse_loss(z_t, z, reduction='none').sum(-1)
        else:
            l4 = 0.0
        return (loss_action + lambda_state * loss_state + beta * kl + gamma * l4).mean()

# ---------- Diffusion Model ----------
class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim, state_dim):
        super().__init__()
        self.net = MLP(latent_dim + state_dim + 1, [128, 128], latent_dim)

    def forward(self, z_t, state, t):  # Fixed method signature
        t_embed = t.float().unsqueeze(-1) / 1000.0
        input_vec = torch.cat([z_t, state, t_embed], dim=-1)
        pred_z0 = self.net(input_vec)
        return pred_z0  # Fixed unmatched parenthesis

def ddpm_sample(diffusion_model, state, latent_dim=16, steps=10):
    B = state.size(0)
    z = torch.randn(B, latent_dim, device=state.device)
    for j in reversed(range(1, steps + 1)):
        t = torch.full((B,), j, device=state.device)
        z_0 = diffusion_model(z, state, t)
        beta_t = 1.0 / steps  # variance schedule
        alpha_t = 1.0 - beta_t
        z = alpha_t * z_0 + (1 - alpha_t) * z + torch.randn_like(z) * (beta_t ** 0.5)
    return z

# ---------- Q and V Networks ----------
class DoubleQNet(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        self.q1 = MLP(state_dim + latent_dim, [256, 256], 1)
        self.q2 = MLP(state_dim + latent_dim, [256, 256], 1)

    def forward(self, s, z):
        x = torch.cat([s, z], dim=-1)
        q1 = torch.clamp(self.q1(x), -100.0, 100.0).squeeze(-1)
        q2 = torch.clamp(self.q2(x), -100.0, 100.0).squeeze(-1)
        return q1, q2

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.v = MLP(state_dim, [256, 256], 1)

    def forward(self, s):
        return torch.clamp(self.v(s), -100.0, 100.0).squeeze(-1)

# ---------- Losses ----------
def expectile_loss(q_val, v_val, tau=0.9):
    u = q_val - v_val
    weight = torch.where(u > 0, tau, 1 - tau)
    return (weight * u**2).mean()

def compute_q_loss(q_net_target, v_net_target, s, z, r, s_next, gamma=0.99):
    with torch.no_grad():
        v_target = v_net_target(s_next)
        q_target = r + gamma * v_target
    q1, q2 = q_net_target(s, z)
    return F.mse_loss(torch.min(q1, q2), q_target)

def sample_n_latents(diffusion_model, s, n=500, latent_dim=16, steps=10):
    B = s.size(0)
    s_rep = s.repeat_interleave(n, dim=0)             # (B*n, state_dim)
    z_rep = ddpm_sample(diffusion_model, s_rep,
                        latent_dim=latent_dim, steps=steps)  # (B*n, latent_dim)
    return z_rep.view(B, n, latent_dim)               # (B, n, latent_dim)

def compute_v_loss(q_net_t, v_net, s,
                   z_ddpm, z_data,
                   k=100, tau=0.9, lambda_v=0.1,
                   return_stats=False):
    """
    z_ddpm : (B, n_d, latent)   – латенты генератора
    z_data : (B, n_g, latent)   – латенты из датасета β-VAE.encode
    """
    B = s.size(0)

    # ---- объединяем -------------------------------------------------------
    z_all = torch.cat([z_ddpm, z_data], dim=1)          # (B, n_d+n_g, latent)
    n_all = z_all.size(1)

    with torch.no_grad():
        s_exp = s.unsqueeze(1).expand(-1, n_all, -1)
        q1, q2 = q_net_t(
            s_exp.reshape(-1, s.shape[-1]),
            z_all.reshape(-1, z_all.shape[-1]))
        q_vals = torch.min(q1, q2).view(B, n_all)       # (B, n_all)

        # ---- top-k фильтр -------------------------------------------------
        topk_q, idx = torch.topk(q_vals, k=min(k, n_all), dim=1)
        z_sel  = torch.gather(z_all, 1,
                              idx.unsqueeze(-1).expand(-1, -1, z_all.shape[-1]))
        q_sel  = topk_q                                   # (B, k)

    v_pred = v_net(s).unsqueeze(1)                    # (B,1)
    u      = q_sel - v_pred                           # (B,k)
    w      = torch.where(u > 0, tau, 1 - tau)
    loss   = lambda_v * (w * u.pow(2)).mean()

    if return_stats:
        gap      = (q_sel.mean(dim=1) - v_pred.squeeze(1)).mean().detach()
        v_mean   = v_pred.mean().detach()
        q_mean   = q_sel.mean().detach()
        return loss, gap, v_mean, q_mean, v_pred.detach(), q_sel.detach()
    return loss
# -----------------------------------------------------------
# TRAINING-STEP
# -----------------------------------------------------------
def train_diar_step(
    replay_buffer,
    q_net, v_net,
    q_net_target, v_net_target,
    diffusion_model, beta_vae,
    optimizer_q, optimizer_v,
    tau_now,                           # <-- адаптивный τ передаётся извне
    gamma=0.995,
    latent_dim=16, ddpm_steps=10,
    device="cpu", step=0,
    gap_stop=-2.5                     # порог runaway-guard
):
    # -------- sample batch --------------------------------------------------
    batch   = replay_buffer.sample(128)
    s, a    = batch['state'], batch['action']
    r, s_n  = batch['reward'], batch['next_state']
    w, idxs = batch['weights'], batch['indices']

    # -------- Q-update ------------------------------------------------------
    with torch.no_grad():
        z, _, _   = beta_vae.encode(s, a)
        v_tar  = v_net_target(s_n)
        q_target = torch.clamp(r + gamma * v_tar, -100.0, 100.0)

    q1, q2     = q_net(s, z)
    q_pred     = torch.min(q1, q2)
    td_error   = q_target - q_pred
    loss_q     = (w * td_error.pow(2).unsqueeze(1)).mean()

    optimizer_q.zero_grad()
    loss_q.backward()
    optimizer_q.step()

    # PER приоритеты
    with torch.no_grad():
        replay_buffer.update_priorities(idxs, td_error.abs() + 1e-6)

    # -------- V-update ------------------------------------------------------
    with torch.no_grad():
        z_ddpm = sample_n_latents(diffusion_model, s,
                                  n=500, latent_dim=latent_dim,
                                  steps=ddpm_steps)                  # (B,500,L)
        # 32 латентов из датасета
        z_data, _, _ = beta_vae.encode(s, a)
        z_data = z_data.unsqueeze(1).expand(-1, 32, -1)              # (B,32,L)

    loss_v, expect_gap, v_mean, q_mean, v_pred, q_sel = compute_v_loss(
        q_net_target, v_net, s,
        z_ddpm, z_data,
        k=100, tau=tau_now,
        lambda_v=0.5, return_stats=True)

    if expect_gap < -2.5:      # runaway-guard
        loss_v = torch.zeros_like(loss_v)

    optimizer_v.zero_grad()
    loss_v.backward()
    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
    optimizer_v.step()

    # -------- soft target ---------------------------------------------------
    with torch.no_grad():
        for p, tp in zip(q_net.parameters(), q_net_target.parameters()):
            tp.data.mul_(0.995).add_(0.005 * p.data)
        for p, tp in zip(v_net.parameters(), v_net_target.parameters()):
            tp.data.mul_(0.995).add_(0.005 * p.data)

    # -------- logging -------------------------------------------------------
    if step % 500 == 0 and wandb.run:
        wandb.log({
            "loss/q": loss_q.item(),
            "loss/v": loss_v.item(),
            "expectile_gap": expect_gap.item(),
            "v_mean": v_mean.item(),
            "q_sel_mean": q_mean.item(),
            "q_max": q_sel.max().item(),
            "v_max": v_pred.max().item()
        }, step=step)

    return expect_gap.item()          # для адаптивного τ в основной петле

# ---------- Policy Execution ----------
def policy_execute(
    env,
    q_net,
    v_net,
    beta_vae,
    diffusion_model,
    steps=30,
    latent_dim=16,
    ddpm_steps=10,
    revaluation_attempts=3,
    device="cpu"
):
    state = env.reset()
    done = False
    total_reward = 0
    t = 0

    while not done and t < steps:
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for _ in range(revaluation_attempts):
            z = ddpm_sample(diffusion_model, s, latent_dim, ddpm_steps)
            a = beta_vae.decode_action(s, z)
            s_pred = beta_vae.decode_state(s, z)

            if v_net(s) > v_net(s_pred):     # пересэмплировать, если текущее лучше
                continue                     # вместо break
            else:
                break

        a_np = a.squeeze(0).detach().cpu().numpy()
        next_state, reward, done, info = env.step(a_np)
        state = next_state
        total_reward += reward
        t += 1

    return total_reward

# ---------- Prioritized Replay Buffer ----------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device, alpha=0.6, beta_start=0.3, beta_frames=100000):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity,), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.priorities = torch.ones((capacity,), device=device)

    def add(self, s, a, r, s_next):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.priorities[self.ptr] = self.priorities.max() if self.size > 0 else 1.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == self.capacity:
            probs = self.priorities ** self.alpha
        else:
            probs = self.priorities[:self.ptr] ** self.alpha
        probs /= probs.sum()

        indices = torch.multinomial(probs, batch_size, replacement=False)

        beta = self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames)
        beta = min(beta, 1.0)
        self.frame += 1

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return {
            'state': self.states[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices],
            'next_state': self.next_states[indices],
            'weights': weights.unsqueeze(1).to(self.device),
            'indices': indices
        }

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

# === Beta-VAE and Diffusion Training ===
def train_beta_vae(
    env,
    dataset,
    beta_vae,
    diffusion_model,
    device="cuda",
    epochs=100,
    save_every=10,
    save_dir="output",
):
    # wandb.watch(beta_vae)

    H = 30
    obs = torch.tensor(dataset["observations"], dtype=torch.float32)
    acts = torch.tensor(dataset["actions"], dtype=torch.float32)
    N = len(obs) - H
    sequences = [(obs[i:i+H], acts[i:i+H]) for i in range(N)]

    optimizer = torch.optim.Adam(beta_vae.parameters(), lr=5e-5)
    batch_size = 1024

    for epoch in trange(epochs, desc="BetaVAE Training"):
        random.shuffle(sequences)
        losses = []

        for i in range(0, N, batch_size):
            batch = sequences[i:i+batch_size]
            s = torch.stack([item[0][0] for item in batch]).to(device)
            a = torch.stack([item[1][0] for item in batch]).to(device)
            loss = beta_vae.elbo_loss(s, a, beta=0.1, diffusion_model=diffusion_model, steps=10)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        wandb.log({"beta_vae/elbo_loss": avg_loss}, step=epoch)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(beta_vae.state_dict(), f"{save_dir}/vae/beta_vae_epoch{epoch}.pt")

def train_diffusion_model(
    dataset,
    beta_vae,
    diffusion_model,
    device="cuda",
    epochs=450,
    save_every=50,
    save_dir="output",
):
    # wandb.watch(diffusion_model)

    H = 30
    obs = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
    acts = torch.tensor(dataset["actions"], dtype=torch.float32).to(device)
    N = len(obs) - H

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)
    batch_size = 1024

    for epoch in trange(epochs, desc="Diffusion Training"):
        indices = torch.randperm(N)
        losses = []

        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            s = obs[idx]
            a = acts[idx]
            with torch.no_grad():
                z, _, _ = beta_vae.encode(s, a)
            t = torch.randint(1, 11, (z.shape[0],), device=device)
            z_j = z + torch.randn_like(z) * (1.0 / 10)**0.5
            pred = diffusion_model(z_j, s, t)
            loss = F.mse_loss(pred, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        wandb.log({"diffusion/loss": avg_loss}, step=epoch)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(diffusion_model.state_dict(), f"{save_dir}/diffusion/diffusion_model_epoch{epoch}.pt")


def adaptive_tau(gap, hi=0.9, lo=0.7, lo2=0.6, up_thresh=1.0, low_thresh=-0.5):
    if gap > up_thresh:
        return lo2       # V сильно ниже Q — делаем τ≈0.6
    return lo  if gap < low_thresh else hi

# -----------------------------------------------------------
# MAIN TRAINING LOOP
# -----------------------------------------------------------
def train_diar(
    env, dataset,
    state_dim, action_dim,
    latent_dim=16,
    beta_vae=None, diffusion_model=None,
    num_steps=100_000,
    device="cuda",
    save_every=10_000,
    save_dir="output",
):
    # ---------- replay buffer preload --------------------------------------
    horizon = 30
    rb_size = len(dataset['observations']) - horizon
    replay_buffer = PrioritizedReplayBuffer(
        capacity=rb_size, state_dim=state_dim, action_dim=action_dim,
        device=device, alpha=0.6, beta_start=0.3, beta_frames=num_steps)

    print(f"Preloading {rb_size} transitions ...")
    for i in trange(rb_size):
        s  = torch.tensor(dataset['observations'][i]      , device=device)
        a  = torch.tensor(dataset['actions'][i]           , device=device)
        r  = sum((0.995**j) * dataset['rewards'][i+j] for j in range(horizon))
        sn = torch.tensor(dataset['next_observations'][i+horizon], device=device)
        replay_buffer.add(s, a, r, sn)

    # ---------- models ------------------------------------------------------
    beta_vae       = beta_vae or BetaVAE(state_dim, action_dim, latent_dim).to(device)
    diffusion_model= diffusion_model or LatentDiffusionModel(latent_dim, state_dim).to(device)
    q_net, v_net   = DoubleQNet(state_dim, latent_dim).to(device), ValueNet(state_dim).to(device)
    q_tgt, v_tgt   = DoubleQNet(state_dim, latent_dim).to(device), ValueNet(state_dim).to(device)
    q_tgt.load_state_dict(q_net.state_dict())
    v_tgt.load_state_dict(v_net.state_dict())

    optimizer_q = torch.optim.Adam(q_net.parameters(), lr=5e-4)
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=1e-4, weight_decay=1e-4)

    # ---------- scheduler: 25 эпох -----------------------------------------
    batch_size   = 128
    steps_per_ep = max(1, rb_size // batch_size)
    step_size_lr = 25 * steps_per_ep       # ~25 эпох
    # print(step_size_lr)
    # exit()
    step_size_lr = 50000
    scheduler_v  = torch.optim.lr_scheduler.StepLR(
        optimizer_v, step_size=step_size_lr, gamma=0.3)

    # ---------- loop --------------------------------------------------------
    expect_gap_prev = 0.0
    pbar = trange(num_steps, desc="DIAR Training")
    for step in pbar:
        tau_now = adaptive_tau(expect_gap_prev)     # 0.95 / 0.5

        expect_gap_prev = train_diar_step(
            replay_buffer,
            q_net, v_net,
            q_tgt, v_tgt,
            diffusion_model, beta_vae,
            optimizer_q, optimizer_v,
            tau_now,
            gamma=0.995,
            latent_dim=latent_dim,
            ddpm_steps=10,
            device=device,
            step=step)

        scheduler_v.step()                          # LR-sheduler для V

        # ---- периодическая оценка -----------------------------------------
        if step % 500 == 0:
            rew = policy_execute(env, q_net, v_net,
                                 beta_vae, diffusion_model,
                                 device=device)
            wandb.log({"eval/reward": rew, "lr_v": scheduler_v.get_last_lr()[0]}, step=step)
            pbar.set_postfix({"eval_reward": rew})

        # ---- чекпоинты -----------------------------------------------------
        if step % save_every == 0 or step == num_steps - 1:
            torch.save(q_net.state_dict()      , f"{save_dir}/diar/q_net_step{step}.pt")
            torch.save(v_net.state_dict()      , f"{save_dir}/diar/v_net_step{step}.pt")
            torch.save(beta_vae.state_dict()   , f"{save_dir}/diar/beta_vae_step{step}.pt")
            torch.save(diffusion_model.state_dict(),
                       f"{save_dir}/diar/diffusion_model_step{step}.pt")



def load_latest_checkpoint(model, name, folder="output"):
    checkpoints = glob.glob(os.path.join(folder, f"{name}_*.pt"))
    if not checkpoints:
        print(f"No checkpoint found for {name}")
        return model
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Loading {name} from {latest}")
    model.load_state_dict(torch.load(latest))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="maze2d-umaze-v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="output")
    args = parser.parse_args()

    wandb.init(project="diar", name="diar_run")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "/vae", exist_ok=True)
    os.makedirs(args.save_dir + "/diffusion", exist_ok=True)
    os.makedirs(args.save_dir + "/diar", exist_ok=True)

    env = gym.make(args.env)
    dataset = env.get_dataset()
    if "next_observations" not in dataset:
        print("Generating next_observations from observations...")
        dataset["next_observations"] = dataset["observations"][1:]
        dataset["observations"] = dataset["observations"][:-1]
        dataset["actions"] = dataset["actions"][:-1]
        dataset["rewards"] = dataset["rewards"][:-1]
        if "terminals" in dataset:
            dataset["terminals"] = dataset["terminals"][:-1]
        if "timeouts" in dataset:
            dataset["timeouts"] = dataset["timeouts"][:-1]

    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]
    latent_dim = 16

    beta_vae = BetaVAE(state_dim, action_dim, latent_dim).to(args.device)
    diffusion_model = LatentDiffusionModel(latent_dim, state_dim).to(args.device)

    # print("=== Training Beta-VAE ===")
    # train_beta_vae(env, dataset, beta_vae, diffusion_model, device=args.device, epochs=100, save_dir=args.save_dir)

    # print("=== Training Diffusion Model ===")
    # train_diffusion_model(dataset, beta_vae, diffusion_model, device=args.device, epochs=450, save_dir=args.save_dir)

    beta_vae = load_latest_checkpoint(beta_vae, "beta_vae", folder=args.save_dir + "/vae")
    beta_vae.eval()
    for p in beta_vae.parameters():
        p.requires_grad = False
    diffusion_model = load_latest_checkpoint(diffusion_model, "diffusion_model", folder=args.save_dir + "/diffusion")

    print("=== Starting DIAR Training ===")
    train_diar(env=env, dataset=dataset, state_dim=state_dim, action_dim=action_dim,
               latent_dim=latent_dim, beta_vae=beta_vae, diffusion_model=diffusion_model,
               num_steps=100000, device=args.device, save_dir=args.save_dir)
