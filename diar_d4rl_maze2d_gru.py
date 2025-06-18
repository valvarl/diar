import argparse
import glob
import math
import os
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import d4rl

from tqdm import trange
import wandb


# ---------------------------------------------------------------------------
# Helper blocks
# ---------------------------------------------------------------------------

def mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    dims = (in_dim, *hidden)
    for i in range(len(hidden)):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.LayerNorm(dims[i + 1])]
    layers.append(nn.Linear(hidden[-1] if hidden else in_dim, out_dim))
    return nn.Sequential(*layers)

# ---------------------------------------------------------------------------
# β‑VAE (Bi‑GRU encoder, policy/state decoders, state prior)
# ---------------------------------------------------------------------------

class EncoderGRU(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.rnn = nn.GRU(in_dim, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.mu  = nn.Linear(128 * 2, latent_dim)
        self.logvar = nn.Linear(128 * 2, latent_dim)

    def forward(self, seq):  # seq: (B, H, in_dim)
        _, h = self.rnn(seq)  # h: (4, B, 128)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 256)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

class BetaVAE(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = EncoderGRU(state_dim + action_dim, latent_dim)
        self.policy_dec = mlp(state_dim + latent_dim, (128, 128), action_dim)
        self.state_dec  = mlp(state_dim + latent_dim, (128, 128), state_dim)
        self.state_prior = mlp(state_dim, (128,), latent_dim * 2)

    def encode(self, s_seq, a_seq):
        # s_seq, a_seq: (B,H,D)
        seq = torch.cat([s_seq, a_seq], dim=-1)
        return self.encoder(seq)

    def decode_action(self, s, z):
        return self.policy_dec(torch.cat([s, z], dim=-1))

    def decode_state(self, s, z):
        return self.state_dec(torch.cat([s, z], dim=-1))

    def prior(self, s):
        h = self.state_prior(s)
        return h.chunk(2, dim=-1)

    def elbo_loss(self, s_seq, a_seq, beta: float = 0.1, lambda_state: float = 1.0,
                  diffusion_model: Optional[nn.Module] = None, steps: int = 200):
        """ELBO from Eq.(2) **plus** optional diffusion-prior consistency.
        The extra term corresponds to the 200-step transformer prior described
        in Appendix A (we approximate it with the passed *diffusion_model*).
        Args:
            s_seq, a_seq    : (B, H, D) sequences (H=30 for Maze2D)
            beta            : β-VAE weight for KL term (0.1)
            lambda_state    : weight for state reconstruction (1.0)
            diffusion_model : if provided, adds L4 term ‖z_T - z‖² (see paper)
            steps           : number of noising steps (default 200)
        """
        z, mu, logvar = self.encode(s_seq, a_seq)                 # (B, latent)
        s0 = s_seq[:, 0]; sH = s_seq[:, -1]
        # reconstructions ---------------------------------------------------
        rec_a = self.decode_action(s0, z)
        rec_s = self.decode_state(s0, z)
        # KL with state prior ----------------------------------------------
        p_mu, p_logv = self.prior(s0)
        kl = -0.5 * (1 + logvar - p_logv - ((mu - p_mu) ** 2 + logvar.exp()) / p_logv.exp()).sum(-1)
        # additional diffusion‑prior term (optional) ------------------------
        if diffusion_model is not None:
            beta_t  = 1.0 / steps
            alpha_t = 1.0 - beta_t
            z_t = z.clone()
            for t in reversed(range(1, steps + 1)):
                tt = torch.full((z_t.size(0),), t, device=z_t.device)
                z0_pred = diffusion_model(z_t, s0, tt)
                z_t = alpha_t * z0_pred + (1 - alpha_t) * z_t  # denoise one step
            l4 = F.mse_loss(z_t, z, reduction='none').sum(-1)
        else:
            l4 = 0.0
        # final ELBO --------------------------------------------------------
        l_a = F.mse_loss(rec_a, a_seq[:, 0], reduction='none').sum(-1)
        l_s = F.mse_loss(rec_s, sH, reduction='none').sum(-1)
        return (l_a + lambda_state * l_s + beta * kl + l4).mean()


# ---------------------------------------------------------------------------
# Simple U‑Net‑like latent diffusion (down → up with residual blocks)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.ln1(F.gelu(self.fc1(x)))
        h = self.ln2(F.gelu(self.fc2(h)))
        return x + 0.5 * h

class LatentDiffusionUNet(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int):
        super().__init__()
        D = latent_dim + state_dim + 1  # + timestep embed
        self.down1 = mlp(D, (256,), 128)
        self.down2 = mlp(128, (128,), 64)
        self.mid  = ResBlock(64)
        self.up1  = mlp(64, (128,), 128)
        self.up2  = mlp(128, (256,), latent_dim)

    def forward(self, z_t, s, t):
        t_embed = t.float().unsqueeze(-1) / 1000.0
        x = torch.cat([z_t, s, t_embed], dim=-1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m  = self.mid(d2)
        up = self.up1(m)
        out = self.up2(up + d1)  # skip
        return out

# DDPM sample (T=500 by default)

def ddpm_sample(model, s, latent_dim, steps=500):
    z = torch.randn(s.size(0), latent_dim, device=s.device)
    beta = 1.0 / steps
    alpha = 1.0 - beta
    for t in reversed(range(1, steps + 1)):
        tt = torch.full((s.size(0),), t, device=s.device)
        z0 = model(z, s, tt)
        z = alpha * z0 + (1 - alpha) * z + torch.randn_like(z) * math.sqrt(beta)
    return z

# ---------------------------------------------------------------------------
# Q‑network / V‑network (as in Appendix A.3)
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = mlp(in_dim, (256,), 256)

    def forward(self, x):
        return self.net(x)

class LatentEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = mlp(in_dim, (128,), 128)

    def forward(self, z):
        return self.net(z)

class DoubleQNet(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        self.s_enc = StateEncoder(state_dim)
        self.z_enc = LatentEncoder(latent_dim)
        self.head1 = mlp(256 + 128, (256, 256), 1)
        self.head2 = mlp(256 + 128, (256, 256), 1)

    def forward(self, s, z):
        sz = torch.cat([self.s_enc(s), self.z_enc(z)], dim=-1)
        return self.head1(sz).squeeze(-1), self.head2(sz).squeeze(-1)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.enc = StateEncoder(state_dim)
        self.v   = mlp(256, (256, 256), 1)

    def forward(self, s):
        return self.v(self.enc(s)).squeeze(-1)

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

# def sample_n_latents(diffusion_model, s, n=500, latent_dim=16, steps=10):
#     B = s.size(0)
#     s_rep = s.repeat_interleave(n, dim=0)             # (B*n, state_dim)
#     z_rep = ddpm_sample(diffusion_model, s_rep,
#                         latent_dim=latent_dim, steps=steps)  # (B*n, latent_dim)
#     return z_rep.view(B, n, latent_dim)               # (B, n, latent_dim)

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
    beta_vae: BetaVAE,
    diffusion_model: LatentDiffusionUNet,
    diffusion_prior_steps: int = 200,
    device: str = "cuda",
    epochs: int = 100,
    save_every: int = 10,
    save_dir: str = "output",
):
    # wandb.watch(beta_vae)

    H = 30
    obs = torch.tensor(dataset["observations"], dtype=torch.float32)
    acts = torch.tensor(dataset["actions"], dtype=torch.float32)
    N = len(obs) - H
    sequences = [(obs[i:i+H], acts[i:i+H]) for i in range(N)]

    optimizer = torch.optim.Adam(beta_vae.parameters(), lr=5e-5)
    batch_size = 128

    for epoch in trange(epochs, desc="BetaVAE Training"):
        random.shuffle(sequences)
        losses = []

        for i in range(0, N, batch_size):
            batch = sequences[i:i+batch_size]
            s_seq = torch.stack([item[0] for item in batch]).to(device)  # (B, H, s_dim)
            a_seq = torch.stack([item[1] for item in batch]).to(device)  # (B, H, a_dim)
            loss = beta_vae.elbo_loss(s_seq, a_seq, diffusion_model=diffusion_model, beta=0.1, steps=diffusion_prior_steps)
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
    beta_vae: BetaVAE,
    diffusion_model: LatentDiffusionUNet,
    device="cuda",
    epochs: int = 450,
    save_every: int = 50,
    save_dir: str = "output",
):
    H = 30                                   # горизонт навыка (Maze2D)
    obs = torch.tensor(dataset["observations"], dtype=torch.float32, device=device)
    acts = torch.tensor(dataset["actions"],      dtype=torch.float32, device=device)
    N   = len(obs) - H                          # кол-во стартовых индексов

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)
    batch_size = 128

    for epoch in trange(epochs, desc="Diffusion Training"):
        idx_perm = torch.randperm(N, device=device)
        losses   = []

        for i in range(0, N, batch_size):
            idx = idx_perm[i : i + batch_size]          # стартовые позиции (B,)
            # -------- формируем последовательности длиной H ----------------
            s_seq = torch.stack([obs[j : j + H] for j in idx])   # (B, H, s_dim)
            a_seq = torch.stack([acts[j : j + H] for j in idx])  # (B, H, a_dim)
            s0    = s_seq[:, 0]                                   # текущий state (B, s_dim)

            with torch.no_grad():
                z, _, _ = beta_vae.encode(s_seq, a_seq)          # (B, latent)

            # -------- добавляем шум ----------------------------------------
            steps = 500
            t = torch.randint(1, steps + 1, (z.size(0),), device=device)
            noise = torch.randn_like(z)
            z_t   = z + noise * 0.1                              # простой гаусс-ноиз

            pred = diffusion_model(z_t, s0, t)                  # предсказываем z_0
            loss = F.mse_loss(pred, z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        if wandb and wandb.run:
            wandb.log({"diffusion/loss": avg_loss}, step=epoch)

        if epoch % save_every == 0 or epoch == epochs - 1:
            os.makedirs(f"{save_dir}/diffusion", exist_ok=True)
            torch.save(diffusion_model.state_dict(),
                       f"{save_dir}/diffusion/diffusion_model_epoch{epoch}.pt")


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
    diffusion_model= diffusion_model or LatentDiffusionUNet(latent_dim, state_dim).to(device)
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
    diffusion_model = LatentDiffusionUNet(latent_dim, state_dim).to(args.device)

    print("=== Training Beta-VAE ===")
    train_beta_vae(env, dataset, beta_vae, diffusion_model, device=args.device, epochs=100, save_dir=args.save_dir)

    # beta_vae = load_latest_checkpoint(beta_vae, "beta_vae", folder=args.save_dir + "/vae")

    print("=== Training Diffusion Model ===")
    train_diffusion_model(dataset, beta_vae, diffusion_model, device=args.device, epochs=450, save_dir=args.save_dir)

    # beta_vae.eval()
    # for p in beta_vae.parameters():
    #     p.requires_grad = False
    # diffusion_model = load_latest_checkpoint(diffusion_model, "diffusion_model", folder=args.save_dir + "/diffusion")

    # print("=== Starting DIAR Training ===")
    # train_diar(env=env, dataset=dataset, state_dim=state_dim, action_dim=action_dim,
    #            latent_dim=latent_dim, beta_vae=beta_vae, diffusion_model=diffusion_model,
    #            num_steps=100000, device=args.device, save_dir=args.save_dir)
