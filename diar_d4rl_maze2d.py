import argparse
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

    def elbo_loss(self, s, a, beta=0.1, lambda_state=1.0, gamma=1.0, diffusion_model=None, steps=10):
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
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.v = MLP(state_dim, [256, 256], 1)

    def forward(self, s):
        return self.v(s).squeeze(-1)

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

def compute_v_loss(q_net_target, v_net, s, z_samples, tau=0.9):
    with torch.no_grad():
        s_expand = s.unsqueeze(1).expand(-1, z_samples.size(1), -1)
        q1, q2 = q_net_target(
            s_expand.reshape(-1, s.shape[-1]), z_samples.reshape(-1, z_samples.shape[-1])
        )
        q_vals = torch.min(q1, q2).view(s.shape[0], -1)
        q_max = q_vals.max(dim=1)[0]
    v_pred = v_net(s)
    return expectile_loss(q_max, v_pred, tau)

# ---------- Training Loop Step ----------
def train_diar_step(
    replay_buffer,
    q_net,
    v_net,
    q_net_target,
    v_net_target,
    diffusion_model,
    beta_vae,
    optimizer_q,
    optimizer_v,
    gamma=0.99,
    tau=0.9,
    latent_dim=16,
    ddpm_steps=10,
    device="cpu"
):
    batch = replay_buffer.sample()
    s = batch['state'].to(device)
    a = batch['action'].to(device)
    r = batch['reward'].to(device)
    s_next = batch['next_state'].to(device)

    z, _, _ = beta_vae.encode(s, a)

    loss_q = compute_q_loss(q_net_target, v_net_target, s, z, r, s_next, gamma)
    optimizer_q.zero_grad()
    loss_q.backward()
    optimizer_q.step()

    with torch.no_grad():
        z_samples = torch.stack([
            ddpm_sample(diffusion_model, s, latent_dim, ddpm_steps)
            for _ in range(5)
        ], dim=1)

    loss_v = compute_v_loss(q_net_target, v_net, s, z_samples, tau)
    optimizer_v.zero_grad()
    loss_v.backward()
    optimizer_v.step()

    with torch.no_grad():
        for p, tp in zip(q_net.parameters(), q_net_target.parameters()):
            tp.data.mul_(0.995).add_(0.005 * p.data)
        for p, tp in zip(v_net.parameters(), v_net_target.parameters()):
            tp.data.mul_(0.995).add_(0.005 * p.data)

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

            if v_net(s) <= v_net(s_pred):
                break

        a_np = a.squeeze(0).detach().cpu().numpy()
        next_state, reward, done, info = env.step(a_np)
        state = next_state
        total_reward += reward
        t += 1

    return total_reward

# === Replay Buffer for D4RL Dataset ===
class ReplayBuffer:
    def __init__(self, dataset, device):
        self.device = device
        self.states = torch.tensor(dataset['observations'], dtype=torch.float32).to(device)
        self.actions = torch.tensor(dataset['actions'], dtype=torch.float32).to(device)
        self.next_states = torch.tensor(dataset['next_observations'], dtype=torch.float32).to(device)
        self.rewards = torch.tensor(dataset['rewards'], dtype=torch.float32).to(device)

    def sample(self, batch_size=256):
        idx = torch.randint(0, self.states.size(0), (batch_size,))
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
        }

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
    wandb.watch(beta_vae)

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
    wandb.watch(diffusion_model)

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

# === Training Script ===
def train_diar(
    env,
    dataset,
    state_dim,
    action_dim,
    latent_dim=16,
    beta_vae=None,
    diffusion_model=None,
    num_steps=100_000,
    device="cuda",
    save_every=10000,
    save_dir="output",
):
    replay_buffer = ReplayBuffer(dataset, device)

    beta_vae = beta_vae or BetaVAE(state_dim, action_dim, latent_dim).to(device)
    diffusion_model = diffusion_model or LatentDiffusionModel(latent_dim, state_dim).to(device)
    q_net = DoubleQNet(state_dim, latent_dim).to(device)
    v_net = ValueNet(state_dim).to(device)
    q_net_target = DoubleQNet(state_dim, latent_dim).to(device)
    v_net_target = ValueNet(state_dim).to(device)

    q_net_target.load_state_dict(q_net.state_dict())
    v_net_target.load_state_dict(v_net.state_dict())

    optimizer_q = torch.optim.Adam(q_net.parameters(), lr=5e-4)
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=5e-4)

    wandb.watch([q_net, v_net])

    pbar = trange(num_steps, desc="DIAR Training")
    for step in pbar:
        train_diar_step(
            replay_buffer, q_net, v_net, q_net_target, v_net_target,
            diffusion_model, beta_vae, optimizer_q, optimizer_v,
            gamma=0.995, tau=0.9, latent_dim=latent_dim,
            ddpm_steps=10, device=device
        )

        if step % 500 == 0:
            reward = policy_execute(env, q_net, v_net, beta_vae, diffusion_model, device=device)
            wandb.log({"eval/reward": reward}, step=step)
            pbar.set_postfix({"eval_reward": reward})

        if step % save_every == 0 or step == num_steps - 1:
            torch.save(q_net.state_dict(), f"{save_dir}/diar/q_net_step{step}.pt")
            torch.save(v_net.state_dict(), f"{save_dir}/diar/v_net_step{step}.pt")
            torch.save(beta_vae.state_dict(), f"{save_dir}/diar/beta_vae_step{step}.pt")
            torch.save(diffusion_model.state_dict(), f"{save_dir}/diar/diffusion_model_step{step}.pt")


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

    print("=== Training Beta-VAE ===")
    train_beta_vae(env, dataset, beta_vae, diffusion_model, device=args.device, epochs=100, save_dir=args.save_dir)

    print("=== Training Diffusion Model ===")
    train_diffusion_model(dataset, beta_vae, diffusion_model, device=args.device, epochs=450, save_dir=args.save_dir)

    print("=== Starting DIAR Training ===")
    train_diar(env=env, dataset=dataset, state_dim=state_dim, action_dim=action_dim,
               latent_dim=latent_dim, beta_vae=beta_vae, diffusion_model=diffusion_model,
               num_steps=100000, device=args.device, save_dir=args.save_dir)
