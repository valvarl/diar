import argparse
import glob
import math
import os
import random
from typing import Optional, Tuple

import numpy as np
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

class TransformerPrior(nn.Module):
    """
    200-step diffusion prior из β-VAE-фазы.
    Векторное (не 2-D) U-Net здесь заменён 2-блочным Transformer-еncoder’ом.
    """
    def __init__(self, latent_dim: int, state_dim: int, n_layers: int = 2,
                 n_heads: int = 4, d_model: int = 256, d_ff: int = 512):
        super().__init__()
        self.latent_dim = latent_dim

        # --- эмбеддинги ----------------------------------------------------
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.state_proj  = nn.Linear(state_dim , d_model)
        self.time_emb    = nn.Embedding(201, d_model)       # t ∈ [0,200]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            activation="gelu", norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out = nn.Linear(d_model, latent_dim)

    def forward(self, z_t, s, t):
        """
        z_t : (B, latent_dim)  — зашумлённый латент
        s   : (B, state_dim)   — текущий state
        t   : (B,)             — integer timestep 1…200
        """
        # два токена: [state, latent]; добавляем time-embedding к **обоим**
        z_tok = self.latent_proj(z_t)
        s_tok = self.state_proj(s)
        te    = self.time_emb(t)

        tokens = torch.stack([s_tok + te, z_tok + te], dim=1)  # (B, 2, d_model)
        h = self.encoder(tokens)                               # (B, 2, d_model)

        z0_pred = self.out(h[:, 1])        # берём позицию «latent»
        return z0_pred

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

    def elbo_loss(
        self,
        s_seq: torch.Tensor,                # (B, H, s_dim)
        a_seq: torch.Tensor,                # (B, H, a_dim)
        beta: float = 0.1,
        lambda_state: float = 1.0,
        diff_prior: Optional[nn.Module] = None,   # ⟵ prior, участвует в градиентах
        steps: int = 200,
        gamma_snr: float = 5.0
    ):
        """
        L = L_a + λ_s·L_s + β·KL + L₄           (см. Eq.(1) + алгоритм 3)
        """
        B, H, _ = s_seq.shape
        device  = s_seq.device
        state_dim = s_seq.size(-1)
        action_dim = a_seq.size(-1)

        # --------------------------- Encode -----------------------------------
        z, mu, logvar = self.encode(s_seq, a_seq)       # (B, latent)
        s0, sH = s_seq[:, 0], s_seq[:, -1]

        # ----------- Reconstruction of the whole action sequence --------------
        s_rep = s_seq.reshape(-1, state_dim)            # (B·H, s_dim)
        z_rep = z.repeat_interleave(H, dim=0)           # (B·H, latent)
        a_pred = self.decode_action(s_rep, z_rep)
        l_a = F.mse_loss(a_pred,
                        a_seq.reshape(-1, action_dim),
                        reduction='none').sum(-1).view(B, H).sum(-1)   # (B,)

        # ------------------ Reconstruction of the future state ----------------
        s_pred = self.decode_state(s0, z)
        l_s    = F.mse_loss(s_pred, sH, reduction='none').sum(-1)       # (B,)

        # ------------------------------- KL -----------------------------------
        p_mu, p_logv = self.prior(s0)
        kl = -0.5 * (
            1 + logvar - p_logv - ((mu - p_mu) ** 2 + logvar.exp()) / p_logv.exp()
        ).sum(-1)                                                       # (B,)

        # ------------------ Diffusion-prior regulariser (L₄) ------------------
        if diff_prior is not None:
            # --- выбираем случайный шаг j для каждого объекта -----------------
            t_int = torch.randint(1, steps + 1, (B,), device=device)    # (B,)
            beta_t  = 1.0 / steps
            alpha   = 1.0 - beta_t
            alpha_bar_t = alpha ** t_int.float()                        # (B,)

            # --- добавляем гауссов шум ----------------------------------------
            eps   = torch.randn_like(z)                                 # (B, latent)
            z_t   = (alpha_bar_t.sqrt().unsqueeze(1) * z +
                    (1 - alpha_bar_t).sqrt().unsqueeze(1) * eps)       # (B, latent)

            # --- denoise ------------------------------------------------------
            z0_pred = diff_prior(z_t, s0, t_int)                   # (B, latent)

            # --- Min-SNR-γ вес -----------------------------------------------
            snr = alpha_bar_t / (1.0 - alpha_bar_t)                     # (B,)
            weight = torch.minimum(snr, torch.full_like(snr, gamma_snr))

            l4 = (weight.unsqueeze(1) *
                (z0_pred - z).pow(2)).sum(-1)                         # (B,)
        else:
            l4 = torch.zeros_like(l_a)                                  # (B,)

        # ----------------------------- ELBO -----------------------------------
        loss = (l_a + lambda_state * l_s + beta * kl + l4).mean()
        return loss


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
def compute_v_loss(q_net_t, v_net, s,
                   z_ddpm, z_data,
                   k=100, tau=0.9, lambda_v=0.5,  # λ_v = 0.5 как в цикле
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
    def __init__(
        self, 
        capacity, 
        state_dim, 
        action_dim,
        latent_dim, 
        device,
        alpha=0.7, 
        beta_start=0.3, 
        beta_frames=100_000
    ):
        self.capacity, self.device = capacity, device
        self.alpha, self.beta_start, self.beta_frames = alpha, beta_start, beta_frames
        self.frame = 1
        # --- cyclic index --------------------------------------------------
        self.ptr, self.size = 0, 0
        # --- data ----------------------------------------------------------
        self.states      = torch.zeros((capacity, state_dim ), device=device)
        self.actions     = torch.zeros((capacity, action_dim), device=device)
        self.rewards     = torch.zeros((capacity,),          device=device)
        self.next_states = torch.zeros((capacity, state_dim ), device=device)
        self.latents     = torch.zeros((capacity, latent_dim), device=device)
        self.priorities  = torch.ones ((capacity,),          device=device)

    # --------------------------- API --------------------------------------
    def add(self, s, a, r, s_next, z):
        self.states     [self.ptr] = s
        self.actions    [self.ptr] = a
        self.rewards    [self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.latents    [self.ptr] = z
        self.priorities [self.ptr] = self.priorities.max() if self.size else 1.0

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        probs = (self.priorities if self.size == self.capacity
                 else self.priorities[: self.ptr]) ** self.alpha
        probs /= probs.sum()

        idx = torch.multinomial(probs, batch_size, replacement=False)
        beta = min(self.beta_start + (1 - self.beta_start) *
                   (self.frame / self.beta_frames), 1.0)
        self.frame += 1

        w = (self.capacity * probs[idx]) ** (-beta)
        w /= w.max()

        return dict(
            state      = self.states[idx],
            action     = self.actions[idx],
            reward     = self.rewards[idx],
            next_state = self.next_states[idx],
            latent     = self.latents[idx],
            weights    = w.unsqueeze(1).to(self.device),
            indices    = idx)

    def update_priorities(self, idx, prio):
        self.priorities[idx] = prio

    # --------------------------- I/O --------------------------------------
    def save(self, path: str):
        cpu_tensors = {k: v.cpu() for k, v in self.__dict__.items()
                       if isinstance(v, torch.Tensor)}
        meta = {k: v for k, v in self.__dict__.items()
                if not isinstance(v, torch.Tensor)}
        torch.save({'tensors': cpu_tensors, 'meta': meta}, path)

    @classmethod
    def load(cls, path: str, device="cpu"):
        ckpt = torch.load(path, map_location="cpu")
        obj  = cls(**{**ckpt['meta'],
                      'device': device})          # capacity, dims, alpha …
        for k, v in ckpt['tensors'].items():
            setattr(obj, k, v.to(device))
        return obj

# === phase-1: β-VAE  +  Transformer-prior ================================
def train_beta_vae(
    dataset: dict,
    valid: np.ndarray,                  # 1-D bool mask длиной (len(obs)-H)
    beta_vae: BetaVAE,
    diff_prior: TransformerPrior,
    *,
    horizon: int = 30,
    steps: int = 200,                   # diffusion-prior steps
    device: str = "cuda",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 5e-5,
    save_every: int = 10,
    save_dir: str = "output/vae",
):
    """Совместное обучение β-VAE (θ) и TransformerPrior (ψ)."""
    os.makedirs(save_dir, exist_ok=True)

    # --- тензоры на нужном девайсе ----------------------------------------
    obs  = torch.tensor(dataset["observations"], dtype=torch.float32,
                        device=device)
    acts = torch.tensor(dataset["actions"],      dtype=torch.float32,
                        device=device)

    start_idx = torch.nonzero(torch.tensor(valid, device=device), as_tuple=False).squeeze(1)
    N = len(start_idx)

    opt = torch.optim.Adam(
        list(beta_vae.parameters()) + list(diff_prior.parameters()),
        lr=lr)

    for epoch in trange(epochs, desc="β-VAE training"):
        # случайная перестановка валидных индексов
        perm = start_idx[torch.randperm(N, device=device)]
        losses = []

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]           # (B,)
            # --- формируем батч длиной horizon ---------------------------
            s_seq = torch.stack([obs [j : j + horizon] for j in idx])
            a_seq = torch.stack([acts[j : j + horizon] for j in idx])

            loss = beta_vae.elbo_loss(
                s_seq, a_seq,
                diff_prior = diff_prior,
                beta       = 0.1,
                steps      = steps)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(beta_vae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(diff_prior.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        wandb.log({"beta_vae/elbo": np.mean(losses)}, step=epoch)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(beta_vae.state_dict(),
                       f"{save_dir}/beta_vae_epoch{epoch}.pt")
            torch.save(diff_prior.state_dict(),
                       f"{save_dir}/diff_prior_epoch{epoch}.pt")


def train_diffusion_model(
    dataset: dict,
    valid: np.ndarray,                       # маска «хвост H не пересекает done»
    beta_vae: BetaVAE,                       # уже обученный, .eval() & frozen
    diffusion_model: LatentDiffusionUNet,    # U-Net (ψ)
    *,
    horizon: int = 30,
    steps:   int = 500,                      # diffusion T
    gamma_snr: float = 5.0,
    device: str = "cuda",
    epochs: int = 450,
    batch_size: int = 128,
    lr: float = 1e-4,
    save_every: int = 50,
    save_dir: str = "output/diffusion",
):
    os.makedirs(save_dir, exist_ok=True)

    obs  = torch.tensor(dataset["observations"], dtype=torch.float32,
                        device=device)
    acts = torch.tensor(dataset["actions"],      dtype=torch.float32,
                        device=device)

    start_idx = torch.nonzero(torch.tensor(valid, device=device), as_tuple=False).squeeze(1)
    N = len(start_idx)

    opt = torch.optim.Adam(diffusion_model.parameters(), lr=lr)

    # ------- pre-compute variance schedule ---------------------------------
    beta_sched  = torch.linspace(1e-4, 0.02, steps, device=device)  # βₜ
    alpha       = 1.0 - beta_sched
    alpha_bar   = torch.cumprod(alpha, dim=0)                       # ᾱₜ

    for epoch in trange(epochs, desc="Latent-Diffusion"):
        perm = start_idx[torch.randperm(N, device=device)]
        losses = []

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            s_seq = torch.stack([obs [j : j + horizon] for j in idx])
            a_seq = torch.stack([acts[j : j + horizon] for j in idx])
            s0    = s_seq[:, 0]

            with torch.no_grad():
                z, _, _ = beta_vae.encode(s_seq, a_seq)            # (B, L)

            # --------- sample j and noised latent -------------------------
            j = torch.randint(1, steps + 1, (z.size(0),), device=device)
            a_bar_j = alpha_bar[j - 1].unsqueeze(1)                # (B,1)
            eps  = torch.randn_like(z)
            z_j  = (a_bar_j.sqrt() * z +
                    (1 - a_bar_j).sqrt() * eps)

            # --------- predict, weigh by Min-SNR --------------------------
            z0_pred = diffusion_model(z_j, s0, j)                  # (B, L)
            snr = a_bar_j / (1 - a_bar_j)                          # (B,1)
            weight = torch.minimum(snr, torch.tensor(gamma_snr, device=device))
            loss = (weight * (z0_pred - z).pow(2)).sum(-1).mean()  # scalar

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        wandb.log({"diffusion/loss": np.mean(losses)}, step=epoch)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(diffusion_model.state_dict(),
                       f"{save_dir}/ldm_epoch{epoch}.pt")


# ---------------------------------------------------------------------
#   DIAR · шаг 3  –  Q- / V-обучение  (Maze2D horizon = 30)
# ---------------------------------------------------------------------
def build_replay_buffer(dataset, valid_mask, horizon,
                        beta_vae, state_dim, action_dim, latent_dim, 
                        device, *,
                        cache_path: str = "replay_buffer.pt",
                        use_cache: bool = True):
    if use_cache and os.path.exists(cache_path):
        print(f"→ загружаю буфер из {cache_path}")
        return PrioritizedReplayBuffer.load(cache_path, device)

    print("→ строю буфер заново …")
    cap = int(valid_mask.sum())
    rb = PrioritizedReplayBuffer(cap, state_dim, action_dim,
                                 latent_dim, device,
                                 alpha=0.7, beta_start=0.3,
                                 beta_frames=100_000)

    γ = 0.995
    obs, acts, rews, nexto = (dataset[k] for k in
                              ("observations","actions","rewards","next_observations"))

    idx_all = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    for i in idx_all.cpu():
        # -------- траектория длиной H -----------------------------------
        s_seq = torch.tensor(obs [i : i+horizon], device=device, dtype=torch.float32)
        a_seq = torch.tensor(acts[i : i+horizon], device=device, dtype=torch.float32)
        s0    = s_seq[0]
        a0    = a_seq[0]
        s_H   = torch.tensor(nexto[i + horizon], device=device, dtype=torch.float32)

        # -------- γ-суммированная награда --------------------------------
        R = sum((γ ** j) * float(rews[i + j]) for j in range(horizon))
        R = torch.tensor(R, dtype=torch.float32, device=device)

        # -------- латент zₜ ---------------------------------------------
        with torch.no_grad():
            z, _, _ = beta_vae.encode(s_seq.unsqueeze(0), a_seq.unsqueeze(0))
            z = z.squeeze(0)                               # (latent_dim,)

        rb.add(s0, a0, R, s_H, z)          # ← добавляем z

    if use_cache:
        rb.save(cache_path)
        print(f"→ буфер сохранён в {cache_path}")
    return rb
# ---------------------------------------------------------------------

def train_diar(
    env,                                         # ← единственный позиционный
    *,                                           # ─────────────────────────
    dataset:  dict,
    valid_mask: np.ndarray,                      # = valid_mask
    beta_vae: BetaVAE,                           # заморожен, eval()
    diffusion_model: LatentDiffusionUNet,        # U-Net, eval()
    state_dim:  int,
    action_dim: int,
    latent_dim: int           = 16,
    # ------------ тренинг-параметры ---------------------------------------
    num_steps:  int          = 100_000,          # ⇆ epochs у diffusion-шагa
    horizon:    int          = 30,
    batch_size: int          = 128,
    lr_q:       float        = 5e-4,             # как в табл. 6
    lr_v:       float        = 1e-4,
    step_lr_ep: int          = 50,
    device:     str          = "cuda",
    save_every: int          = 10_000,
    save_dir:   str          = "output/diar",
    use_cache_rb: bool       = True,
):
    os.makedirs(save_dir, exist_ok=True)

    # -------- модели ------------------------------------------------------
    beta_vae = beta_vae or BetaVAE(state_dim, action_dim, latent_dim).to(device).eval()
    diffusion_mod = diffusion_model or LatentDiffusionUNet(
                        latent_dim, state_dim).to(device).eval()

    q_net   = DoubleQNet(state_dim, latent_dim).to(device)
    v_net   = ValueNet(state_dim).to(device)
    q_tgt   = DoubleQNet(state_dim, latent_dim).to(device)
    v_tgt   = ValueNet(state_dim).to(device)
    q_tgt.load_state_dict(q_net.state_dict())
    v_tgt.load_state_dict(v_net.state_dict())

    opt_q = torch.optim.Adam(q_net.parameters(), lr=lr_q)  # табл. 6
    opt_v = torch.optim.Adam(v_net.parameters(), lr=lr_v, weight_decay=1e-4)

    # -------- реплэй-буфер -----------------------------------------------
    rb = build_replay_buffer(
        dataset      = dataset,
        valid_mask   = valid_mask,
        horizon      = horizon,
        beta_vae     = beta_vae,
        state_dim    = state_dim,
        action_dim   = action_dim,
        latent_dim   = latent_dim,
        device       = device,
        cache_path   = os.path.join(save_dir, "replay_buffer.pt"),
        use_cache    = use_cache_rb)

    # ---- StepLR на V-сеть: шаг = 50 итераций  ----------------------------
    steps_per_epoch = math.ceil(rb.size / batch_size)      # или len(valid_mask)//128
    sched_v = torch.optim.lr_scheduler.StepLR(
        opt_v, step_size = steps_per_epoch * step_lr_ep, gamma = 0.3)

    τ = 0.9                    # фиксированный expectile  (табл. 6)
    γ = 0.995                  # discount

    pbar = trange(num_steps, desc="DIAR-Q/V")
    for step in pbar:
        # ------------------------- Q-update ------------------------------
        batch = rb.sample(batch_size)
        s  = batch['state']
        z   = batch['latent']                  # ← готовый zₜ
        r  = batch['reward'].unsqueeze(1)          # (B,1)
        sn = batch['next_state']
        w  = batch['weights']
        idx= batch['indices']

        #  latent из β-VAE
        with torch.no_grad():
            v_target = v_tgt(sn)
            q_tar    = r + γ * v_target

        q1, q2 = q_net(s, z)
        q_pred = torch.min(q1, q2)

        td = (q_tar - q_pred).detach()
        loss_q = (w * td.pow(2)).mean()

        opt_q.zero_grad()
        loss_q.backward()
        opt_q.step()

        rb.update_priorities(idx, td.abs() + 1e-6)

        # ------------------------- V-update ------------------------------
        with torch.no_grad():
            # 500 DDPM-латентов
            z_ddpm = torch.stack(
                [ddpm_sample(diffusion_mod, s, latent_dim, steps=10)   # (B,L)
                for _ in range(500)], dim=0).permute(1,0,2)           # → (B,500,L)

            # 32 латента из β-VAE
            z_data = z.unsqueeze(1).expand(-1, 32, -1)                # (B,32,L)

        loss_v = compute_v_loss(
            q_tgt, v_net, s,
            z_ddpm, z_data,
            k=100, tau=τ, lambda_v=0.5)

        opt_v.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        opt_v.step()
        sched_v.step()

        # ------------------------- target soft-update --------------------
        with torch.no_grad():
            for p, tp in zip(q_net.parameters(), q_tgt.parameters()):
                tp.data.mul_(0.995).add_(0.005 * p.data)
            for p, tp in zip(v_net.parameters(), v_tgt.parameters()):
                tp.data.mul_(0.995).add_(0.005 * p.data)

        # ------------------ периодическая проверка -----------------------
        if step % 500 == 0:
            rew = policy_execute(
                env, q_net, v_net, beta_vae, diffusion_mod,
                device=device)
            wandb.log({"eval/reward": rew,
                       "loss/q": loss_q.item(),
                       "loss/v": loss_v.item()}, step=step)
            pbar.set_postfix({"rew": rew})

        # --------------- чекпоинт ---------------------------------------
        if step % save_every == 0 or step == num_steps - 1:
            torch.save(q_net.state_dict(),
                       f"{save_dir}/q_net_{step}.pt")
            torch.save(v_net.state_dict(),
                       f"{save_dir}/v_net_{step}.pt")


def load_latest_checkpoint(model, name, folder="output"):
    checkpoints = glob.glob(os.path.join(folder, f"{name}_*.pt"))
    if not checkpoints:
        print(f"No checkpoint found for {name}")
        return model
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Loading {name} from {latest}")
    model.load_state_dict(torch.load(latest))
    return model


def build_next_obs(dset, horizon):
    """Создаём dset["next_observations"], не переходя границы эпизода."""
    obs = dset["observations"]
    terms = dset["terminals"].astype(bool) | dset["timeouts"].astype(bool)
    next_obs = np.empty_like(obs)
    next_obs[:-1] = obs[1:]
    next_obs[-1]  = obs[-1]                         # dummy в конце

    # обнуляем переходы через done
    next_obs[terms] = obs[terms]

    dset["next_observations"] = next_obs

    # флаг — какие индексы ещё имеют H-шаговый «хвост» в том же эпизоде
    valid_mask = np.ones(len(obs) - horizon, dtype=bool)
    for i in range(len(valid_mask)):
        if terms[i : i + horizon].any():
            valid_mask[i] = False
    return valid_mask

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="maze2d-umaze-v1")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--save_dir",   default="output")
    parser.add_argument("--latent_dim", default=16, type=int)
    parser.add_argument("--horizon",    default=30, type=int)
    args = parser.parse_args()

    wandb.init(project="diar", name=f"diar_{args.env}")

    # --------- папки --------------------------------------------------------
    for sub in ("vae", "diffusion", "diar"):
        os.makedirs(os.path.join(args.save_dir, sub), exist_ok=True)

    # --------- данные -------------------------------------------------------
    env = gym.make(args.env)
    dset = env.get_dataset()
    if "next_observations" not in dset:
        print("→ генерируем next_observations ...")
        valid_mask = build_next_obs(dset, args.horizon)
    else:
        valid_mask = np.ones(len(dset["observations"]) - args.horizon, bool)

    state_dim  = dset["observations"].shape[1]
    action_dim = dset["actions"].shape[1]

    # --------- модели -------------------------------------------------------
    beta_vae   = BetaVAE(state_dim, action_dim, args.latent_dim).to(args.device)
    t_prior    = TransformerPrior(args.latent_dim, state_dim).to(args.device)
    ldm_unet   = LatentDiffusionUNet(args.latent_dim, state_dim).to(args.device)

    # -------------------- (1) β-VAE + Transformer prior --------------------
    # train_beta_vae(
    #     dataset  = dset,
    #     valid    = valid_mask,
    #     beta_vae = beta_vae,
    #     diff_prior = t_prior,
    #     device   = args.device,
    #     epochs   = 100,
    #     save_dir = os.path.join(args.save_dir, "vae"))

    beta_vae = load_latest_checkpoint(beta_vae, "beta_vae", os.path.join(args.save_dir, "vae"))
    t_prior = load_latest_checkpoint(t_prior, "diff_prior", os.path.join(args.save_dir, "vae"))

    # замораживаем
    beta_vae.eval()
    for p in beta_vae.parameters(): 
        p.requires_grad = False
    t_prior.eval()
    for p in t_prior.parameters(): 
        p.requires_grad = False

    # -------------------- (2) Latent diffusion U-Net -----------------------
    train_diffusion_model(
        dataset  = dset,
        valid    = valid_mask,
        beta_vae = beta_vae,
        diffusion_model = ldm_unet,
        device   = args.device,
        epochs   = 450,
        save_dir = os.path.join(args.save_dir, "diffusion"))
    
    # ldm_unet = load_latest_checkpoint(ldm_unet, "ldm", os.path.join(args.save_dir, "diffusion"))
    
    ldm_unet.eval()
    for p in ldm_unet.parameters():
        p.requires_grad = False

    # # -------------------- (3) DIAR Q / V training --------------------------
    # train_diar(
    #     env        = env,
    #     dataset    = dset,
    #     valid_mask = valid_mask,
    #     beta_vae         = beta_vae,        # .eval(), заморожен
    #     diffusion_model  = ldm_unet,        # .eval(), заморожен
    #     state_dim        = state_dim,
    #     action_dim       = action_dim,
    #     latent_dim       = args.latent_dim,
    #     num_steps        = 100_000,
    #     device     = args.device,
    #     save_dir   = os.path.join(args.save_dir, "diar"))

if __name__ == "__main__":
    main()
