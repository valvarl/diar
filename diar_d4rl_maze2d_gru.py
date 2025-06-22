import argparse
import glob
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import d4rl

from tqdm import trange, tqdm
import wandb


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# ---------------------------------------------------------------------------
# Helper blocks
# ---------------------------------------------------------------------------

def mlp(
    in_dim: int,
    hidden: tuple[int, ...],
    out_dim: int
) -> nn.Sequential:
    """Helper to build GELU->LayerNorm MLP blocks."""
    layers: list[nn.Module] = []
    dims = (in_dim, *hidden)
    for i in range(len(hidden)):
        layers += [
            nn.Linear(dims[i], dims[i + 1]),
            nn.GELU(),
            nn.LayerNorm(dims[i + 1]),
        ]
    layers.append(nn.Linear(hidden[-1] if hidden else in_dim, out_dim))
    return nn.Sequential(*layers)

# ---------------------------------------------------------------------------
# β‑VAE (Bi‑GRU encoder, policy/state decoders, state prior)
# ---------------------------------------------------------------------------

class TransformerPrior(nn.Module):
    """
    200-step diffusion prior из β-VAE-фазы.
    Векторное (не 2-D) U-Net здесь заменён 2-блочным Transformer-encoder.
    """
    def __init__(self, latent_dim: int, state_dim: int, n_layers: int = 2,
                 n_heads: int = 4, d_model: int = 256, d_ff: int = 512):
        super().__init__()
        self.latent_dim = latent_dim

        # --- эмбеддинги ----------------------------------------------------
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.state_proj  = nn.Linear(state_dim , d_model)
        self.time_emb    = nn.Embedding(200, d_model)       # t ∈ [0,200]

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
        te    = self.time_emb(t - 1)

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
    def __init__(self, state_dim: int, action_dim: int,
                 latent_dim: int, steps_alpha: int = 200,
                 beta_max: float = 0.02):
        super().__init__()
        self.encoder = EncoderGRU(state_dim + action_dim, latent_dim)
        self.policy_dec = mlp(state_dim + latent_dim, (128, 128), action_dim)
        self.state_dec  = mlp(state_dim + latent_dim, (128, 128), state_dim)
        self.state_prior = mlp(state_dim, (128,), latent_dim * 2)
        self._precompute_alpha(steps_alpha, beta_max)

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
    
    def _precompute_alpha(self, steps: int, beta_max: float):
        beta_sched = torch.linspace(1e-4, beta_max, steps,
                                    dtype=torch.float32,
                                    device=next(self.parameters()).device)
        alpha_bar = torch.cumprod(1.0 - beta_sched, 0)     # (T,)
        self.register_buffer("alpha_bar", alpha_bar)       # ← уже на GPU / CPU

    def elbo_loss(
            self,
            s_seq: torch.Tensor,          # (B, H, state_dim)
            a_seq: torch.Tensor,          # (B, H, action_dim)
            beta: float       = 0.1,      # weight for KL-term
            lambda_state: float = 1.0,    # weight for state-reconstruction
            diff_prior: Optional[nn.Module] = None,   # transformer prior (ψ)
            gamma_snr: float = 5.0,       # Min-SNR-γ clipping
        ):
        """
        Evidence-Lower-Bound used in phase-1 (β-VAE training)

            L =  L_a                     (action reconstruction over horizon H)
            + λ_s · L_s                  (future-state reconstruction)
            + β     · KL(q||p)           (state-conditioned prior)
            +       L_4                  (diffusion-prior consistency)

        All reconstruction terms are summed over feature dimensions; KL and L_4
        are averaged by latent_dim so their scale is comparable across models.
        """
        # ------------------------------------------------------------
        B, H, _   = s_seq.shape
        device    = s_seq.device
        s_dim     = s_seq.size(-1)
        a_dim     = a_seq.size(-1)
        latent_dim= self.encoder.mu.out_features
        steps     = self.alpha_bar.numel()

        # -------------------- encode whole trajectory ----------------
        z, mu, logvar = self.encode(s_seq, a_seq)     # (B, latent_dim)
        s0, sH = s_seq[:, 0], s_seq[:, -1]            # current & future state

        # 1) action-sequence reconstruction  L_a
        s_rep = s_seq.reshape(-1, s_dim)              # (B·H, state_dim)
        z_rep = z.repeat_interleave(H, 0)             # (B·H, latent_dim)
        a_hat = self.decode_action(s_rep, z_rep)
        l_a   = F.mse_loss(
                    a_hat,
                    a_seq.reshape(-1, a_dim),
                    reduction='none'
                ).sum(-1)                             # sum over action dims
        l_a   = l_a.view(B, H).sum(-1) / a_dim        # normalise by |A|

        # 2) future-state reconstruction  L_s
        s_hat = self.decode_state(s0, z)
        l_s   = F.mse_loss(s_hat, sH, reduction='none').sum(-1) / s_dim # (B,)

        # 3) KL-divergence  KL(q||p)
        p_mu, p_logv = self.prior(s0)
        kl = -0.5 * (
                1 + logvar - p_logv
                - ((mu - p_mu).pow(2) + logvar.exp()) / p_logv.exp()
            ).sum(-1) / latent_dim                     # (B,)

        # 4) diffusion-prior consistency  L_4
        if diff_prior is not None:
            # ----- linear β_t  (t/T · β_max) -------------------------
            t_int  = torch.randint(1, steps + 1, (B,), device=device)  # (B,)
            alpha_bar_t = self.alpha_bar[t_int - 1]

            # ----- add noise ----------------------------------------
            eps   = torch.randn_like(z)
            z_t   = alpha_bar_t.sqrt().unsqueeze(1) * z + \
                    (1.0 - alpha_bar_t).sqrt().unsqueeze(1) * eps

            # ----- denoise with transformer prior -------------------
            z0_hat = diff_prior(z_t, s0, t_int)

            # ----- Min-SNR weighting --------------------------------
            # snr    = alpha_bar_t / (1.0 - alpha_bar_t)
            snr = alpha_bar_t / torch.clamp(1 - alpha_bar_t, 1e-5)
            weight = torch.minimum(snr, torch.full_like(snr, gamma_snr))

            l4 = (weight.unsqueeze(1) *
                (z0_hat - z).pow(2)).sum(-1) / latent_dim            # (B,)
        else:
            l4 = torch.zeros_like(kl)                                # (B,)

        # -------------------- final ELBO ----------------------------
        elbo = (l_a + lambda_state * l_s + beta * kl + l4).mean()
        return elbo

# ---------------------------------------------------------------------------
# Simple U‑Net‑like latent diffusion (down → up with residual blocks)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Two-layer residual MLP block
        y = x + 0.5 · LN(GELU(W2 · LN(GELU(W1 x))))
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(F.gelu(self.fc1(x)))
        h = self.ln2(F.gelu(self.fc2(h)))
        return x + 0.5 * h

class Rotary(nn.Module):
    """
    Rotary Positional Embedding (RoPE) helper for 2-D packed features.

    Accepts a tensor (B, D) and a step vector t: (B,) and returns the
    rotated tensor of the same shape.
    """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        freqs = torch.exp(-torch.arange(0, dim, 2) * math.log(theta) / dim)
        self.register_buffer("freqs", freqs)            # (D/2,)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, D)   – feature vector to rotate
            t : (B,)     – integer diffusion step 1 … T
        Returns:
            (B, D) – RoPE-rotated features
        """
        angles = t.float().unsqueeze(1) * self.freqs     # (B, D/2)
        sin, cos = angles.sin(), angles.cos()
        x1, x2 = x[:, 0::2], x[:, 1::2]                  # even / odd dims
        x_rot = torch.stack([x1 * cos - x2 * sin,        # apply complex rot
                             x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(1)                          # (B, D)

# --------------------------------------------------------------------------- #
#  Latent-Diffusion U-Net (vector version, §A.2)                              #
# --------------------------------------------------------------------------- #

class LatentDiffusionUNet(nn.Module):
    """
    One-dimensional (vector) U-Net used as latent diffusion model (ψ).

    Architecture:  (latent + state + t_emb) →
        down-128 → down-64 → ResBlock(64) → up-128 → up-latent_dim
    """
    def __init__(self, latent_dim: int, state_dim: int) -> None:
        super().__init__()
        in_dim = latent_dim + state_dim + 1              # + scalar t_emb

        self.down1 = mlp(in_dim,  (256,), 128)
        self.down2 = mlp(128,    (128,),  64)
        self.mid   = ResBlock(64)
        self.up1   = mlp(64,     (128,), 128)
        self.up2   = mlp(128,    (256,), latent_dim)

        self.rope  = Rotary(latent_dim + state_dim)

    def forward(
        self,
        z_t: torch.Tensor,       # (B, latent_dim)   – noised latent
        s:   torch.Tensor,       # (B, state_dim)    – current state
        t:   torch.Tensor        # (B,)              – diffusion step
    ) -> torch.Tensor:
        # concatenate latent and state, apply RoPE, append scalar step emb
        base = torch.cat([z_t, s], dim=-1)               # (B, D_no_t)
        x = self.rope(base, t)                           # rotary encoding
        x = torch.cat([x, t.float().unsqueeze(1) / 1000.0], dim=-1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m  = self.mid(d2)
        u1 = self.up1(m)
        out = self.up2(u1 + d1)                          # skip-connection
        return out

# --------------------------------------------------------------
# DDPM-инференс через обученный U-Net (используется после тренировки LDM)
# --------------------------------------------------------------
def ddpm_sample(model: nn.Module,               # обученный LatentDiffusionUNet
                s: torch.Tensor,                # (B , state_dim)
                latent_dim: int,
                steps: int = 500,               # T из табл. 5
                beta_max: float = 0.02):
    """
    DDPM-инференс c линейным β_t = t/T·β_max (§A.2, Algo 4, Tab 5)
    """
    B, device = s.size(0), s.device
    z = torch.randn(B, latent_dim, device=device)

    t_arr = torch.arange(steps, 0, -1, device=device)        # T…1
    beta  = (t_arr / steps) * beta_max                       # (T,)
    alpha = 1.0 - beta
    for t_idx, bt, at in zip(t_arr, beta, alpha):
        tt = torch.full((B,), int(t_idx), device=device)
        z0 = model(z, s, tt)                                 # предсказанный z_0
        z  = at * z0 + (1 - at) * z + torch.randn_like(z) * bt.sqrt()
    return z                                                 # (B, latent_dim)

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
def compute_v_loss(
    q_net_t,
    v_net,
    s: torch.Tensor,             # (B, state_dim)
    z_ddpm: torch.Tensor,        # (B, 500, L)
    z_data: torch.Tensor,        # (B, 32,  L)
    *,
    k: int        = 100,
    tau: float    = 0.9,
    lambda_v: float = 0.5,
    return_stats: bool = False,
):
    """
    L_V  из Eq.(6): expectile-loss между top-k Q-значениями и V(s).
    """
    B = s.size(0)
    z_all = torch.cat([z_ddpm, z_data], dim=1)         # (B, 532, L)
    n_all = z_all.size(1)

    with torch.no_grad():
        s_rep = s.unsqueeze(1).expand(-1, n_all, -1)   # (B,n_all, state_dim)
        q1, q2 = q_net_t(
            s_rep.reshape(-1, s.shape[-1]),
            z_all.reshape(-1, z_all.shape[-1]))
        q_vals = torch.min(q1, q2).view(B, n_all)      # (B,n_all)
        q_sel  = torch.topk(q_vals, k=min(k, n_all), dim=1).values  # (B,k)

    v_pred = v_net(s).unsqueeze(1)                     # (B,1)
    u  = q_sel - v_pred                                # (B,k)
    w  = torch.where(u > 0, tau, 1 - tau)
    loss = lambda_v * (w * u.pow(2)).mean()

    if return_stats:
        gap      = (q_sel.mean(dim=1) - v_pred.squeeze(1)).mean().detach()
        v_mean   = v_pred.mean().detach()
        q_mean   = q_sel.mean().detach()
        v_pred   = v_pred.detach()
        q_sel    = q_sel.detach()
        return loss, gap, v_mean, q_mean, v_pred, q_sel
    return loss


# ---------- Policy Execution ----------
@torch.no_grad()
def policy_execute(
    env:        gym.Env,
    q_net:      DoubleQNet,
    v_net:      ValueNet,
    beta_vae:   BetaVAE,
    diffusion:  LatentDiffusionUNet,
    *,
    latent_dim:     int   = 16,
    ddpm_steps:     int   = 10,      # «extra-steps = 5»  ⇒  H / ddpm_steps
    n_latents:      int   = 500,     # табл. 6
    reval_attempts: int   = 5,       # табл. 6 «extra steps 5»
    value_eps:      float = 0.2,     # допускаем небольшой прогноз-шум
    device:         str   = "cpu",
    max_actions:    Optional[int] = None
) -> float:
    """
    Инференс DIAR (Algo 2).

    1. Для каждого состояния:
       • генерируем `n_latents` кандидатов батчем через DDPM;
       • берём `argmax Q(s, z)` (c учётом clipped-double);
       • проверяем Adaptive Revaluation:  V(s_pred) ≥ V(s) - ε.
    2. Если условие не выполнено → пересэмплируем (до `reval_attempts` раз).
    3. Выполняем одно действие, повторяем цикл.
    """
    def augment(state_np: np.ndarray, goal_abs: np.ndarray) -> np.ndarray:
            """Добавляем относительную цель ⇒ (6,)."""
            return np.concatenate([state_np, goal_abs - state_np[:2]], axis=0)

    # ---------------------------------------------------------------------
    goal_abs = np.asarray(env.unwrapped._target, dtype=np.float32)
    state_np = env.reset()
    state_np = augment(state_np, goal_abs)
    done = False
    total_reward = 0.0
    action_count = 0
    max_actions = max_actions or env._max_episode_steps

    while not done and action_count < max_actions:
        # -------- текущее состояние → tensor -----------------------------
        state = torch.tensor(state_np, dtype=torch.float32,
                             device=device).unsqueeze(0)        # (1, s_dim)

        # -------- ищем «приемлемый» латент ------------------------------
        for _ in range(reval_attempts):
            # (B = n_latents) батч-семпл из DDPM
            z_pool = ddpm_sample(
                diffusion,
                state.repeat(n_latents, 1),                     # (n,s_dim)
                latent_dim,
                ddpm_steps)                                     # (n, L)

            # clipped-double Q(s,z)
            q1, q2  = q_net(state.repeat(n_latents, 1), z_pool)
            z_star  = z_pool[torch.min(q1, q2).argmax()].unsqueeze(0)  # (1,L)

            # восстановим действие и будущий стейт
            action_pred = beta_vae.decode_action(state, z_star)  # (1,a_dim)
            state_pred  = beta_vae.decode_state(state, z_star)   # (1,s_dim)

            # Adaptive Revaluation
            if v_net(state_pred) + value_eps >= v_net(state):
                break  # z_star принят

        # -------- выполняем действие ------------------------------------
        action_np   = action_pred.squeeze(0).cpu().detach().numpy()
        state_np, r, done, _ = env.step(action_np)
        state_np = augment(state_np, goal_abs)
        total_reward += r
        action_count += 1

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
        # --- Store dimensions for reconstruction during loading ---
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
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
        # Copy tensors to CPU and exclude device from metadata
        cpu_tensors = {}
        meta = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                cpu_tensors[k] = v.cpu()
            elif k != 'device':  # Exclude device from metadata
                meta[k] = v
                
        torch.save({'tensors': cpu_tensors, 'meta': meta}, path)

    @classmethod
    def load(cls, path: str, device="cpu"):
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt['meta']
        tensors = ckpt['tensors']
        
        # Extract initialization parameters
        init_params = {
            'capacity': meta['capacity'],
            'state_dim': meta['state_dim'],
            'action_dim': meta['action_dim'],
            'latent_dim': meta['latent_dim'],
            'alpha': meta['alpha'],
            'beta_start': meta['beta_start'],
            'beta_frames': meta['beta_frames'],
            'device': device
        }
        
        # Create buffer instance
        obj = cls(**init_params)
        
        # Load non-initialization attributes
        for k, v in meta.items():
            if k not in init_params and k != 'device':
                setattr(obj, k, v)
                
        # Load tensors to specified device
        for k, v in tensors.items():
            getattr(obj, k).copy_(v.to(device))
            
        return obj

# === phase-1: β-VAE  +  Transformer-prior ================================
def train_beta_vae(
    dataset: dict,
    valid: np.ndarray,                  # 1-D bool mask длиной (len(obs)-H)
    beta_vae: BetaVAE,
    diff_prior: TransformerPrior,
    *,
    horizon: int = 30,
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
                beta       = 0.1)

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
    valid: np.ndarray,                       # True — индекс i имеет «хвост» длиной H
    beta_vae: BetaVAE,                       # уже обученный (eval / frozen)
    diffusion_model: LatentDiffusionUNet,    # U-Net  ψ
    *,
    horizon:      int   = 30,                # H (Maze2D)
    steps:        int   = 500,               # T в табл. 5
    gamma_snr:    float = 5.0,               # Min-SNR-γ
    drop_prob:    float = 0.10,              # “drop-prob 0.1” из табл. 5
    device:       str   = "cuda",
    epochs:       int   = 450,
    batch_size:   int   = 128,
    lr:           float = 1e-4,
    save_every:   int   = 50,
    save_dir:     str   = "output/diffusion",
) -> None:
    """
    Фаза 2 (§A.2): обучаем латент-диффузионную модель ψ.

    • β-schedule: линейно 1e-4 → 0.02 (как в репо авторов)
    • Min-SNR-γ взвешивание (γ = 5)
    • Drop-prob 0.1 — Min-SNR dropout-trick: 10 % примеров пропускаем
      при расчёте градиента (см. GitHub-код авторов).
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- перенос датасета в GPU -------------------------------------------
    obs  = torch.tensor(dataset["observations"], dtype=torch.float32,
                        device=device)
    acts = torch.tensor(dataset["actions"],      dtype=torch.float32,
                        device=device)

    start_idx = torch.tensor(valid, device=device).nonzero(as_tuple=False).squeeze(1)
    N = start_idx.numel()

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)

    # -------- β-schedule (линейный) ---------------------------------------
    beta_sched = torch.linspace(1e-4, 0.02, steps, device=device)  # βₜ
    alpha      = 1.0 - beta_sched
    alpha_bar  = torch.cumprod(alpha, dim=0)                       # ᾱₜ

    for epoch in trange(epochs, desc="Latent-Diffusion"):
        perm = start_idx[torch.randperm(N, device=device)]
        loss_epoch: list[float] = []

        for i in range(0, N, batch_size):
            idx     = perm[i : i + batch_size]                     # (B,)
            s_seq   = torch.stack([obs [j : j + horizon] for j in idx])
            a_seq   = torch.stack([acts[j : j + horizon] for j in idx])
            s0      = s_seq[:, 0]                                  # (B, state_dim)

            with torch.no_grad():
                z, _, _ = beta_vae.encode(s_seq, a_seq)            # (B, latent_dim)

            # ---------- выбор случайного шага j ---------------------------
            j = torch.randint(1, steps + 1, (z.size(0),), device=device)  # (B,)
            a_bar_j = alpha_bar[j - 1].unsqueeze(1)                        # (B,1)

            eps  = torch.randn_like(z)
            z_j  = a_bar_j.sqrt() * z + (1 - a_bar_j).sqrt() * eps        # q(z_j|z_0)

            # ---------- предсказание z_0 ---------------------------------
            z0_pred = diffusion_model(z_j, s0, j)                         # (B, L)

            # ---------- Min-SNR-γ weighting ------------------------------
            snr     = a_bar_j / (1 - a_bar_j)                             # (B,1)
            weight  = torch.minimum(snr, torch.tensor(gamma_snr, device=device))

            # ---------- dropout-trick (табл. 5) --------------------------
            keep_mask = (torch.rand_like(weight) > drop_prob).float()     # (B,1)
            weight   = weight * keep_mask

            loss = ((weight * (z0_pred - z).pow(2)).sum(-1) /
                (keep_mask.mean() + 1e-8)).mean()        # scalar

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            optimizer.step()
            loss_epoch.append(loss.item())

        wandb.log({"diffusion/loss": sum(loss_epoch) / len(loss_epoch)}, step=epoch + 100)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(
                diffusion_model.state_dict(),
                f"{save_dir}/ldm_epoch{epoch}.pt"
            )

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

    gamma = 0.995
    obs, acts, rews, nexto = (dataset[k] for k in
                              ("observations","actions","rewards","next_observations"))

    if isinstance(valid_mask, np.ndarray):
        idx_all = np.nonzero(valid_mask)[0]
        idx_all = torch.from_numpy(idx_all)
    else:
        idx_all = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    
    for i in tqdm(idx_all.cpu()):
        # -------- траектория длиной H -----------------------------------
        s_seq = torch.tensor(obs [i : i+horizon], device=device, dtype=torch.float32)
        a_seq = torch.tensor(acts[i : i+horizon], device=device, dtype=torch.float32)
        s0    = s_seq[0]
        a0    = a_seq[0]
        s_H   = torch.tensor(nexto[i + horizon], device=device, dtype=torch.float32)

        # -------- γ-суммированная награда --------------------------------
        R = sum((gamma ** j) * float(rews[i + j]) for j in range(horizon))
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

# ---------------------------------------------------------------------
#  (3)  DIAR  –  Q / V  training
# ---------------------------------------------------------------------
def train_diar(
    env: gym.Env,
    *,
    dataset       : dict,
    valid_mask    : np.ndarray,
    beta_vae      : BetaVAE,                 # .eval(), frozen
    diffusion_model: LatentDiffusionUNet,    # .eval(), frozen
    state_dim     : int,
    action_dim    : int,
    latent_dim    : int   = 16,
    # ---------------- optimisation ------------------------------------
    num_steps     : int   = 100_000,
    horizon       : int   = 30,
    batch_size    : int   = 128,
    lr_q          : float = 5e-4,            # Tab-6
    lr_v          : float = 5e-4,
    step_lr_iters : int   = 50_000,          # TODO: 50 epochs
    device        : str   = "cuda",
    save_every    : int   = 10_000,
    save_dir      : str   = "output/diar",
    use_cache_rb  : bool  = True,
):
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- models ------------------------------------------
    beta_vae.eval();           diffusion_model.eval()
    q_net   = DoubleQNet(state_dim, latent_dim).to(device)
    v_net   = ValueNet(state_dim).to(device)
    q_tgt   = DoubleQNet(state_dim, latent_dim).to(device)
    v_tgt   = ValueNet(state_dim).to(device)
    q_tgt.load_state_dict(q_net.state_dict())
    v_tgt.load_state_dict(v_net.state_dict())

    opt_q = torch.optim.Adam(q_net.parameters(), lr=lr_q)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=lr_v)

    # ---------------- replay-buffer -----------------------------------
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

    # StepLR: каждые 50 **итераций** (см. табл. 6)
    sched_v = torch.optim.lr_scheduler.StepLR(opt_v, step_size=step_lr_iters,
                                              gamma=0.3)

    tau, gamma = 0.9, 0.995        # expectile / discount
    n_ddpm_lat = 500               # Tab-6 “# latent samples 500”

    pbar = trange(num_steps, desc="DIAR-train")
    for step in pbar:
        # =======================  Q-update  ==========================
        batch   = rb.sample(batch_size)
        s       = batch['state']                 # (B, s)
        z       = batch['latent']                # (B, L)
        r       = batch['reward']                # (B,)
        sn      = batch['next_state']            # (B, s)
        w       = batch['weights']               # (B,1)
        idx     = batch['indices']

        with torch.no_grad():
            v_tar = v_tgt(sn)                    # (B,)
            q_tar = r + gamma * v_tar            # (B,)

        q1, q2  = q_net(s, z)                    # (B,) ×2
        q_pred  = torch.min(q1, q2)              # (B,)
        td      = q_tar - q_pred                 # (B,)

        loss_q  = (w.squeeze(1) * td.pow(2)).mean()

        opt_q.zero_grad()
        loss_q.backward()
        opt_q.step()

        rb.update_priorities(idx, (td.abs() + 1e-6).detach())

        # =======================  V-update  ==========================
        with torch.no_grad():
            # ---- DDPM: батч-семпл 500 латентов одним вызовом ----------
            s_rep   = s.repeat_interleave(n_ddpm_lat, 0)          # (B·500, s)
            z_ddpm  = ddpm_sample(diffusion_model,
                                  s_rep,
                                  latent_dim,
                                  steps=10                       # “extra steps 5”
                                  ).view(batch_size,             # (B,500,L)
                                         n_ddpm_lat, latent_dim)

            # ---- 32 «dataset» латентов -------------------------------
            z_data = z.unsqueeze(1).expand(-1, 32, -1)            # (B,32,L)

        loss_v = compute_v_loss(
            q_net_t = q_tgt,
            v_net   = v_net,
            s       = s,
            z_ddpm  = z_ddpm,
            z_data  = z_data,
            k       = 100,
            tau     = tau,
            lambda_v= 0.5)

        opt_v.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        opt_v.step()
        sched_v.step()

        # ---------------- target soft-update -------------------------
        with torch.no_grad():
            for p, tp in zip(q_net.parameters(), q_tgt.parameters()):
                tp.mul_(0.995).add_(0.005 * p)
            for p, tp in zip(v_net.parameters(), v_tgt.parameters()):
                tp.mul_(0.995).add_(0.005 * p)

        # ---------------- periodic eval ------------------------------
        if step % 500 == 0:
            rew = policy_execute(env, q_net, v_net,
                                 beta_vae, diffusion_model,
                                 max_actions=150,  # horizon * reval_attempts
                                 device=device)
            wandb.log({"eval/reward": rew,
                       "loss/q":      loss_q.item(),
                       "loss/v":      loss_v.item()}, step=step + 550)
            pbar.set_postfix({"R": f"{rew:6.1f}"})

        # ---------------- checkpoints -------------------------------
        if step % save_every == 0 or step == num_steps - 1:
            torch.save(q_net.state_dict(), f"{save_dir}/q_net_{step}.pt")
            torch.save(v_net.state_dict(), f"{save_dir}/v_net_{step}.pt")


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
    obs = dset["observations"]                    # (N,4)
    goal = dset["infos/goal"]                     # (N,2)

    # вариант B (relative goal)
    obs_aug = np.concatenate([obs,
                            goal - obs[:, :2]], axis=1)   # (N,6)

    dset["observations"] = obs_aug

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
    train_beta_vae(
        dataset  = dset,
        valid    = valid_mask,
        beta_vae = beta_vae,
        diff_prior = t_prior,
        device   = args.device,
        epochs   = 100,
        save_dir = os.path.join(args.save_dir, "vae"))

    # beta_vae = load_latest_checkpoint(beta_vae, "beta_vae", os.path.join(args.save_dir, "vae"))
    # t_prior = load_latest_checkpoint(t_prior, "diff_prior", os.path.join(args.save_dir, "vae"))

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
    train_diar(
        env        = env,
        dataset    = dset,
        valid_mask = valid_mask,
        beta_vae         = beta_vae,        # .eval(), заморожен
        diffusion_model  = ldm_unet,        # .eval(), заморожен
        state_dim        = state_dim,
        action_dim       = action_dim,
        latent_dim       = args.latent_dim,
        num_steps        = 100_000,
        device     = args.device,
        save_dir   = os.path.join(args.save_dir, "diar"))

if __name__ == "__main__":
    main()
