# =============================================================================
# DIAR: Diffusion-model-guided Implicit Q-learning with Adaptive Revaluation
# -----------------------------------------------------------------------------
# Implementation of DIAR (Park et al. 2024) in PyTorch.
# 
# Modules:
#   [1] Utilities and Helpers
#   [2] β-VAE Phase (encoding trajectories into latent space)
#   [3] Diffusion Prior Model (Transformer-based for β-VAE)
#   [4] Latent Diffusion Model (U-Net for trajectory generation)
#   [5] Q- and Value Networks (critic and baseline)
#   [6] Policy Execution (Algorithm 2 from the paper)
#   [7] Replay Buffer with Prioritized Experience Replay
#   [8] Training Loops:
#         8.1 β-VAE + Transformer Prior
#         8.2 Latent Diffusion Model
#         8.3 DIAR Training (Q- and V-networks)
#   [9] Utilities: checkpointing, next_obs builder
#   [10] Main training entrypoint
# =============================================================================

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
from torch.utils.data import Dataset, TensorDataset, DataLoader

from tqdm import trange, tqdm
import wandb


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# -----------------------------------------------------------------------------
# [1] Utilities and MLP Builder
# -----------------------------------------------------------------------------

def mlp(
    in_dim: int,
    hidden: tuple[int, ...],
    out_dim: int
) -> nn.Sequential:
    """
    Builds a multi-layer perceptron (MLP) with GELU activation and LayerNorm.

    Each hidden layer is followed by:
        - GELU non-linearity
        - LayerNorm

    Final layer is a linear output layer with no activation.

    Args:
        in_dim (int): Dimension of input features.
        hidden (tuple[int]): Sizes of hidden layers.
        out_dim (int): Dimension of output features.

    Returns:
        nn.Sequential: MLP model.
    """
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

# -----------------------------------------------------------------------------
# [2] β‑VAE Phase: Encoder, Decoders, Prior, Loss (ELBO)
# -----------------------------------------------------------------------------

class EncoderGRU(nn.Module):
    """
    Bidirectional GRU encoder for encoding (state, action) sequences into a latent vector.

    Input:
        seq: Tensor of shape (B, H, in_dim)
    
    Output:
        z:      Sampled latent vector        (B, latent_dim)
        mu:     Mean of latent distribution  (B, latent_dim)
        logvar: Log-variance of latent dist. (B, latent_dim)
    """
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.rnn = nn.GRU(in_dim, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, seq):
        _, h = self.rnn(seq)  # h: (4, B, 128)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 256)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

class BetaVAE(nn.Module):
    """
    Beta-VAE architecture for learning latent skill representations.

    Components:
        - Encoder: Bi-GRU over (s,a) trajectory → latent z
        - Policy Decoder: reconstruct a_seq from (s, z)
        - State Decoder: predict future s_H from (s_0, z)
        - State Prior: Gaussian prior p(z | s_0)
        - Diffusion Prior Consistency (optional): L_4
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        steps_alpha: int = 200,
        beta_max: float = 0.02
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = EncoderGRU(state_dim + action_dim, latent_dim)
        self.policy_dec = mlp(state_dim + latent_dim, (128, 128), action_dim)
        self.state_dec  = mlp(state_dim + latent_dim, (128, 128), state_dim)
        self.state_prior = mlp(state_dim, (128,), latent_dim * 2)
        
        self._precompute_alpha(steps_alpha, beta_max)

    def _precompute_alpha(self, steps: int, beta_max: float):
        beta_sched = torch.linspace(1e-4, beta_max, steps,
                                    dtype=torch.float32,
                                    device=next(self.parameters()).device)
        alpha_bar = torch.cumprod(1.0 - beta_sched, 0)
        self.register_buffer("alpha_bar", alpha_bar)

    def encode(self, s_seq, a_seq):
        """
        Encode trajectory (s_seq, a_seq) → latent z.
        """
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
        s_seq: torch.Tensor,
        a_seq: torch.Tensor,
        beta: float = 0.1,
        lambda_state: float = 1.0,
        diff_prior: Optional[nn.Module] = None,
        gamma_snr: float = 5.0
    ):
        """
        Computes the full β-VAE ELBO loss.

        Loss = L_a (action reconstruction)
             + L_s (future state prediction)
             + β · KL(q||p) (prior regularization)
             + L_4 (diffusion prior consistency, optional)
        """
        B, H, _ = s_seq.shape
        device = s_seq.device
        s_dim, a_dim = s_seq.size(-1), a_seq.size(-1)
        steps = self.alpha_bar.numel()

        # -------- Encode trajectory
        z, mu, logvar = self.encode(s_seq, a_seq)
        s0, sH = s_seq[:, 0], s_seq[:, -1]

        # -------- L_a: action sequence reconstruction
        s_rep = s_seq.reshape(-1, s_dim)
        z_rep = z.repeat_interleave(H, 0)
        a_hat = self.decode_action(s_rep, z_rep)
        l_a = F.mse_loss(a_hat, a_seq.reshape(-1, a_dim), reduction='none').sum(-1)
        l_a = l_a.view(B, H).sum(-1) / a_dim

        # -------- L_s: future state prediction
        s_hat = self.decode_state(s0, z)
        l_s = F.mse_loss(s_hat, sH, reduction='none').sum(-1) / s_dim

        # -------- KL divergence: q(z|s,a) || p(z|s)
        p_mu, p_logv = self.prior(s0)
        kl = -0.5 * (
            1 + logvar - p_logv
            - ((mu - p_mu).pow(2) + logvar.exp()) / p_logv.exp()
        ).sum(-1) / self.latent_dim

        # -------- L_4: diffusion prior consistency
        if diff_prior is not None:
            # ----- linear β_t  (t/T · β_max) ------------------------
            t = torch.randint(1, steps + 1, (B,), device=device)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(1)

            # ----- add noise ----------------------------------------
            eps = torch.randn_like(z)
            z_t = alpha_bar_t.sqrt() * z + (1.0 - alpha_bar_t).sqrt() * eps

            # ----- denoise with transformer prior -------------------
            z0_hat = diff_prior(z_t, s0, t)

            # ----- Min-SNR weighting --------------------------------
            snr = alpha_bar_t.squeeze(1) / torch.clamp(1 - alpha_bar_t.squeeze(1), min=1e-5)
            weight = torch.minimum(snr, torch.full_like(snr, gamma_snr))
            l4 = (weight.unsqueeze(1) * (z0_hat - z).pow(2)).sum(-1) / self.latent_dim
        else:
            l4 = torch.zeros_like(kl)

        # -------- Total ELBO loss
        elbo = (l_a + lambda_state * l_s + beta * kl + l4).mean()
        return elbo
    
# -----------------------------------------------------------------------------
# [3] Diffusion Prior: Transformer for Latent Prediction in β‑VAE
# -----------------------------------------------------------------------------
    
class TransformerPrior(nn.Module):
    """
    Transformer-based prior network used in β-VAE training.
    
    Purpose:
        Predicts denoised latent vector z₀ from noisy latent z_t,
        conditioned on the current state s and timestep t.

    Architecture:
        - Two-token transformer encoder: [state_token, latent_token]
        - Shared time embedding is added to both tokens
        - Output is taken from the latent token position (index 1)

    Inputs:
        z_t : (B, latent_dim)    — noised latent vector at step t
        s   : (B, state_dim)     — current environment state
        t   : (B,)               — timestep ∈ [1, 200]
    
    Output:
        z₀ : (B, latent_dim)     — predicted denoised latent
    """
    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        d_model: int = 256,
        d_ff: int = 512
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Input projections to transformer dimension
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.state_proj  = nn.Linear(state_dim, d_model)
        self.time_emb    = nn.Embedding(200, d_model)  # max t = 200

        # --- Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            activation="gelu",
            norm_first=True,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Output projection back to latent space
        self.out = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,   # (B, latent_dim)
        s:   torch.Tensor,   # (B, state_dim)
        t:   torch.Tensor    # (B,)
    ) -> torch.Tensor:
        # Project inputs to transformer dimension
        z_tok = self.latent_proj(z_t)  # (B, D)
        s_tok = self.state_proj(s)    # (B, D)

        # Time embedding (shared across tokens)
        te = self.time_emb(t - 1)     # (B, D)

        # Combine state and latent as token sequence: [s_tok, z_tok]
        tokens = torch.stack([s_tok + te, z_tok + te], dim=1)  # (B, 2, D)

        # Transformer encoder
        h = self.encoder(tokens)  # (B, 2, D)
        
        # Return prediction from the latent position (index 1)
        return self.out(h[:, 1])  # (B, latent_dim)

# -----------------------------------------------------------------------------
# [4] Latent Diffusion Model: Vector-based U-Net (§A.2)
# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Two-layer residual MLP block.

    Forward:
        y = x + 0.5 * LayerNorm(GELU(W2 · LayerNorm(GELU(W1 · x))))

    Args:
        dim (int): Input and output dimension (must match).
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
    Rotary Positional Embedding (RoPE) for diffusion steps.

    Applies sinusoidal rotation to even/odd feature pairs based on step t.

    Args:
        dim (int): Feature dimension (must be even).
        theta (float): Base frequency for sinusoidal embeddings.
    """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        freqs = torch.exp(-torch.arange(0, dim, 2) * math.log(theta) / dim)
        self.register_buffer("freqs", freqs)  # shape: (dim/2,)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, D) - input feature vector
            t : (B,)   - timestep ∈ [1, T]

        Returns:
            x_rot : (B, D) - time-encoded feature vector
        """
        angles = t.float().unsqueeze(1) * self.freqs         # (B, D/2)
        sin, cos = angles.sin(), angles.cos()
        x1, x2 = x[:, 0::2], x[:, 1::2]                      # split even/odd
        x_rot = torch.stack([
            x1 * cos - x2 * sin,  # real part
            x1 * sin + x2 * cos   # imaginary part
        ], dim=-1)
        return x_rot.flatten(1)                              # (B, D)

class LatentDiffusionUNet(nn.Module):
    """
    1D U-Net used as latent diffusion model ψ (see Appendix A.2 in the paper).

    Input:
        - z_t : noisy latent vector       (B, latent_dim)
        - s   : conditioning state        (B, state_dim)
        - t   : diffusion step            (B,)

    Architecture:
        [z_t ⊕ s] → Rotary → +t → down1 → down2 → ResBlock → up1 → up2
                     skip ----------------------------^
    """
    def __init__(self, latent_dim: int, state_dim: int) -> None:
        super().__init__()
        in_dim = latent_dim + state_dim + 1  # last 1 = scalar t embedding

        self.down1 = mlp(in_dim,  (256,), 128)
        self.down2 = mlp(128,     (128,),  64)
        self.mid   = ResBlock(64)
        self.up1   = mlp(64,      (128,), 128)
        self.up2   = mlp(128,     (256,), latent_dim)

        self.rope  = Rotary(latent_dim + state_dim)

    def forward(
        self,
        z_t: torch.Tensor,  # (B, latent_dim)
        s:   torch.Tensor,  # (B, state_dim)
        t:   torch.Tensor   # (B,)
    ) -> torch.Tensor:
        # Concatenate latent and state
        base = torch.cat([z_t, s], dim=-1)                    # (B, D_no_t)
        
        # Apply rotary positional embedding
        x = self.rope(base, t)                                # (B, D_no_t)
        
        # Append scalar timestep as normalized float
        t_emb = t.float().unsqueeze(1) / 1000.0               # (B, 1)
        x = torch.cat([x, t_emb], dim=-1)                     # (B, D)
        
        # Forward through U-Net with skip connection
        d1 = self.down1(x)                                    # (B, 128)
        d2 = self.down2(d1)                                   # (B, 64)
        m  = self.mid(d2)                                     # (B, 64)
        u1 = self.up1(m)                                      # (B, 128)
        out = self.up2(u1 + d1)                               # (B, latent_dim)
        
        return out

# -----------------------------------------------------------------------------
# [4.1] Latent Diffusion Inference: DDPM / DDIM Sampling Procedures
# -----------------------------------------------------------------------------

def ddpm_sample(
    model: nn.Module,
    s: torch.Tensor,
    latent_dim: int,
    steps: int = 500,
    beta_max: float = 0.02
) -> torch.Tensor:
    """
    DDPM inference using the trained LatentDiffusionUNet.

    Args:
        model (nn.Module): trained diffusion model ψ
        s (Tensor): current state, shape (B, state_dim)
        latent_dim (int): latent dimensionality
        steps (int): number of diffusion steps T
        beta_max (float): max noise schedule β_T

    Returns:
        Tensor: sampled z₀ latent, shape (B, latent_dim)
    """
    B, device = s.size(0), s.device
    z = torch.randn(B, latent_dim, device=device)

    t_arr = torch.arange(steps, 0, -1, device=device)  # T … 1
    beta = (t_arr / steps) * beta_max
    alpha = 1.0 - beta

    for t_idx, bt, at in zip(t_arr, beta, alpha):
        tt = torch.full((B,), int(t_idx), device=device)
        z0 = model(z, s, tt)
        z = at * z0 + (1 - at) * z + torch.randn_like(z) * bt.sqrt()

    return z                                               # (B, latent_dim)

def ddim_sample(
    model: nn.Module,
    s: torch.Tensor,
    latent_dim: int,
    T_ddim: int = 100,
    beta_max: float = 0.02
) -> torch.Tensor:
    """
    DDIM inference (η = 0): deterministic reverse diffusion.

    Args:
        model (nn.Module): trained diffusion model ψ
        s (Tensor): current state, shape (B, state_dim)
        latent_dim (int): latent dimensionality
        T_ddim (int): number of DDIM steps
        beta_max (float): max β for noise schedule

    Returns:
        Tensor: sampled z₀ latent, shape (B, latent_dim)
    """
    B, device = s.size(0), s.device

    beta_full = torch.linspace(1e-4, beta_max, 500, device=device)
    alpha_bar_f = torch.cumprod(1.0 - beta_full, dim=0)
    idx_ddim = torch.linspace(0, 499, T_ddim, dtype=torch.long, device=device)
    a_bar = alpha_bar_f[idx_ddim]
    a_bar_next = torch.cat([a_bar[1:], a_bar.new_ones(1)])

    z_t = torch.randn(B, latent_dim, device=device)
    for t, a_t, a_prev in zip(
        reversed(range(1, T_ddim + 1)), a_bar.flip(0), a_bar_next.flip(0)
    ):
        tt = torch.full((B,), t, device=device)
        z0_hat = model(z_t, s, tt)
        z_t = (
            a_prev.sqrt() * z0_hat +
            (1 - a_prev).sqrt() *
            (z_t - a_t.sqrt() * z0_hat) / (1 - a_t).sqrt()
        )

    return z_t

# -----------------------------------------------------------------------------
# [5] Q- and Value Networks: DoubleQNet, ValueNet (Appendix A.3)
# -----------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """
    MLP-based encoder for state vector.

    Input:  s ∈ R^{state_dim}
    Output: h_s ∈ R^{256}
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, (256,), 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentEncoder(nn.Module):
    """
    MLP-based encoder for latent vector z.

    Input:  z ∈ R^{latent_dim}
    Output: h_z ∈ R^{128}
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, (128,), 128)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DoubleQNet(nn.Module):
    """
    Double Q-network (clipped double Q-learning).

    Architecture:
        - state encoder → 256 dim
        - latent encoder → 128 dim
        - concat → MLP → Q₁, Q₂ (scalar values)

    Input:
        s ∈ R^{B x state_dim}
        z ∈ R^{B x latent_dim}

    Output:
        q1, q2 ∈ R^{B}
    """
    def __init__(self, state_dim: int, latent_dim: int):
        super().__init__()
        self.s_enc = StateEncoder(state_dim)
        self.z_enc = LatentEncoder(latent_dim)
        self.head1 = mlp(256 + 128, (256, 256), 1)
        self.head2 = mlp(256 + 128, (256, 256), 1)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        sz = torch.cat([self.s_enc(s), self.z_enc(z)], dim=-1)
        return self.head1(sz).squeeze(-1), self.head2(sz).squeeze(-1)


class ValueNet(nn.Module):
    """
    Value network V(s) to estimate scalar state value.

    Architecture:
        - state encoder → 256 dim
        - MLP → scalar value

    Input:
        s ∈ R^{B x state_dim}
    Output:
        v ∈ R^{B}
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.enc = StateEncoder(state_dim)
        self.v   = mlp(256, (256, 256), 1)

    def forward(self, s: torch.Tensor):
        return self.v(self.enc(s)).squeeze(-1)

# -----------------------------------------------------------------------------
# [5.1] Value Loss: Expectile Regression from Top-k Q-values (Eq. 6)
# -----------------------------------------------------------------------------

def compute_v_loss(
    q_net_t,
    v_net,
    s: torch.Tensor,             # (B, state_dim)
    z_ddpm: torch.Tensor,        # (B, 500, latent_dim)
    z_data: torch.Tensor,        # (B, 32, latent_dim)
    *,
    k: int = 100,
    tau: float = 0.9,
    lambda_v: float = 0.5,
    return_stats: bool = False
):
    """
    Computes the V-loss (Eq. 6 from paper):
        L_V = ExpectileLoss(top-k(Q(s, z)) - V(s))

    Args:
        q_net_t     : target Q-network
        v_net       : current value network
        s           : batch of states, shape (B, state_dim)
        z_ddpm      : latents sampled from diffusion model, (B, 500, L)
        z_data      : latents from replay buffer, (B, 32, L)
        k           : number of top Q-values to use
        tau         : expectile (asymmetry coefficient)
        lambda_v    : loss scale factor
        return_stats: whether to return extra statistics

    Returns:
        loss: scalar loss
        [optional stats]: gap, v_mean, q_mean, v_pred, q_sel
    """
    B = s.size(0)
    z_all = torch.cat([z_ddpm, z_data], dim=1)  # (B, 532, L)
    n_all = z_all.size(1)

    with torch.no_grad():
        s_rep = s.unsqueeze(1).expand(-1, n_all, -1)  # (B, n_all, state_dim)
        q1, q2 = q_net_t(
            s_rep.reshape(-1, s.shape[-1]),
            z_all.reshape(-1, z_all.shape[-1])
        )
        q_vals = torch.min(q1, q2).view(B, n_all)  # clipped double Q
        q_sel = torch.topk(q_vals, k=min(k, n_all), dim=1).values  # (B, k)

    v_pred = v_net(s).unsqueeze(1)  # (B, 1)
    u = q_sel - v_pred              # (B, k)
    w = torch.where(u > 0, tau, 1 - tau)
    loss = lambda_v * (w * u.pow(2)).mean()
    
    if return_stats:
        gap = (q_sel.mean(dim=1) - v_pred.squeeze(1)).mean().detach()
        return loss, gap, v_pred.mean().detach(), q_sel.mean().detach(), v_pred.detach(), q_sel.detach()
    
    return loss

# -----------------------------------------------------------------------------
# [6] Policy Execution: Latent Sampling, Adaptive Revaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def policy_execute(
    env:        gym.Env,
    q_net:      DoubleQNet,
    v_net:      ValueNet,
    beta_vae:   BetaVAE,
    diffusion:  LatentDiffusionUNet,
    *,
    latent_dim:     int = 16,
    ddpm_steps:     int = 10,        # DDIM steps; "extra steps = 5"
    n_latents:      int = 500,       # number of latent candidates
    reval_attempts: int = 5,         # max resample attempts for AR
    value_eps:      float = 0.2,     # tolerance for Adaptive Revaluation
    device:         str = "cpu",
    max_actions:    Optional[int] = None
) -> float:
    """
    Executes a DIAR policy in the environment (Algorithm 2).

    Workflow:
        1. For each state s:
           - Generate N latent samples z_i using DDIM
           - Select z* = argmax Q(s, z_i)
           - Decode a*, s' ← BetaVAE(s, z*)
           - Accept z* if V(s') ≥ V(s) - ε
           - If not, resample up to `reval_attempts` times
        2. Execute decoded action and update environment
        3. Repeat until done or action limit

    Returns:
        float: total episode reward
    """

    def augment(state_np: np.ndarray, goal_abs: np.ndarray) -> np.ndarray:
        """
        Concatenate relative goal to state: s_aug = [s, goal - pos]
        """
        return np.concatenate([state_np, goal_abs - state_np[:2]], axis=0)

    # -------------------------------------------------------------------------
    goal_abs = np.asarray(env.unwrapped._target, dtype=np.float32)
    state_np = env.reset()
    state_np = augment(state_np, goal_abs)
    done = False
    total_reward = 0.0
    action_count = 0
    max_actions = max_actions or env._max_episode_steps

    while not done and action_count < max_actions:
        # Convert state to torch tensor
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, s_dim)

        # Try multiple revaluation attempts
        for _ in range(reval_attempts):
            # Sample candidate latents via DDIM
            z_pool = ddim_sample(
                model=diffusion,
                s=state.repeat(n_latents, 1),  # (B = n_latents, s_dim)
                latent_dim=latent_dim,
                T_ddim=ddpm_steps             # short reverse path
            )  # → (B, latent_dim)

            # Evaluate Q-values
            q1, q2 = q_net(state.repeat(n_latents, 1), z_pool)
            q_min = torch.min(q1, q2)
            z_star = z_pool[q_min.argmax()].unsqueeze(0)  # select best latent (1, latent_dim)

            # Decode action and next state
            action_pred = beta_vae.decode_action(state, z_star)  # (1, a_dim)
            state_pred  = beta_vae.decode_state(state, z_star)   # (1, s_dim)

            # Adaptive Revaluation
            if v_net(state_pred) + value_eps >= v_net(state):
                break  # Accept current z_star

        # Execute decoded action in environment
        action_np = action_pred.squeeze(0).cpu().detach().numpy()
        state_np, reward, done, _ = env.step(action_np)
        state_np = augment(state_np, goal_abs)
        total_reward += reward
        action_count += 1

    return total_reward

# -----------------------------------------------------------------------------
# [7] Replay Buffer: Prioritized Sampling (PER)
# -----------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Stores transitions (s, a, r, s', z) and allows prioritized sampling
    based on temporal-difference error magnitudes (used in DIAR Q-learning).
    """
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        device: torch.device,
        alpha: float = 0.7,
        beta_start: float = 0.3,
        beta_frames: int = 100_000
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # used to schedule beta

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Initialize tensors
        self.ptr = 0
        self.size = 0
        
        self.states      = torch.zeros((capacity, state_dim),  device=device)
        self.actions     = torch.zeros((capacity, action_dim), device=device)
        self.rewards     = torch.zeros((capacity,),            device=device)
        self.next_states = torch.zeros((capacity, state_dim),  device=device)
        self.latents     = torch.zeros((capacity, latent_dim), device=device)
        self.priorities  = torch.ones ((capacity,),            device=device)

    def add(self, s, a, r, s_next, z):
        """
        Adds a new transition to the buffer.
        """
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.latents[self.ptr] = z
        self.priorities[self.ptr] = self.priorities.max() if self.size > 0 else 1.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Samples a batch using prioritized sampling.
        """
        probs = (self.priorities[:self.size] if self.size < self.capacity else self.priorities) ** self.alpha
        probs /= probs.sum()

        idx = torch.multinomial(probs, batch_size, replacement=False)
        
        beta = min(self.beta_start + (1 - self.beta_start) * (self.frame / self.beta_frames), 1.0)
        self.frame += 1

        weights = (self.capacity * probs[idx]) ** (-beta)
        weights /= weights.max()

        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
            "latent": self.latents[idx],
            "weights": weights.unsqueeze(1).to(self.device),
            "indices": idx
        }

    def update_priorities(self, idx, prio):
        """
        Updates sampling priorities for the specified indices.
        """
        self.priorities[idx] = prio

    # --------------------------- I/O --------------------------------------
    def save(self, path: str):
        """
        Saves buffer to disk (CPU tensors).
        """
        cpu_tensors = {k: v.cpu() for k, v in self.__dict__.items() if isinstance(v, torch.Tensor)}
        meta = {k: v for k, v in self.__dict__.items() if not isinstance(v, torch.Tensor) and k != "device"}
        torch.save({'tensors': cpu_tensors, 'meta': meta}, path)

    @classmethod
    def load(cls, path: str, device="cpu"):
        """
        Loads buffer from disk.
        """
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt['meta']
        tensors = ckpt['tensors']
        
        init_args = {
            'capacity': meta['capacity'],
            'state_dim': meta['state_dim'],
            'action_dim': meta['action_dim'],
            'latent_dim': meta['latent_dim'],
            'alpha': meta['alpha'],
            'beta_start': meta['beta_start'],
            'beta_frames': meta['beta_frames'],
            'device': device
        }
        
        obj = cls(**init_args)
        
        for k, v in meta.items():
            if k not in init_args and k != 'device':
                setattr(obj, k, v)
                
        for k, v in tensors.items():
            getattr(obj, k).copy_(v.to(device))
            
        return obj
    
# -----------------------------------------------------------------------------
# [7.1] Replay Buffer Construction from D4RL Dataset
# -----------------------------------------------------------------------------
    
def build_replay_buffer(
    dataset,
    valid_mask,
    horizon,
    beta_vae,
    state_dim,
    action_dim,
    latent_dim,
    device,
    *,
    cache_path: str = "replay_buffer.pt",
    use_cache: bool = True
) -> PrioritizedReplayBuffer:
    """
    Constructs and populates the replay buffer from D4RL dataset.

    Uses the β-VAE to encode each trajectory into a latent vector.
    """
    if use_cache and os.path.exists(cache_path):
        print(f"→ Loading buffer from {cache_path}")
        return PrioritizedReplayBuffer.load(cache_path, device)

    print("→ Building replay buffer from scratch …")
    cap = int(valid_mask.sum())
    rb = PrioritizedReplayBuffer(
        capacity=cap,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        device=device
    )

    gamma = 0.995
    obs, acts, rews, next_obs = (
        dataset["observations"],
        dataset["actions"],
        dataset["rewards"],
        dataset["next_observations"]
    )

    idx_all = (
        torch.from_numpy(np.nonzero(valid_mask)[0])
        if isinstance(valid_mask, np.ndarray)
        else torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    )
    
    for i in tqdm(idx_all.cpu()):
        s_seq = torch.tensor(obs[i : i + horizon], device=device, dtype=torch.float32)
        a_seq = torch.tensor(acts[i : i + horizon], device=device, dtype=torch.float32)
        s0 = s_seq[0]
        a0 = a_seq[0]
        sH = torch.tensor(next_obs[i + horizon], device=device, dtype=torch.float32)

        R = sum((gamma ** j) * float(rews[i + j]) for j in range(horizon))
        R = torch.tensor(R, dtype=torch.float32, device=device)

        with torch.no_grad():
            z, _, _ = beta_vae.encode(s_seq.unsqueeze(0), a_seq.unsqueeze(0))
            z = z.squeeze(0)  # (latent_dim,)

        rb.add(s0, a0, R, sH, z)

    if use_cache:
        rb.save(cache_path)
        print(f"→ Buffer saved to {cache_path}")

    return rb

# -----------------------------------------------------------------------------
# [8.1] Phase 1 – β‑VAE + Transformer Prior Training
# -----------------------------------------------------------------------------

class HorizonDataset(Dataset):
    """
    Dataset of fixed-length trajectories for ELBO training.

    Extracts (s_seq, a_seq) pairs of length H starting at valid indices.
    """
    def __init__(self, obs: torch.Tensor, acts: torch.Tensor, valid_idx: torch.Tensor, H: int):
        self.obs = obs
        self.acts = acts
        self.idx = valid_idx
        self.H = H

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i].item()
        return self.obs[j:j + self.H], self.acts[j:j + self.H]  # (H, obs_dim), (H, act_dim)

def train_beta_vae(
    dataset: dict,
    valid: np.ndarray,
    beta_vae: BetaVAE,
    diff_prior: TransformerPrior,
    *,
    horizon: int = 30,
    device: str = "cuda",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 5e-5,
    save_every: int = 10,
    save_dir: str = "output/vae"
):
    """
    Trains β-VAE and TransformerPrior jointly using ELBO objective (phase 1).

    Args:
        dataset     : raw offline dataset (D4RL format)
        valid       : boolean mask of valid H-length sequences
        beta_vae    : β-VAE model (encoder/decoder/prior)
        diff_prior  : Transformer prior model
        horizon     : sequence length H
        device      : CUDA or CPU
        epochs      : number of training epochs
        batch_size  : batch size
        lr          : learning rate
        save_every  : checkpoint interval
        save_dir    : directory to store checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to tensors
    obs  = torch.tensor(dataset["observations"], dtype=torch.float32)
    acts = torch.tensor(dataset["actions"],      dtype=torch.float32)
    valid_idx = torch.nonzero(torch.tensor(valid), as_tuple=False).squeeze(1)

    # Create dataloader over H-step trajectory segments
    data = HorizonDataset(obs, acts, valid_idx, horizon)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True
    )

    # Optimizer over β-VAE + Transformer prior
    opt = torch.optim.Adam(
        list(beta_vae.parameters()) + list(diff_prior.parameters()),
        lr=lr
    )

    for epoch in trange(epochs, desc="β-VAE training"):
        losses = []

        for s_seq, a_seq in loader:
            s_seq = s_seq.to(device, non_blocking=True)  # (B, H, obs_dim)
            a_seq = a_seq.to(device, non_blocking=True)  # (B, H, act_dim)

            loss = beta_vae.elbo_loss(
                s_seq,
                a_seq,
                diff_prior=diff_prior,
                beta=0.1  # fixed β as in paper
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(beta_vae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(diff_prior.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        wandb.log({"beta_vae/elbo": np.mean(losses)}, step=epoch)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(beta_vae.state_dict(),     os.path.join(save_dir, f"beta_vae_epoch{epoch}.pt"))
            torch.save(diff_prior.state_dict(),   os.path.join(save_dir, f"diff_prior_epoch{epoch}.pt"))

# -----------------------------------------------------------------------------
# [8.2] Phase 2 – Latent Diffusion U-Net Training
# -----------------------------------------------------------------------------

def preprocess_latents(beta_vae, obs, acts, start_idx, horizon, latent_dim, device, batch_size=1024):
    """
    Computes z and s0 for each index in start_idx using β-VAE encoder.
    """
    N = start_idx.numel()
    latent_dataset = torch.zeros((N, latent_dim), device=device)
    s0_dataset = torch.zeros((N, obs.shape[-1]), device=device)

    for i in trange(0, N, batch_size, desc="Encoding latents"):
        idx = start_idx[i:i+batch_size]
        s_seq = torch.stack([obs[j : j + horizon] for j in idx])  # (B, H, obs_dim)
        a_seq = torch.stack([acts[j : j + horizon] for j in idx])  # (B, H, act_dim)
        with torch.no_grad():
            z, _, _ = beta_vae.encode(s_seq, a_seq)
        latent_dataset[i:i+z.shape[0]] = z
        s0_dataset[i:i+z.shape[0]] = s_seq[:, 0]

    return s0_dataset, latent_dataset


def train_diffusion_model(
    dataset: dict,
    valid: np.ndarray,
    beta_vae,
    diffusion_model,
    *,
    horizon: int = 30,
    T_ddim: int = 100,
    gamma_snr: float = 5.0,
    drop_prob: float = 0.10,
    device: str = "cuda",
    epochs: int = 450,
    batch_size: int = 128,
    lr: float = 1e-4,
    save_every: int = 50,
    save_dir: str = "output/diffusion",
    latent_cache_path: str = "output/diffusion/latents.pt",
    use_cache: bool = True
) -> None:
    """
    Trains the Latent Diffusion Model (LDM) to denoise latent vectors produced by β-VAE.

    Steps:
        1. Optionally loads cached latent vectors z and initial states s₀ (from β-VAE).
        2. Precomputes diffusion coefficients ᾱ, sqrt(ᾱ), and SNR weights.
        3. Trains the U-Net-based denoiser to predict clean z₀ from noisy z_t,
           conditioned on s₀ and timestep t, using SNR-weighted MSE loss.

    Args:
        dataset           : Offline dataset with "observations" and "actions".
        valid             : Boolean mask of valid starting indices.
        beta_vae          : Pretrained β-VAE (in eval mode).
        diffusion_model   : LatentDiffusionUNet (ψ).
        horizon           : Trajectory segment length H.
        T_ddim            : Number of diffusion timesteps (discrete).
        gamma_snr         : Min-SNR-γ clipping value for weighting loss.
        drop_prob         : Probability to randomly drop samples (stochastic masking).
        device            : Training device.
        epochs            : Number of training epochs.
        batch_size        : Minibatch size.
        lr                : Learning rate.
        save_every        : Save checkpoint every N epochs.
        save_dir          : Path to save checkpoints.
        latent_cache_path : Optional path to cached s₀/z pairs.
        use_cache         : Whether to use (or write) latent cache.
    """
    os.makedirs(save_dir, exist_ok=True)

    obs  = torch.tensor(dataset["observations"], dtype=torch.float32, device=device)
    acts = torch.tensor(dataset["actions"],      dtype=torch.float32, device=device)

    start_idx = torch.tensor(valid, device=device).nonzero(as_tuple=False).squeeze(1)
    latent_dim = beta_vae.latent_dim

    # --- Try loading cached latents ---
    if use_cache and os.path.exists(latent_cache_path):
        print(f"→ Loading latents from {latent_cache_path}")
        cache = torch.load(latent_cache_path, map_location=device, weights_only=True)
        s0_dataset = cache["s0"]
        latent_dataset = cache["z"]
    else:
        # --- Preprocess latents from β-VAE ---
        print("→ Preprocessing latents …")
        s0_dataset, latent_dataset = preprocess_latents(
            beta_vae=beta_vae,
            obs=obs,
            acts=acts,
            start_idx=start_idx,
            horizon=horizon,
            latent_dim=latent_dim,
            device=device,
            batch_size=1024
        )
        # --- Save to cache ---
        torch.save({"s0": s0_dataset, "z": latent_dataset}, latent_cache_path)
        print(f"→ Latents saved to {latent_cache_path}")

    # --- Precompute alpha_bar lookup ---
    beta_full   = torch.linspace(1e-4, 0.02, 500, device=device)
    alpha_bar_f = torch.cumprod(1.0 - beta_full, dim=0)
    idx_ddim    = torch.linspace(0, 499, T_ddim, dtype=torch.long, device=device)
    alpha_bar   = alpha_bar_f[idx_ddim]

    # --- Precompute SNR lookup [sqrt(ab), sqrt(1-ab), SNR] ---
    sqrt_ab     = alpha_bar.sqrt()
    sqrt_1m_ab  = (1 - alpha_bar).sqrt()
    snr_weights = torch.minimum(alpha_bar / (1 - alpha_bar), torch.tensor(gamma_snr, device=device))
    alpha_bar_lookup = torch.stack([sqrt_ab, sqrt_1m_ab, snr_weights], dim=1)

    # --- DataLoader ---
    loader = DataLoader(
        TensorDataset(s0_dataset, latent_dataset),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)

    for epoch in trange(epochs, desc="Latent-Diffusion"):
        loss_epoch: list[float] = []

        for s0, z in loader:
            s0, z = s0.to(device), z.to(device)
            B = z.size(0)

            # Sample random diffusion timestep t ∈ [1, T_ddim]
            t = torch.randint(1, T_ddim + 1, (B,), device=device)

            # Lookup corresponding alpha_bar_t, 1 - alpha_bar_t, and SNR weights
            lookup = alpha_bar_lookup[t - 1]  # (B, 3)
            sqrt_ab      = lookup[:, 0:1]     # √ᾱ_t
            sqrt_1m_ab   = lookup[:, 1:2]     # √(1-ᾱ_t)
            weight       = lookup[:, 2:3]     # min(SNR_t, γ)

            # Forward diffusion process: add noise to latent z → z_t
            eps = torch.randn_like(z)
            z_j = sqrt_ab * z + sqrt_1m_ab * eps

            # Predict z₀ from z_t using diffusion model ψ
            z0_pred = diffusion_model(z_j, s0, t)

            # Apply stochastic masking (Dropout-style noise dropout)
            keep_mask = (torch.rand_like(weight) > drop_prob).float()
            weight = weight * keep_mask

            # SNR-weighted MSE loss
            loss = ((weight * (z0_pred - z).pow(2)).sum(-1) /
                    (keep_mask.mean() + 1e-8)).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            optimizer.step()
            loss_epoch.append(loss.item())

        wandb.log({"diffusion/loss": sum(loss_epoch) / len(loss_epoch)}, step=epoch + 100)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(diffusion_model.state_dict(), os.path.join(save_dir, f"ldm_epoch{epoch}.pt"))

# -----------------------------------------------------------------------------
# [8.3] Phase 3 – DIAR Training Loop: Q / V
# -----------------------------------------------------------------------------

def train_diar(
    env: gym.Env,
    *,
    dataset: dict,
    valid_mask: np.ndarray,
    beta_vae: BetaVAE,                # pretrained, frozen
    diffusion_model: LatentDiffusionUNet,  # pretrained, frozen
    state_dim: int,
    action_dim: int,
    latent_dim: int = 16,
    # ---------------- optimization params ------------------------------------
    num_steps: int = 100_000,
    horizon: int = 30,
    batch_size: int = 128,
    lr_q: float = 5e-4,
    lr_v: float = 5e-4,
    step_lr_iters: int = 50_000,
    device: str = "cuda",
    save_every: int = 10_000,
    save_dir: str = "output/diar",
    use_cache_rb: bool = True
):
    """
    Trains DIAR Q-network and Value-network with a diffusion-model-guided replay buffer.

    Q-learning updates are performed using the β-VAE encoded latents z from the dataset.
    Value learning uses top-k Q(s, z) estimates from both diffusion-sampled and dataset latents.

    Args:
        env           : Gym environment for evaluation
        dataset       : D4RL dataset dict
        valid_mask    : boolean mask for valid sequences of length H
        beta_vae      : trained and frozen β-VAE model
        diffusion_model: trained and frozen LatentDiffusionUNet
        state_dim     : dimensionality of states
        action_dim    : dimensionality of actions
        latent_dim    : dimensionality of latent z
        num_steps     : total number of training steps
        horizon       : H (trajectory length used for encoding)
        batch_size    : number of samples per batch
        lr_q, lr_v    : learning rates for Q and V networks
        step_lr_iters : V-network LR scheduler step size
        save_every    : how often to checkpoint models
        save_dir      : path to save models
        use_cache_rb  : whether to cache/load the replay buffer
    """
    os.makedirs(save_dir, exist_ok=True)

    # ------------------- initialize Q / V networks ---------------------------
    beta_vae.eval()
    diffusion_model.eval()

    q_net   = DoubleQNet(state_dim, latent_dim).to(device)
    v_net   = ValueNet(state_dim).to(device)
    q_tgt   = DoubleQNet(state_dim, latent_dim).to(device)
    v_tgt   = ValueNet(state_dim).to(device)

    q_tgt.load_state_dict(q_net.state_dict())
    v_tgt.load_state_dict(v_net.state_dict())

    opt_q = torch.optim.Adam(q_net.parameters(), lr=lr_q)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=lr_v)
    sched_v = torch.optim.lr_scheduler.StepLR(opt_v, step_size=step_lr_iters, gamma=0.3)

    # ------------------- prioritized replay buffer ---------------------------
    rb = build_replay_buffer(
        dataset=dataset,
        valid_mask=valid_mask,
        horizon=horizon,
        beta_vae=beta_vae,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        device=device,
        cache_path=os.path.join(save_dir, "replay_buffer.pt"),
        use_cache=use_cache_rb
    )

    # ------------------- training constants ----------------------------------
    tau = 0.9
    gamma = 0.995
    n_ddpm_lat = 500  # number of sampled latents from diffusion

    pbar = trange(num_steps, desc="DIAR-train")
    for step in pbar:
        # ====================== Q-network update ============================
        batch = rb.sample(batch_size)
        s = batch['state']        # (B, s_dim)
        z = batch['latent']       # (B, latent_dim)
        r = batch['reward']       # (B,)
        sn = batch['next_state']  # (B, s_dim)
        w = batch['weights']      # (B,1)
        idx = batch['indices']
        
        with torch.no_grad():
            v_target = v_tgt(sn)                  # V(s')
            q_target = r + gamma * v_target       # r + γV(s')

        q1, q2 = q_net(s, z)
        q_pred = torch.min(q1, q2)                # clipped double Q
        td = q_target - q_pred                    # TD-error

        loss_q = (w.squeeze(1) * td.pow(2)).mean()

        opt_q.zero_grad()
        loss_q.backward()
        opt_q.step()

        # update priorities in replay buffer
        rb.update_priorities(idx, (td.abs() + 1e-6).detach())

        # ====================== V-network update ============================
        with torch.no_grad():
            # Sample z from diffusion model conditioned on s
            s_rep = s.repeat_interleave(n_ddpm_lat, 0)  # (B×500, s_dim)
            z_ddpm = ddim_sample(
                model=diffusion_model,
                s=s_rep,
                latent_dim=latent_dim,
                T_ddim=10
            ).view(batch_size, n_ddpm_lat, latent_dim)

            # Expand z from dataset for top-k matching
            z_data = z.unsqueeze(1).expand(-1, 32, -1)  # (B,32,L)

        loss_v = compute_v_loss(
            q_net_t=q_tgt,
            v_net=v_net,
            s=s,
            z_ddpm=z_ddpm,
            z_data=z_data,
            k=100,
            tau=tau,
            lambda_v=0.5
        )

        opt_v.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        opt_v.step()
        sched_v.step()

        # ====================== soft update targets =========================
        with torch.no_grad():
            for p, tp in zip(q_net.parameters(), q_tgt.parameters()):
                tp.mul_(0.995).add_(0.005 * p)
            for p, tp in zip(v_net.parameters(), v_tgt.parameters()):
                tp.mul_(0.995).add_(0.005 * p)

        # ====================== evaluation logging ==========================
        if step % 500 == 0:
            rew = policy_execute(
                env,
                q_net=q_net,
                v_net=v_net,
                beta_vae=beta_vae,
                diffusion=diffusion_model,
                max_actions=150,
                device=device
            )
            wandb.log({
                "eval/reward": rew,
                "loss/q": loss_q.item(),
                "loss/v": loss_v.item()
            }, step=step + 550)
            pbar.set_postfix({"R": f"{rew:6.1f}"})

        # ====================== save checkpoints ============================
        if step % save_every == 0 or step == num_steps - 1:
            torch.save(q_net.state_dict(), os.path.join(save_dir, f"q_net_{step}.pt"))
            torch.save(v_net.state_dict(), os.path.join(save_dir, f"v_net_{step}.pt"))

# -----------------------------------------------------------------------------
# [9] Utility Functions
# -----------------------------------------------------------------------------

def load_latest_checkpoint(model, name, folder="output"):
    """
    Loads the most recent checkpoint for a model from a specified folder.

    Args:
        model  : torch.nn.Module to be loaded.
        name   : base name used in saved checkpoints (e.g., "q_net").
        folder : directory where checkpoints are stored.

    Returns:
        model with updated weights from latest checkpoint.
    """
    checkpoints = glob.glob(os.path.join(folder, f"{name}_*.pt"))
    if not checkpoints:
        print(f"No checkpoint found for {name}")
        return model
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Loading {name} from {latest}")
    model.load_state_dict(torch.load(latest, weights_only=True))
    return model


def build_next_obs(dset, horizon):
    """
    Constructs dset["next_observations"] for a dataset and identifies
    which starting indices are valid for H-step sequences (i.e., don't cross episode ends).

    Behavior:
        - Next observation is shifted version of current observation.
        - If the transition crosses a terminal or timeout, it is considered invalid.
        - Constructs a boolean valid_mask for use in β-VAE training and buffer creation.

    Args:
        dset    : dataset with keys "observations", "terminals", "timeouts".
        horizon: sequence length H to ensure continuity.

    Returns:
        valid_mask : np.ndarray of shape (len(obs) - H,) with True for valid starts.
    """
    obs = dset["observations"]
    terms = dset["terminals"].astype(bool) | dset["timeouts"].astype(bool)

    # Step-wise shifted observations (obs_t+1)
    next_obs = np.empty_like(obs)
    next_obs[:-1] = obs[1:]
    next_obs[-1]  = obs[-1]  # last obs has no true "next", so we duplicate

    # If transition crosses terminal boundary, keep same observation
    next_obs[terms] = obs[terms]

    dset["next_observations"] = next_obs

    # Validity mask for sampling H-step sequences within same episode
    valid_mask = np.ones(len(obs) - horizon, dtype=bool)
    for i in range(len(valid_mask)):
        if terms[i : i + horizon].any():
            valid_mask[i] = False

    return valid_mask

# -----------------------------------------------------------------------------
# [10] Main Execution Pipeline: β‑VAE → Diffusion → Q/V Training
# -----------------------------------------------------------------------------

def main():
    """
    DIAR training entry point.

    This script executes the full DIAR pipeline:
        1. Loads D4RL environment and augments observations.
        2. Trains β-VAE and Transformer prior (phase 1).
        3. Trains Latent Diffusion U-Net to denoise z (phase 2).
        4. Trains Q- and Value-networks using DDIM sampling + latent evaluation (phase 3).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="maze2d-umaze-v1", help="Gym environment name")
    parser.add_argument("--device",     default="cuda",            help="Device to use: cuda or cpu")
    parser.add_argument("--save_dir",   default="output",          help="Directory to store checkpoints and outputs")
    parser.add_argument("--latent_dim", default=16, type=int,      help="Dimensionality of β-VAE latent space")
    parser.add_argument("--horizon",    default=30, type=int,      help="Length of trajectory segments")
    args = parser.parse_args()

    wandb.init(project="diar", name=f"diar_{args.env}")

    # ------------------- Create subdirectories ------------------------------
    for sub in ("vae", "diffusion", "diar"):
        os.makedirs(os.path.join(args.save_dir, sub), exist_ok=True)

    # ------------------- Load D4RL dataset ----------------------------------
    env = gym.make(args.env)
    dset = env.get_dataset()

    obs = dset["observations"]         # (N, 4)
    goal = dset["infos/goal"]          # (N, 2)
    obs_aug = np.concatenate([obs, goal - obs[:, :2]], axis=1)  # relative goal → (N, 6)
    dset["observations"] = obs_aug

    if "next_observations" not in dset:
        print("→ Generating next_observations and valid mask …")
        valid_mask = build_next_obs(dset, args.horizon)
    else:
        valid_mask = np.ones(len(dset["observations"]) - args.horizon, dtype=bool)

    state_dim = dset["observations"].shape[1]
    action_dim = dset["actions"].shape[1]

    # ------------------- Initialize models ----------------------------------
    beta_vae = BetaVAE(state_dim, action_dim, args.latent_dim).to(args.device)
    t_prior  = TransformerPrior(args.latent_dim, state_dim).to(args.device)
    ldm_unet = LatentDiffusionUNet(args.latent_dim, state_dim).to(args.device)

    # ------------------- (1) β-VAE + Transformer Prior Training ------------
    train_beta_vae(
        dataset     = dset,
        valid       = valid_mask,
        beta_vae    = beta_vae,
        diff_prior  = t_prior,
        device      = args.device,
        epochs      = 100,
        save_dir    = os.path.join(args.save_dir, "vae")
    )

    #beta_vae = load_latest_checkpoint(beta_vae, "beta_vae", os.path.join(args.save_dir, "vae"))
    #t_prior  = load_latest_checkpoint(t_prior,  "diff_prior", os.path.join(args.save_dir, "vae"))

    # Freeze β-VAE and Transformer prior
    beta_vae.eval()
    for p in beta_vae.parameters(): p.requires_grad = False
    t_prior.eval()
    for p in t_prior.parameters(): p.requires_grad = False

    # ------------------- (2) Latent Diffusion Model Training ----------------
    train_diffusion_model(
        dataset         = dset,
        valid           = valid_mask,
        beta_vae        = beta_vae,
        diffusion_model = ldm_unet,
        device          = args.device,
        epochs          = 450,
        save_dir        = os.path.join(args.save_dir, "diffusion")
    )
    
    #ldm_unet = load_latest_checkpoint(ldm_unet, "ldm", os.path.join(args.save_dir, "diffusion"))
    
    ldm_unet.eval()
    for p in ldm_unet.parameters(): p.requires_grad = False

    # ------------------- (3) DIAR Q / V Training ----------------------------
    train_diar(
        env              = env,
        dataset          = dset,
        valid_mask       = valid_mask,
        beta_vae         = beta_vae,         # frozen
        diffusion_model  = ldm_unet,         # frozen
        state_dim        = state_dim,
        action_dim       = action_dim,
        latent_dim       = args.latent_dim,
        num_steps        = 100_000,
        device           = args.device,
        save_dir         = os.path.join(args.save_dir, "diar")
    )


if __name__ == "__main__":
    main()
