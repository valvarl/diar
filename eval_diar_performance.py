import argparse
from typing import Optional

import torch
import gym
import d4rl
import numpy as np
from train_diar_d4rl_maze2d import (
    BetaVAE,
    LatentDiffusionUNet,
    DoubleQNet,
    ValueNet,
    ddim_sample,
    policy_execute,
)

# ---------------------------------------------------------------------------- #
#  DIAR evaluation that mirrors the original maze2d-large metric computation   #
# ---------------------------------------------------------------------------- #
import torch
import numpy as np
from typing import List

@torch.no_grad()
def diar_plan_latent(
    diffusion:  LatentDiffusionUNet,
    beta_vae:   BetaVAE,
    q_net:      DoubleQNet,
    v_net:      Optional[ValueNet],
    state:      torch.Tensor,         # (B_alive, s_dim)
    *,
    n_latents:  int = 500,
    ddpm_steps: int = 10,
    value_eps:  float = 0.2,
) -> torch.Tensor:
    """
    Batched latent selection with Adaptive Revaluation (один проход).
    """
    assert q_net is not None, "q_net required for planning"

    B, _     = state.shape
    device   = state.device
    z_dim    = beta_vae.latent_dim

    # DDIM sampling -------------------------------------------------------
    z_pool = ddim_sample(
        model      = diffusion,
        s          = state.repeat_interleave(n_latents, dim=0),
        latent_dim = z_dim,
        T_ddim     = ddpm_steps,
    )  # (B*n_latents, z_dim)

    # Q-оценка ------------------------------------------------------------
    rep_state = state.repeat_interleave(n_latents, 0)
    q1, q2    = q_net(rep_state, z_pool)
    q_min     = torch.min(q1, q2).view(B, n_latents)
    best_idx  = q_min.argmax(dim=1)

    best_latent = torch.stack([
        z_pool[i*n_latents + best_idx[i]] for i in range(B)
    ], dim=0)  # (B, z_dim)

    # One-shot Adaptive Revaluation --------------------------------------
    if v_net is not None:
        s_pred  = beta_vae.decode_state(state, best_latent)
        accept  = (v_net(s_pred) + value_eps >= v_net(state)).flatten()
        if not accept.all():
            bad_mask      = ~accept
            s_bad         = state[bad_mask]
            z_pool_bad    = ddim_sample(
                model      = diffusion,
                s          = s_bad.repeat_interleave(n_latents, 0),
                latent_dim = z_dim,
                T_ddim     = ddpm_steps,
            )
            q1b, q2b      = q_net(s_bad.repeat_interleave(n_latents, 0), z_pool_bad)
            q_min_bad     = torch.min(q1b, q2b).view(-1, n_latents)
            best_bad_idx  = q_min_bad.argmax(dim=1)
            for j, idx in enumerate(best_bad_idx):
                best_latent[bad_mask.nonzero(as_tuple=True)[0][j]] = \
                    z_pool_bad[j*n_latents + idx]

    return best_latent
# --------------------------------------------------------------------------- #

def augment_state(s_phys: torch.Tensor, goal_abs: torch.Tensor) -> torch.Tensor:
    """
    Args:
        s_phys   : (B, 4)  – [x, y, vx, vy]
        goal_abs : (B, 2)  – абсолютные координаты цели
    Returns:
        s_aug    : (B, 6)  – [x, y, vx, vy, goal_x-x, goal_y-y]
    """
    rel = goal_abs - s_phys[:, :2]          # (B, 2)
    return torch.cat([s_phys, rel], dim=1)  # (B, 6)
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_diar_policy(
    envs:               List[gym.Env],
    beta_vae:           BetaVAE,
    diffusion_model:    LatentDiffusionUNet,
    q_net:              Optional[DoubleQNet],
    v_net:              Optional[ValueNet],
    *,
    num_evals:          int  = 100,
    num_parallel_envs:  int  = 10,
    n_latents:          int  = 500,
    ddpm_steps:         int  = 10,
    exec_horizon:       int  = 30,
    device:             str  = "cuda",
    render:             bool = False,
    value_eps:          float= 0.2,
) -> float:
    """
    Оценка DIAR-политики, строго повторяющая метрику maze2d-large из оригинального
    кода.  Главное отличие от предыдущей версии — теперь мы используем
    **аугментированное состояние** (6-мерное), на котором и обучались модели.
    """
    torch_device   = torch.device(device)
    state_dim_raw  = envs[0].observation_space.shape[0]      # 4
    state_dim_aug  = state_dim_raw + 2                       # 6
    goal_fixed     = torch.tensor([1.0, 1.0],
                                  device=torch_device)       # (2,)

    MAX_STEPS      = 300
    batches        = num_evals // num_parallel_envs
    successes      = 0
    raw_score_sum  = 0.0

    for batch in range(batches):
        # ── reset параллельных сред ────────────────────────────────────────
        S_phys = torch.zeros((num_parallel_envs, state_dim_raw),
                             device=torch_device)
        S_aug  = torch.zeros((num_parallel_envs, state_dim_aug),
                             device=torch_device)
        goal   = goal_fixed.expand(num_parallel_envs, -1)
        done   = torch.zeros(num_parallel_envs, dtype=torch.bool,
                             device=torch_device)
        succ   = torch.zeros_like(done)

        for k in range(num_parallel_envs):
            s_np           = envs[k].reset()                # (4,)
            S_phys[k]      = torch.tensor(s_np, device=torch_device)
            S_aug[k]       = augment_state(S_phys[k:k+1], goal[k:k+1])

        env_step = 0
        # ── episode rollout ───────────────────────────────────────────────
        while env_step < MAX_STEPS:
            alive_idx = (~done).nonzero(as_tuple=True)[0]
            if alive_idx.numel() == 0:
                break  # все среды завершили эпизод

            # ── планируем лучший латент только для «живых» сред ──────────
            best_z = diar_plan_latent(
                diffusion   = diffusion_model,
                beta_vae    = beta_vae,
                q_net       = q_net,
                v_net       = v_net,
                state       = S_aug[alive_idx],     # (B_alive, 6)
                n_latents   = n_latents,
                ddpm_steps  = ddpm_steps,
                value_eps   = value_eps,
            )

            # ── исполняем latent «exec_horizon» шагов ────────────────────
            for _ in range(exec_horizon):
                for i, k in enumerate(alive_idx.tolist()):
                    if done[k]:
                        continue

                    # действие по латенту
                    action = beta_vae.decode_action(
                        S_aug[k].unsqueeze(0),           # (1, 6)
                        best_z[i].unsqueeze(0)           # (1, z_dim)
                    ).squeeze(0).cpu().numpy()

                    next_s, reward, terminated, _ = envs[k].step(action)

                    if reward:                  # sparse reward at goal
                        succ[k]   = True
                        raw_score_sum += MAX_STEPS - env_step
                        done[k]   = True

                    # обновляем состояния
                    S_phys[k] = torch.tensor(next_s, device=torch_device)
                    S_aug[k]  = augment_state(
                        S_phys[k:k+1], goal[k:k+1]).squeeze(0)

                    if render and k == 0:
                        envs[k].render()

                env_step += 1
                if env_step >= MAX_STEPS or done.all():
                    break  # пора перепланировать или закончить эпизод

        # ── статистика batch-а ─────────────────────────────────────────────
        successes += succ.sum().item()
        total      = (batch + 1) * num_parallel_envs
        print(f"[{total:3d}/{num_evals}]  success {successes}"
              f"  ({successes / total * 100:5.1f} %)")

    # ── финальный вывод метрики D4RL ────────────────────────────────────────
    avg_raw    = raw_score_sum / num_evals
    d4rl_score = envs[0].get_normalized_score(avg_raw) * 100
    print(f"\nRaw return: {avg_raw:.1f} | Normalised D4RL: {d4rl_score:.2f}")
    return d4rl_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="maze2d-umaze-v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--episodes", type=int, default=100)          # num_evals
    parser.add_argument("--parallel_envs", type=int, default=10)      # num_parallel_envs
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--beta_vae_path", type=str, default="beta_vae.pt")
    parser.add_argument("--diffusion_path", type=str, default="diffusion_model.pt")
    parser.add_argument("--q_net_path", type=str, default=None)
    parser.add_argument("--v_net_path", type=str, default=None)
    args = parser.parse_args()

    # --------------------------------------------------------------------- #
    #  создаём несколько независимых сред                                   #
    # --------------------------------------------------------------------- #
    envs = [gym.make(args.env) for _ in range(args.parallel_envs)]
    env_sample = envs[0]                         # одна для датасета / метрик

    # --------------------------------------------------------------------- #
    #  готовим датасет и статистики                                         #
    # --------------------------------------------------------------------- #
    dataset = env_sample.get_dataset()
    obs   = dataset["observations"]              # (N, 4)
    goal  = dataset["infos/goal"]                # (N, 2)
    obs_aug = np.concatenate([obs, goal - obs[:, :2]], axis=1)  # (N, 6)
    dataset["observations"] = obs_aug

    state_dim  = obs_aug.shape[1]
    action_dim = dataset["actions"].shape[1]

    # для возможной нормализации (eval-функция пока не использует,
    # но передаём, чтобы сигнатура совпадала)
    state_mean = obs_aug.mean(axis=0)
    state_std  = obs_aug.std(axis=0) + 1e-6
    latent_mean = np.zeros(args.latent_dim, dtype=np.float32)
    latent_std  = np.ones(args.latent_dim, dtype=np.float32)

    # --------------------------------------------------------------------- #
    #  загружаем модели                                                     #
    # --------------------------------------------------------------------- #
    device = torch.device(args.device)

    beta_vae = BetaVAE(state_dim, action_dim, args.latent_dim).to(device)
    beta_vae.load_state_dict(torch.load(args.beta_vae_path, map_location=device))

    diffusion_model = LatentDiffusionUNet(args.latent_dim, state_dim).to(device)
    diffusion_model.load_state_dict(torch.load(args.diffusion_path, map_location=device))

    q_net = None
    if args.q_net_path:
        q_net = DoubleQNet(state_dim, args.latent_dim).to(device)
        q_net.load_state_dict(torch.load(args.q_net_path, map_location=device))

    v_net = None
    if args.v_net_path:
        v_net = ValueNet(state_dim).to(device)
        v_net.load_state_dict(torch.load(args.v_net_path, map_location=device))

    # --------------------------------------------------------------------- #
    #  ставим в режим eval и «замораживаем» параметры                       #
    # --------------------------------------------------------------------- #
    for net in (beta_vae, diffusion_model, q_net, v_net):
        if net is not None:
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)

    # --------------------------------------------------------------------- #
    #  оценка                                                               #
    # --------------------------------------------------------------------- #
    evaluate_diar_policy(
        envs            = envs,
        beta_vae        = beta_vae,
        diffusion_model = diffusion_model,
        q_net           = q_net,
        v_net           = v_net,
        # state_mean      = state_mean,
        # state_std       = state_std,
        # latent_mean     = latent_mean,
        # latent_std      = latent_std,
        num_evals       = args.episodes,
        num_parallel_envs = args.parallel_envs,
        device          = args.device,
        # render=True,
    )
