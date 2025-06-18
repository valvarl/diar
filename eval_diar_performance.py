import torch
import gym
import d4rl
import numpy as np
import argparse
from diar_d4rl_maze2d import (
    BetaVAE,
    LatentDiffusionModel,
    ddpm_sample,
    policy_execute,
)

def evaluate_diar_policy(env, beta_vae, diffusion_model, q_net=None, v_net=None, episodes=100, device="cuda"):
    rewards = []
    for ep in range(episodes):
        reward = policy_execute(
            env,
            q_net=q_net,
            v_net=v_net,
            beta_vae=beta_vae,
            diffusion_model=diffusion_model,
            steps=30,
            revaluation_attempts=3,
            device=device
        )
        rewards.append(reward)
    rewards = np.array(rewards)
    normalized_scores = env.get_normalized_score(rewards) * 100
    print(f"Raw Return: {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"D4RL Normalized Score: {normalized_scores.mean():.2f} ± {normalized_scores.std():.2f}")
    return rewards, normalized_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="maze2d-umaze-v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--beta_vae_path", type=str, default="beta_vae.pt")
    parser.add_argument("--diffusion_path", type=str, default="diffusion_model.pt")
    parser.add_argument("--q_net_path", type=str, default=None)
    parser.add_argument("--v_net_path", type=str, default=None)
    args = parser.parse_args()

    env = gym.make(args.env)
    dataset = env.get_dataset()
    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]

    beta_vae = BetaVAE(state_dim, action_dim, args.latent_dim).to(args.device)
    beta_vae.load_state_dict(torch.load(args.beta_vae_path))

    diffusion_model = LatentDiffusionModel(args.latent_dim, state_dim).to(args.device)
    diffusion_model.load_state_dict(torch.load(args.diffusion_path))

    q_net = v_net = None
    if args.q_net_path:
        from diar_d4rl_maze2d import DoubleQNet
        q_net = DoubleQNet(state_dim, args.latent_dim).to(args.device)
        q_net.load_state_dict(torch.load(args.q_net_path))

    if args.v_net_path:
        from diar_d4rl_maze2d import ValueNet
        v_net = ValueNet(state_dim).to(args.device)
        v_net.load_state_dict(torch.load(args.v_net_path))

    evaluate_diar_policy(env, beta_vae, diffusion_model, q_net, v_net, episodes=args.episodes, device=args.device)