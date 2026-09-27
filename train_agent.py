import argparse

import numpy as np
import torch

from dqn_agent import DQNAgent
from env_advanced import CarbonAwareComputeEnv


def train(episodes=500, episode_length=60, output="carbon_agent.pt"):
    env = CarbonAwareComputeEnv(episode_length=episode_length, seed=0)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{episodes} | average reward: {np.mean(rewards[-50:]):.2f}")

    torch.save(agent.q.state_dict(), output)
    print(f"Saved trained policy to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the carbon-aware scheduling policy")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=60)
    parser.add_argument("--output", default="carbon_agent.pt")
    args = parser.parse_args()
    train(args.episodes, args.episode_length, args.output)
