import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import sys
sys.stderr = open(os.devnull, 'w')

import numpy as np
import time
import torch
import pygame
import matplotlib.pyplot as plt

from checkers_env import CheckersEnv
from ppo_agent import PPOAgent
from dqn_agent import DQNAgent
from visualizer import CheckersVisualizer
from checkers_game import CheckersGame

def plot_learning_curve(episode_rewards, window_size=10, title="Learning Curve"):
    if not episode_rewards:
        return
    window_size = min(window_size, len(episode_rewards))
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Raw Rewards')
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, label=f'MA (w={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def train(n_episodes=200, update_freq=20):
    env = CheckersEnv()
    ppo_agent = PPOAgent()
    dqn_agent = DQNAgent()
    visualizer = CheckersVisualizer()

    ppo_rewards = []
    dqn_rewards = []
    best_ppo_reward = float('-inf')
    best_dqn_reward = float('-inf')

    try:
        for ep in range(n_episodes):
            print(f"\nStarting Episode {ep}")
            state, _ = env.reset()
            ppo_total = 0
            dqn_total = 0
            step = 0
            done = False

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                visualizer.draw_board(state)
                time.sleep(0.1)

                moves = env.game.get_valid_moves(env.current_player)
                if not moves:
                    done = True
                    continue

                if env.current_player == CheckersGame.RED:
                    action, value, prob = ppo_agent.act(state, moves)
                    if action is None:
                        done = True
                        continue
                else:
                    action = dqn_agent.act(state, moves)
                    if action is None:
                        done = True
                        continue

                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                if env.current_player == CheckersGame.RED:
                    ppo_agent.remember(state, action, reward, value, prob, done)
                    ppo_total += reward
                else:
                    dqn_agent.remember(state, action, reward, next_state, done)
                    dqn_total += reward

                state = next_state
                step += 1

                if step % update_freq == 0 or done:
                    if env.current_player == CheckersGame.RED:
                        next_val = 0 if done else ppo_agent.act(state, env.game.get_valid_moves(env.current_player))[1]
                        ppo_agent.update(next_val)
                    else:
                        dqn_agent.replay(32)

            ppo_rewards.append(ppo_total)
            dqn_rewards.append(dqn_total)

            # Save best models
            if ppo_total > best_ppo_reward:
                best_ppo_reward = ppo_total
                torch.save(ppo_agent.network.state_dict(), 'best_ppo_model.pth')
            if dqn_total > best_dqn_reward:
                best_dqn_reward = dqn_total
                torch.save(dqn_agent.model.state_dict(), 'best_dqn_model.pth')

            if ep % 10 == 0:
                plot_learning_curve(ppo_rewards, title="PPO Learning Curve")
                plot_learning_curve(dqn_rewards, title="DQN Learning Curve")

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        plot_learning_curve(ppo_rewards, title="PPO Learning Curve")
        plot_learning_curve(dqn_rewards, title="DQN Learning Curve")
        env.close()
        visualizer.close()

if __name__ == '__main__':
    train()
