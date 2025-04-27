import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import sys
# Suppress SDL/macOS window-move warnings
sys.stderr = open(os.devnull, 'w')

import numpy as np
import time
import torch
import pygame
import matplotlib.pyplot as plt
import gymnasium as gym

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
    ppo_agent = PPOAgent(env)
    dqn_agent = DQNAgent(env)
    visualizer = CheckersVisualizer()
    
    ppo_rewards = []
    dqn_rewards = []
    best_ppo_reward = float('-inf')
    best_dqn_reward = float('-inf')
    
    try:
        for ep in range(n_episodes):
            print(f"\nStarting Episode {ep}")
            state, _ = env.reset()
            ppo_total_reward = 0
            dqn_total_reward = 0
            step = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                visualizer.draw_board(state)
                time.sleep(0.1)

                moves = env.game.get_valid_moves(env.current_player)
                print(f"\nStep {step}")
                print(f"Current player: {'RED' if env.current_player == CheckersGame.RED else 'BLACK'}")
                print(f"Number of valid moves: {len(moves)}")
                
                if not moves:
                    print("No valid moves available")
                    done = True
                    continue

                # PPO plays as RED, DQN plays as BLACK
                if env.current_player == CheckersGame.RED:
                    print("PPO (RED) selecting action")
                    action, value, prob = ppo_agent.act(state, moves)
                    if action is None:
                        print("PPO failed to select action")
                        done = True
                        continue
                else:
                    print("DQN (BLACK) selecting action")
                    action = dqn_agent.act(state, moves)
                    if action is None:
                        print("DQN failed to select action")
                        done = True
                        continue

                print(f"Selected action: {action}")
                next_state, reward, done, truncated, info = env.step(action)
                print(f"Reward: {reward}")
                print(f"Game done: {done}")
                
                # Store experience for respective agent
                if env.current_player == CheckersGame.RED:
                    ppo_agent.remember(state, action, reward, value, prob, done)
                    ppo_total_reward += reward
                    print(f"PPO total reward: {ppo_total_reward}")
                else:
                    dqn_agent.remember(state, action, reward, next_state, done)
                    dqn_total_reward += reward
                    print(f"DQN total reward: {dqn_total_reward}")
                
                state = next_state
                step += 1

                # Update agents periodically
                if step % update_freq == 0 or done:
                    print("Updating agents")
                    if env.current_player == CheckersGame.RED:
                        next_value = 0 if done else ppo_agent.act(state, env.game.get_valid_moves(env.current_player))[1]
                        ppo_agent.update(next_value)
                    else:
                        dqn_agent.replay(32)  # DQN uses batch size of 32

                if done:
                    visualizer.draw_board(state)
                    print(f"\nGame Over! Episode: {ep}")
                    print(f"PPO (RED) Total Reward: {ppo_total_reward}")
                    print(f"DQN (BLACK) Total Reward: {dqn_total_reward}")
                    
                    # Compute final piece counts
                    red_count = env.game.count_pieces(CheckersGame.RED)
                    black_count = env.game.count_pieces(CheckersGame.BLACK)
                    print(f"Final Pieces - RED: {red_count}, BLACK: {black_count}")
                    
                    # Determine winner
                    if red_count > black_count:
                        print("RED (PPO) WON!")
                    else:
                        print("BLACK (DQN) WON!")
                    
                    pygame.display.flip()
                    time.sleep(2)

            # Store rewards
            ppo_rewards.append(ppo_total_reward)
            dqn_rewards.append(dqn_total_reward)
            
            # Save best models
            if ppo_total_reward > best_ppo_reward:
                best_ppo_reward = ppo_total_reward
                torch.save(ppo_agent.network.state_dict(), 'best_ppo_model.pth')
                print(f"New best PPO model saved: {best_ppo_reward}")
            
            if dqn_total_reward > best_dqn_reward:
                best_dqn_reward = dqn_total_reward
                torch.save(dqn_agent.model.state_dict(), 'best_dqn_model.pth')
                print(f"New best DQN model saved: {best_dqn_reward}")
            
            # Plot learning curves every 10 episodes
            if ep % 10 == 0:
                print(f"Episode: {ep}")
                print(f"PPO Reward: {ppo_total_reward}")
                print(f"DQN Reward: {dqn_total_reward}")
                plot_learning_curve(ppo_rewards, title="PPO Learning Curve")
                plot_learning_curve(dqn_rewards, title="DQN Learning Curve")
                
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Show final learning curves
        plot_learning_curve(ppo_rewards, title="PPO Learning Curve")
        plot_learning_curve(dqn_rewards, title="DQN Learning Curve")
        visualizer.close()


if __name__ == '__main__':
    train()
