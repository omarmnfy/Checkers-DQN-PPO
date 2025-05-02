import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Suppress pygame window

import sys
# Suppress SDL/macOS window-move warnings
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


def plot_learning_curve(win_rates, window_size=10, title="Learning Curve"):
    if not win_rates:
        return
    window_size = min(window_size, len(win_rates))
    plt.figure(figsize=(10, 6))
    plt.plot(win_rates, alpha=0.3, label='Raw Win Rate')
    moving_avg = np.convolve(win_rates, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(win_rates)), moving_avg, label=f'MA (w={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()


def train(n_episodes=2000, update_freq=20, visualize=False, n_envs=10, max_steps=400):
    # Create multiple environments and agents
    envs = [CheckersEnv() for _ in range(n_envs)]
    ppo_agent = PPOAgent()  # Single PPO agent shared across environments
    dqn_agent = DQNAgent()  # Single DQN agent shared across environments
    visualizer = CheckersVisualizer() if visualize else None
    
    # Lists to track wins for each environment
    ppo_wins_per_env = [[] for _ in range(n_envs)]
    dqn_wins_per_env = [[] for _ in range(n_envs)]
    # Lists for overall win rates
    ppo_win_rates = []
    dqn_win_rates = []
    
    try:
        for ep in range(n_episodes):
            print(f"\nStarting Episode {ep}")
            
            # Initialize states for all environments
            states = [env.reset() for env in envs]
            ppo_total_rewards = [0 for _ in range(n_envs)]
            dqn_total_rewards = [0 for _ in range(n_envs)]
            steps = [0 for _ in range(n_envs)]
            dones = [False for _ in range(n_envs)]
            
            # Track which environments are still active
            active_envs = list(range(n_envs))
            
            while active_envs:
                # Check if any environment has reached max steps
                for env_idx in active_envs[:]:
                    if steps[env_idx] >= max_steps:
                        print(f"\nMax steps ({max_steps}) reached in env {env_idx}")
                        dones[env_idx] = True
                        active_envs.remove(env_idx)
                        continue

                if visualize and visualizer:  # Only handle visualization if enabled
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    visualizer.draw_board(states[0])
                    time.sleep(0.1)

                # Process each active environment
                for env_idx in active_envs[:]:  # Copy list to allow removal during iteration
                    env = envs[env_idx]
                    state = states[env_idx]
                    step = steps[env_idx]

                    moves = env.game.get_valid_moves(env.current_player)
                    print(f"\nEnv {env_idx}, Step {step}")
                    print(f"Current player: {'RED' if env.current_player == CheckersGame.RED else 'BLACK'}")
                    print(f"Number of valid moves: {len(moves)}")
                    
                    if not moves:
                        print("No valid moves available")
                        dones[env_idx] = True
                        active_envs.remove(env_idx)
                        continue

                    # PPO plays as RED, DQN plays as BLACK
                    if env.current_player == CheckersGame.RED:
                        print(f"PPO (RED) selecting action in env {env_idx}")
                        action, value, prob = ppo_agent.act(state, moves)
                        if action is None:
                            print("PPO failed to select action")
                            dones[env_idx] = True
                            active_envs.remove(env_idx)
                            continue
                    else:
                        print(f"DQN (BLACK) selecting action in env {env_idx}")
                        action = dqn_agent.act(state, moves)
                        if action is None:
                            print("DQN failed to select action")
                            dones[env_idx] = True
                            active_envs.remove(env_idx)
                            continue

                    print(f"Selected action: {action}")
                    next_state, reward, done, _ = env.step(action)
                    print(f"Reward: {reward}")
                    print(f"Game done: {done}")
                    
                    # Store experience for respective agent
                    if env.current_player == CheckersGame.RED:
                        ppo_agent.remember(state, action, reward, value, prob, done)
                        ppo_total_rewards[env_idx] += reward
                        print(f"PPO total reward in env {env_idx}: {ppo_total_rewards[env_idx]}")
                    else:
                        dqn_agent.remember(state, action, reward, next_state, done)
                        dqn_total_rewards[env_idx] += reward
                        print(f"DQN total reward in env {env_idx}: {dqn_total_rewards[env_idx]}")
                    
                    states[env_idx] = next_state
                    steps[env_idx] += 1

                    # Update agents periodically
                    if steps[env_idx] % update_freq == 0 or done:
                        print(f"Updating agents from env {env_idx}")
                        if env.current_player == CheckersGame.RED:
                            next_value = 0 if done else ppo_agent.act(state, env.game.get_valid_moves(env.current_player))[1]
                            ppo_agent.update(next_value)
                        else:
                            dqn_agent.replay(32)  # DQN uses batch size of 32

                    if done:
                        if visualize and visualizer and env_idx == 0:  # Only visualize first environment if enabled
                            visualizer.draw_board(next_state)
                            pygame.display.flip()
                            time.sleep(2)
                            
                        print(f"\nGame Over in env {env_idx}! Episode: {ep}")
                        print(f"PPO (RED) Total Reward: {ppo_total_rewards[env_idx]}")
                        print(f"DQN (BLACK) Total Reward: {dqn_total_rewards[env_idx]}")
                        
                        # Compute final piece counts
                        red_count = env.game.count_pieces(CheckersGame.RED)
                        black_count = env.game.count_pieces(CheckersGame.BLACK)
                        print(f"Final Pieces - RED: {red_count}, BLACK: {black_count}")
                        
                        # Determine winner and update win counts
                        if red_count > black_count:
                            print(f"RED (PPO) WON in env {env_idx}!")
                            ppo_wins_per_env[env_idx].append(1)
                            dqn_wins_per_env[env_idx].append(0)
                        else:
                            print(f"BLACK (DQN) WON in env {env_idx}!")
                            ppo_wins_per_env[env_idx].append(0)
                            dqn_wins_per_env[env_idx].append(1)
                        
                        dones[env_idx] = True
                        active_envs.remove(env_idx)

            # Calculate average win rates across all environments
            ppo_win_rate = sum(sum(wins) for wins in ppo_wins_per_env) / sum(len(wins) for wins in ppo_wins_per_env)
            dqn_win_rate = sum(sum(wins) for wins in dqn_wins_per_env) / sum(len(wins) for wins in dqn_wins_per_env)
            ppo_win_rates.append(ppo_win_rate)
            dqn_win_rates.append(dqn_win_rate)

            # Save best models based on win rates
            if ppo_win_rate > max(ppo_win_rates[:-1], default=0):
                torch.save(ppo_agent.network.state_dict(), 'best_ppo_model.pth')
                print(f"New best PPO model saved with win rate: {ppo_win_rate}")
            
            if dqn_win_rate > max(dqn_win_rates[:-1], default=0):
                torch.save(dqn_agent.model.state_dict(), 'best_dqn_model.pth')
                print(f"New best DQN model saved with win rate: {dqn_win_rate}")
            
            # Plot learning curves every 10 episodes
            if ep % 10 == 0:
                print(f"Episode: {ep}")
                print(f"Average PPO Win Rate: {ppo_win_rate:.3f}")
                print(f"Average DQN Win Rate: {dqn_win_rate:.3f}")
                plot_learning_curve(ppo_win_rates, title="PPO Average Win Rate")
                plot_learning_curve(dqn_win_rates, title="DQN Average Win Rate")
                
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Show final learning curves
        plot_learning_curve(ppo_win_rates, title="PPO Average Win Rate")
        plot_learning_curve(dqn_win_rates, title="DQN Average Win Rate")
        if visualize and visualizer:
            visualizer.close()


if __name__ == '__main__':
    # To toggle visualization:
    # - Set visualize=True to see the game board and animations
    # - Set visualize=False for faster training without visualization
    train(visualize=False, n_envs=10, max_steps=400)  # Will stop each game after 400 steps
