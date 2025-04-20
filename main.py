from checkers_env import CheckersEnv
from ppo_agent import PPOAgent
from visualizer import CheckersVisualizer
from checkers_game import CheckersGame
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import pygame

def plot_learning_curve(episode_rewards, window_size=10):
    if len(episode_rewards) == 0:
        print("No episodes to plot")
        return
        
    if len(episode_rewards) < window_size:
        window_size = len(episode_rewards)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Raw Rewards')
    
    # Calculate moving average
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, label=f'Moving Average (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.close()

def train():
    env = CheckersEnv()
    agent = PPOAgent()
    visualizer = CheckersVisualizer()
    
    n_episodes = 200
    update_frequency = 20
    best_reward = float('-inf')
    episode_rewards = []
    running = True
    
    try:
        for episode in range(n_episodes):
            if not running:
                break
                
            state = env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            while not done and running:
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        raise KeyboardInterrupt
                    
                # Render the game
                visualizer.draw_board(state)
                time.sleep(0.1)
                
                # Get valid moves and select action
                valid_moves = env.game.get_valid_moves(env.current_player)
                
                # If no valid moves, end the game
                if not valid_moves:
                    done = True
                    continue
                
                action_tuple = agent.act(state, valid_moves)
                
                # If action selection failed, skip this turn
                if action_tuple is None:
                    env.current_player = CheckersGame.BLACK if env.current_player == CheckersGame.RED else CheckersGame.RED
                    continue
                    
                action, value, action_prob = action_tuple
                
                try:
                    # Take action
                    next_state, reward, done, _ = env.step(action)
                    
                    # Store experience
                    agent.remember(state, action, reward, value, action_prob, done)
                    
                    state = next_state
                    total_reward += reward
                    step_count += 1
                    
                    # Update policy if enough steps have been taken
                    if step_count % update_frequency == 0 or done:
                        if not done:
                            next_action_tuple = agent.act(next_state, 
                                env.game.get_valid_moves(env.current_player))
                            next_value = next_action_tuple[1] if next_action_tuple else 0
                        else:
                            next_value = 0
                            
                        agent.update(next_value)
                    
                    # If game is over, show final state and pause briefly
                    if done:
                        visualizer.draw_board(state)
                        print(f"\nGame Over! Episode: {episode}, Final reward: {total_reward}")
                        pygame.display.flip()
                        time.sleep(2)  # Show final state for 2 seconds
                
                except Exception as e:
                    print(f"Error during step: {e}")
                    done = True
            
            episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward}")
                # Update learning curve plot every 10 episodes
                if episode_rewards:
                    plot_learning_curve(episode_rewards)
                
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(agent.network.state_dict(), 'best_model.pth')
                print(f"New best model saved with reward: {best_reward}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Show final learning curve
        if episode_rewards:
            plot_learning_curve(episode_rewards)
        visualizer.close()

if __name__ == "__main__":
    train() 