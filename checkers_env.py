import numpy as np
import gymnasium as gym
from gymnasium import spaces
from checkers_game import CheckersGame

class CheckersEnv(gym.Env):
    """Checkers environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(CheckersEnv, self).__init__()
        self.game = CheckersGame()
        self.current_player = CheckersGame.RED
        self.done = False
        self.total_reward = 0.0
        self.red_pieces_lost = 0
        self.black_pieces_lost = 0
        self.total_pieces_lost = 0
        self.total_pieces_captured = 0
        self.total_loss_reward = 0.0
        self.total_capture_reward = 0.0

        # Define action and observation space
        # Action space: 4096 possible moves (8*8*8*8)
        self.action_space = spaces.Discrete(4096)
        
        # Observation space: 8x8 board with 5 possible values per cell
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(8, 8), dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.game.reset()
        self.current_player = CheckersGame.RED
        self.done = False
        self.total_reward = 0.0
        self.red_pieces_lost = 0
        self.black_pieces_lost = 0
        self.total_pieces_lost = 0
        self.total_pieces_captured = 0
        self.total_loss_reward = 0.0
        self.total_capture_reward = 0.0
        return self._get_observation(), {}

    def step(self, action):
        """Execute action and return new state, reward, done, and info"""
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        # Convert action index to move tuple
        action_tuple = self._index_to_action(action)
        
        # Store piece counts before move
        old_pieces = self._count_pieces()

        # Make the move
        self.game.make_move(action_tuple)

        # Get new piece counts
        new_pieces = self._count_pieces()

        # Calculate reward before checking game over
        reward = self._calculate_reward(old_pieces, new_pieces)

        # Check if game is over
        self.done = self.game.is_game_over()

        # If game is not over, switch players
        if not self.done:
            self.current_player = (CheckersGame.BLACK 
                                 if self.current_player == CheckersGame.RED 
                                 else CheckersGame.RED)
            self.game.current_player = self.current_player

            # Check if next player has no valid moves
            if len(self.game.get_valid_moves(self.current_player)) == 0:
                self.done = True
                # Recalculate reward with game over
                reward = self._calculate_reward(old_pieces, new_pieces)

        return self._get_observation(), reward, self.done, False, {}

    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            # Print the board state
            state = self.game.get_state()
            symbols = {
                CheckersGame.EMPTY: '.',
                CheckersGame.RED: 'r',
                CheckersGame.BLACK: 'b',
                CheckersGame.RED_KING: 'R',
                CheckersGame.BLACK_KING: 'B'
            }
            print('  0 1 2 3 4 5 6 7')
            for i in range(8):
                row = f'{i} '
                for j in range(8):
                    row += symbols[state[i][j]] + ' '
                print(row)
            print()
        elif mode == 'rgb_array':
            # Return RGB array for visualization
            state = self.game.get_state()
            rgb_array = np.zeros((8, 8, 3), dtype=np.uint8)
            # Map piece values to colors
            for i in range(8):
                for j in range(8):
                    if state[i][j] == CheckersGame.RED:
                        rgb_array[i][j] = [255, 0, 0]  # Red
                    elif state[i][j] == CheckersGame.BLACK:
                        rgb_array[i][j] = [0, 0, 0]  # Black
                    elif state[i][j] == CheckersGame.RED_KING:
                        rgb_array[i][j] = [255, 0, 0]  # Red
                    elif state[i][j] == CheckersGame.BLACK_KING:
                        rgb_array[i][j] = [0, 0, 0]  # Black
                    else:
                        rgb_array[i][j] = [255, 255, 255]  # White
            return rgb_array

    def close(self):
        """Clean up resources"""
        pass

    def _get_observation(self):
        """Get current board state"""
        return self.game.get_state()

    def _count_pieces(self):
        """Count pieces for both players"""
        state = self.game.get_state()
        return {
            CheckersGame.RED: (np.sum(state == CheckersGame.RED) + 
                             np.sum(state == CheckersGame.RED_KING)),
            CheckersGame.BLACK: (np.sum(state == CheckersGame.BLACK) + 
                               np.sum(state == CheckersGame.BLACK_KING))
        }

    def _index_to_action(self, index):
        """Convert action index to move tuple"""
        start_row = index // 512
        start_col = (index % 512) // 64
        end_row = (index % 64) // 8
        end_col = index % 8
        return ((start_row, start_col), (end_row, end_col))

    def _calculate_reward(self, old_pieces, new_pieces):
        """Calculate reward based on piece difference and game outcome"""
        # Get piece difference
        if self.current_player == CheckersGame.RED:
            # For RED player
            pieces_lost = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]  # Positive if pieces lost
            pieces_captured = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]  # Positive if pieces captured
            opponent_pieces_remaining = new_pieces[CheckersGame.BLACK]
            current_pieces_remaining = new_pieces[CheckersGame.RED]
            opponent = CheckersGame.BLACK
        else:
            # For BLACK player
            pieces_lost = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]  # Positive if pieces lost
            pieces_captured = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]  # Positive if pieces captured
            opponent_pieces_remaining = new_pieces[CheckersGame.RED]
            current_pieces_remaining = new_pieces[CheckersGame.BLACK]
            opponent = CheckersGame.RED

        # Update pieces lost for both players
        red_pieces_lost = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]
        black_pieces_lost = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]
        self.red_pieces_lost += red_pieces_lost
        self.black_pieces_lost += black_pieces_lost

        # Validate piece counts
        if self.red_pieces_lost > 12:
            print("Warning: RED pieces lost exceeds maximum (12)")
            self.red_pieces_lost = 12
        if self.black_pieces_lost > 12:
            print("Warning: BLACK pieces lost exceeds maximum (12)")
            self.black_pieces_lost = 12

        # Update total pieces lost and captured
        self.total_pieces_lost += pieces_lost
        self.total_pieces_captured += pieces_captured

        # Calculate rewards for this move
        # Black piece captured: +8.0
        # Red piece lost: -4.0
        if self.current_player == CheckersGame.RED:
            loss_reward = -red_pieces_lost * 4.0  # -4.0 per RED piece lost
            capture_reward = pieces_captured * 8.0  # +8.0 per BLACK piece captured
        else:
            loss_reward = -black_pieces_lost * 4.0  # -4.0 per BLACK piece lost
            capture_reward = pieces_captured * 8.0  # +8.0 per RED piece captured

        # Update total rewards
        self.total_loss_reward += loss_reward
        self.total_capture_reward += capture_reward

        # Calculate base reward for this move
        reward = loss_reward + capture_reward
        
        # Accumulate the base reward
        self.total_reward = self.total_loss_reward + self.total_capture_reward
        
        # If game is not over, return base reward
        if not self.done:
            return reward
        
        # Game is over - calculate final reward
        # Store accumulated rewards before adding game over reward
        accumulated_rewards = self.total_reward
        
        # Check for losing conditions first
        if current_pieces_remaining == 0 or len(self.game.get_valid_moves(self.current_player)) == 0:
            print("Adding -100 for losing (no pieces or no valid moves)")
            final_reward = -100.0  # Fixed penalty for losing
            # Add the game over reward to the accumulated rewards
            self.total_reward = accumulated_rewards - 100.0
            # Print who won/lost
            if self.current_player == CheckersGame.RED:
                print("\nRED LOST!")
                print(f"Final Reward Breakdown:")
                print(f"Accumulated Rewards (from pieces): {accumulated_rewards}")
                print(f"Game Over Reward (Losing): -100")
                print(f"Final Total Reward: {self.total_reward}")
            else:
                print("\nBLACK LOST!")
                print(f"Final Reward Breakdown:")
                print(f"Accumulated Rewards (from pieces): {accumulated_rewards}")
                print(f"Game Over Reward (Losing): -100")
                print(f"Final Total Reward: {self.total_reward}")
        # Then check for winning conditions
        elif opponent_pieces_remaining == 0 or len(self.game.get_valid_moves(opponent)) == 0:
            print("Adding +500 for winning (opponent has no pieces or no valid moves)")
            final_reward = 500.0  # Fixed reward for winning
            # Add the game over reward to the accumulated rewards
            self.total_reward = accumulated_rewards + 500.0
            # Print who won/lost
            if self.current_player == CheckersGame.RED:
                print("\nRED WON!")
                print(f"Final Reward Breakdown:")
                print(f"Accumulated Rewards (from pieces): {accumulated_rewards}")
                print(f"Game Over Reward (Winning): +500")
                print(f"Final Total Reward: {self.total_reward}")
            else:
                print("\nBLACK WON!")
                print(f"Final Reward Breakdown:")
                print(f"Accumulated Rewards (from pieces): {accumulated_rewards}")
                print(f"Game Over Reward (Winning): +500")
                print(f"Final Total Reward: {self.total_reward}")
        else:
            # This should never happen, but just in case
            print("Unexpected game over condition")
            final_reward = 0.0
        
        return self.total_reward
