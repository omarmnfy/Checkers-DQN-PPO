import numpy as np
from checkers_game import CheckersGame

class CheckersEnv:
    def __init__(self):
        self.game = CheckersGame()
        self.current_player = CheckersGame.RED
        self.done = False
        self.total_reward = 0.0  # Accumulated reward from Red's perspective
        self.red_pieces_lost = 0
        self.black_pieces_lost = 0
        self.total_pieces_lost = 0
        self.total_pieces_captured = 0
        self.total_loss_reward = 0.0
        self.total_capture_reward = 0.0

    def reset(self):
        """Reset the environment to initial state"""
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
        return self._get_observation()

    def step(self, action):
        """Execute action and return observation, reward, done, and info"""
        if self.done:
            return self._get_observation(), 0.0, True, {}

        # Record piece counts
        old_counts = self._count_pieces()
        # Apply move
        self.game.make_move(action)
        new_counts = self._count_pieces()

        # Compute reward from Red's perspective always
        reward = self._calculate_reward(old_counts, new_counts)
        self.total_reward += reward

        # Check if game ended
        self.done = self.game.is_game_over()

        if not self.done:
            # Switch current player
            self.current_player = (
                CheckersGame.BLACK
                if self.current_player == CheckersGame.RED
                else CheckersGame.RED
            )
            self.game.current_player = self.current_player
            # If next player has no valid moves, end game
            if not self.game.get_valid_moves(self.current_player):
                self.done = True

        return self._get_observation(), reward, self.done, {}

    def render(self):
        """Print the current board state"""
        state = self.game.get_state()
        symbols = {
            CheckersGame.EMPTY: '.',
            CheckersGame.RED: 'r',
            CheckersGame.BLACK: 'b',
            CheckersGame.RED_KING: 'R',
            CheckersGame.BLACK_KING: 'B',
        }
        print('  ' + ' '.join(map(str, range(8))))
        for i, row in enumerate(state):
            print(f"{i} " + ' '.join(symbols[cell] for cell in row))

    def _get_observation(self):
        return self.game.get_state()

    def _count_pieces(self):
        state = self.game.get_state()
        return {
            CheckersGame.RED: np.sum((state == CheckersGame.RED) | (state == CheckersGame.RED_KING)),
            CheckersGame.BLACK: np.sum((state == CheckersGame.BLACK) | (state == CheckersGame.BLACK_KING)),
        }

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
