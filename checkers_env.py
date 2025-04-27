import numpy as np
from checkers_game import CheckersGame
import matplotlib.pyplot as plt

class CheckersEnv:
    def __init__(self):
        self.game = CheckersGame()
        self.current_player = CheckersGame.RED
        self.done = False
        self.total_reward = 0  # Track accumulated rewards
        self.total_pieces_lost = 0  # Track total pieces lost
        self.total_pieces_captured = 0  # Track total pieces captured
        
    def reset(self):
        """Reset the environment to initial state"""
        self.game.reset()
        self.current_player = CheckersGame.RED
        self.done = False
        self.total_reward = 0  # Reset accumulated rewards
        self.total_pieces_lost = 0  # Reset total pieces lost
        self.total_pieces_captured = 0  # Reset total pieces captured
        return self._get_observation()

    def step(self, action):
        """Execute action and return new state, reward, done, and info"""
        if self.done:
            return self._get_observation(), 0, True, {}

        # Store piece counts before move
        old_pieces = self._count_pieces()

        # Make the move
        self.game.make_move(action)

        # Get new piece counts
        new_pieces = self._count_pieces()

        # Check if game is over BEFORE calculating reward
        self.done = self.game.is_game_over()

        # Calculate reward
        reward = self._calculate_reward(old_pieces, new_pieces)

        # Switch players if game is not over
        if not self.done:
            self.current_player = (CheckersGame.BLACK 
                                 if self.current_player == CheckersGame.RED 
                                 else CheckersGame.RED)
            self.game.current_player = self.current_player

        return self._get_observation(), reward, self.done, {}

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

    def _calculate_reward(self, old_pieces, new_pieces):
        """Calculate reward based on piece difference and game outcome"""
        # Get piece difference
        if self.current_player == CheckersGame.RED:
            # For RED player
            pieces_lost = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]  # Positive if pieces lost
            pieces_captured = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]  # Positive if pieces captured
            opponent_pieces_remaining = new_pieces[CheckersGame.BLACK]
            current_pieces_remaining = new_pieces[CheckersGame.RED]
        else:
            # For BLACK player
            pieces_lost = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]  # Positive if pieces lost
            pieces_captured = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]  # Positive if pieces captured
            opponent_pieces_remaining = new_pieces[CheckersGame.RED]
            current_pieces_remaining = new_pieces[CheckersGame.BLACK]

        # Update total pieces lost and captured
        self.total_pieces_lost += pieces_lost
        self.total_pieces_captured += pieces_captured

        # Calculate base reward
        reward = 0
        # Penalty for losing pieces (pieces_lost is positive when pieces are lost)
        reward -= pieces_lost * 1.0  # This will be negative when pieces are lost
        # Reward for capturing opponent's pieces (pieces_captured is positive when pieces are captured)
        reward += pieces_captured * 4.0  # This will be positive when pieces are captured
        
        # Accumulate the base reward
        self.total_reward += reward
        
        print(f"\nDebug Reward Calculation:")
        print(f"Current Player: {'RED' if self.current_player == CheckersGame.RED else 'BLACK'}")
        print(f"Current Move - Pieces Lost: {pieces_lost}, Reward from loss: {-pieces_lost * 1.0}")
        print(f"Current Move - Pieces Captured: {pieces_captured}, Reward from capture: {pieces_captured * 4.0}")
        print(f"Total Game - Pieces Lost: {self.total_pieces_lost}, Total Loss Reward: {-self.total_pieces_lost * 1.0}")
        print(f"Total Game - Pieces Captured: {self.total_pieces_captured}, Total Capture Reward: {self.total_pieces_captured * 4.0}")
        print(f"Base Reward: {reward}")
        print(f"Total Accumulated Reward: {self.total_reward}")
        
        # If game is not over, return base reward
        if not self.done:
            return reward
        
        # Game is over - calculate final reward
        print(f"\nGame Over Conditions:")
        print(f"Current player pieces remaining: {current_pieces_remaining}")
        print(f"Opponent pieces remaining: {opponent_pieces_remaining}")
        print(f"Valid moves left: {len(self.game.get_valid_moves(self.current_player))}")
        
        # Check for losing conditions first
        if current_pieces_remaining == 0 or len(self.game.get_valid_moves(self.current_player)) == 0:
            print("Adding -100 for losing (no pieces or no valid moves)")
            final_reward = -100.0  # Fixed penalty for losing
        # Then check for winning conditions
        elif opponent_pieces_remaining == 0 or len(self.game.get_valid_moves(CheckersGame.BLACK if self.current_player == CheckersGame.RED else CheckersGame.RED)) == 0:
            print("Adding +1000 for winning (opponent has no pieces or no valid moves)")
            final_reward = 1000.0  # Fixed reward for winning
        else:
            # This should never happen, but just in case
            print("Unexpected game over condition")
            final_reward = 0.0
        
        # Add final reward to accumulated total
        self.total_reward += final_reward
        print(f"Final Total Reward (including game over): {self.total_reward}")
        return self.total_reward

    def render(self):
        """Display the current state of the board"""
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