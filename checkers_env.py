import numpy as np
from checkers_game import CheckersGame
import matplotlib.pyplot as plt

class CheckersEnv:
    def __init__(self):
        self.game = CheckersGame()
        self.current_player = CheckersGame.RED
        self.done = False
        
    def reset(self):
        """Reset the environment to initial state"""
        self.game.reset()
        self.current_player = CheckersGame.RED
        self.done = False
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

        # Calculate reward
        reward = self._calculate_reward(old_pieces, new_pieces)

        # Check if game is over
        self.done = self.game.is_game_over()

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
            pieces_lost = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]
            pieces_captured = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]
            opponent_pieces_remaining = new_pieces[CheckersGame.BLACK]
        else:
            pieces_lost = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]
            pieces_captured = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]
            opponent_pieces_remaining = new_pieces[CheckersGame.RED]

        reward = 0
        # Penalty for losing pieces
        reward -= pieces_lost * 1.0  # -1.0 for each piece lost
        # Reward for capturing opponent's pieces
        reward += pieces_captured * 4.0  # +4.0 points for each capture
        
        # Check for game over
        if self.done:
            if opponent_pieces_remaining == 0:
                reward += 100.0  # Big reward for eliminating all opponent's pieces
            elif len(self.game.get_valid_moves(self.current_player)) == 0:
                reward -= 100.0  # Changed from -50.0 to -100.0 for losing
            else:
                reward += 100.0  # Won (opponent has no valid moves)

        return reward

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