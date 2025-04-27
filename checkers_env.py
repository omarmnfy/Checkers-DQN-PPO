import numpy as np
import gymnasium as gym
from gymnasium import spaces
from checkers_game import CheckersGame

class CheckersEnv(gym.Env):
    """Gymnasium-compatible Checkers environment."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
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

        # Define action & observation spaces
        self.action_list = self._all_possible_actions()
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(8, 8), dtype=np.int8
        )

    def _all_possible_actions(self):
        actions = []
        for r1 in range(8):
            for c1 in range(8):
                for r2 in range(8):
                    for c2 in range(8):
                        actions.append(((r1, c1), (r2, c2)))
        return actions

    def reset(self, *, seed=None, options=None):
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

    def step(self, action_idx):
        move = self.action_list[action_idx]
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        old_counts = self._count_pieces()
        self.game.make_move(move)
        new_counts = self._count_pieces()

        reward = self._calculate_reward(old_counts, new_counts)
        self.total_reward += reward

        self.done = self.game.is_game_over()
        if not self.done:
            # switch player
            self.current_player = (
                CheckersGame.BLACK
                if self.current_player == CheckersGame.RED
                else CheckersGame.RED
            )
            self.game.current_player = self.current_player
            if not self.game.get_valid_moves(self.current_player):
                self.done = True

        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        self.game.render()

    def close(self):
        pass

    def _get_observation(self):
        return self.game.get_state()

    def _count_pieces(self):
        state = self.game.get_state()
        return {
            CheckersGame.RED: np.sum(
                (state == CheckersGame.RED) | (state == CheckersGame.RED_KING)
            ),
            CheckersGame.BLACK: np.sum(
                (state == CheckersGame.BLACK) | (state == CheckersGame.BLACK_KING)
            ),
        }

    def _calculate_reward(self, old_pieces, new_pieces):
        """(Unchanged original reward logic pasted in full)"""
        # Determine current/opponent counts
        if self.current_player == CheckersGame.RED:
            pieces_lost = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]
            pieces_captured = (
                old_pieces[CheckersGame.BLACK] - new_counts[CheckersGame.BLACK]
            )
            opponent = CheckersGame.BLACK
            current_remaining = new_pieces[CheckersGame.RED]
            opponent_remaining = new_pieces[CheckersGame.BLACK]
        else:
            pieces_lost = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]
            pieces_captured = (
                old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]
            )
            opponent = CheckersGame.RED
            current_remaining = new_pieces[CheckersGame.BLACK]
            opponent_remaining = new_pieces[CheckersGame.RED]

        # Update loss/capture counters
        red_loss = old_pieces[CheckersGame.RED] - new_pieces[CheckersGame.RED]
        black_loss = old_pieces[CheckersGame.BLACK] - new_pieces[CheckersGame.BLACK]
        self.red_pieces_lost += red_loss
        self.black_pieces_lost += black_loss
        self.total_pieces_lost += pieces_lost
        self.total_pieces_captured += pieces_captured

        loss_reward = -4.0 * (red_loss if self.current_player == CheckersGame.RED else black_loss)
        capture_reward = 8.0 * pieces_captured
        self.total_loss_reward += loss_reward
        self.total_capture_reward += capture_reward

        base_reward = loss_reward + capture_reward

        if not self.done:
            return base_reward

        # Game-over bonus/penalty
        accumulated = self.total_loss_reward + self.total_capture_reward
        if current_remaining == 0 or not self.game.get_valid_moves(self.current_player):
            # loss
            self.total_reward = accumulated - 100.0
            return self.total_reward
        elif opponent_remaining == 0 or not self.game.get_valid_moves(opponent):
            # win
            self.total_reward = accumulated + 500.0
            return self.total_reward

        # fallback
        return accumulated
