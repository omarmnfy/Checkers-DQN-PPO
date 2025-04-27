# gym_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from checkers_game import CheckersGame

class CheckersGymEnv(gym.Env):
    """Gymnasium wrapper around your CheckersEnv logic."""
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.env = CheckersGame()
        # BOARD: 8×8 grid with values 0–4
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(8, 8), dtype=np.int8
        )
        # Each move is a tuple ((r1,c1),(r2,c2)), flatten to a discrete index
        # You could also use a MultiDiscrete, but simplest is:
        self.action_list = self._all_possible_actions()
        self.action_space = spaces.Discrete(len(self.action_list))

    def _all_possible_actions(self):
        """Precompute every legal move-slot; unused slots are ignored."""
        actions = []
        for r1 in range(8):
            for c1 in range(8):
                for r2 in range(8):
                    for c2 in range(8):
                        actions.append(((r1, c1), (r2, c2)))
        return actions

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        self.env.current_player = CheckersGame.RED
        return state, {}  # obs, info

    def step(self, action_idx):
        move = self.action_list[action_idx]
        obs, reward, done, _ = self.env.step(move)
        # Gymnasium’s API: return obs, reward, terminated, truncated, info
        # Here we never truncate, only terminate when done:
        return obs, reward, done, False, {}

    def render(self):
        self.env.render()

    def close(self):
        pass
