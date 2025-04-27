import numpy as np
import gymnasium as gym
from gymnasium import spaces
from checkers_game import CheckersGame

class CheckersGymEnv(gym.Env):
    """Gymnasium wrapper around your CheckersGame logic."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.env = CheckersGame()

        # Observation: 8×8 grid of 0–4
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(8, 8), dtype=np.int8
        )

        # Action: all possible (r1,c1)->(r2,c2) tuples
        self.action_list = [
            ((r1, c1), (r2, c2))
            for r1 in range(8)
            for c1 in range(8)
            for r2 in range(8)
            for c2 in range(8)
        ]
        self.action_space = spaces.Discrete(len(self.action_list))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        self.env.current_player = CheckersGame.RED
        return state, {}

    def step(self, action_idx):
        move = self.action_list[action_idx]
        obs, reward, done, _ = self.env.step(move)
        return obs, reward, done, False, {}

    def render(self):
        self.env.render()

    def close(self):
        pass
