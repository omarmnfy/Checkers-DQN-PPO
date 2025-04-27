import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, env, memory_size=10000):
        self.env = env
        self.state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self.target_model = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves=None):
        if valid_moves is not None and len(valid_moves) == 0:
            return None

        if random.random() <= self.epsilon:
            if valid_moves is not None:
                return random.choice(valid_moves)
            return self.env.action_space.sample()

        state_tensor = torch.FloatTensor(self.preprocess_state(state)).to(self.device)
        with torch.no_grad():
            action_values = self.model(state_tensor)

        if valid_moves is not None:
            # Create a mapping of valid moves to their indices
            move_to_index = {move: self.action_to_index(move) for move in valid_moves}
            valid_indices = list(move_to_index.values())
            
            # Get Q-values only for valid moves
            valid_q_values = action_values[valid_indices]
            best_valid_index = valid_indices[valid_q_values.argmax().item()]
            
            # Find the move corresponding to the best valid index
            for move, idx in move_to_index.items():
                if idx == best_valid_index:
                    return move
        else:
            return action_values.argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(self.preprocess_state(state)).to(self.device)
            next_state_tensor = torch.FloatTensor(self.preprocess_state(next_state)).to(self.device)

            with torch.no_grad():
                target = self.model(state_tensor)
                target_next = self.target_model(next_state_tensor)

            target_val = target.clone()
            if done:
                target_val[self.action_to_index(action)] = reward
            else:
                target_val[self.action_to_index(action)] = reward + self.gamma * torch.max(target_next)

            states.append(state_tensor)
            targets.append(target_val)

        states = torch.stack(states)
        targets = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def preprocess_state(self, state):
        # Flatten the state and normalize
        return state.flatten() / 4.0  # Divide by 4 since we have 5 possible values (0-4)

    def action_to_index(self, action):
        # Convert action tuple to index
        if isinstance(action, tuple):
            start, end = action
            start_row, start_col = start
            end_row, end_col = end
            return start_row * 512 + start_col * 64 + end_row * 8 + end_col
        return action 