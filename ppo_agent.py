import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPONetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        features = self.shared(x)
        policy = self.policy(features)
        value = self.value(features)
        return policy, value

class PPOAgent:
    def __init__(self, state_size=64, action_size=4096):  # 8*8*8*8 possible moves
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.epsilon = 0.2
        self.epochs = 4
        self.learning_rate = 0.0003
        
        # Initialize network and optimizer
        self.network = PPONetwork(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Memory buffers
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.action_indices = []  # Store indices instead of action tuples
        self.rewards = []
        self.values = []
        self.action_probs = []
        self.dones = []

    def preprocess_state(self, state):
        return torch.FloatTensor(state.flatten() / 4.0).to(self.device)

    def action_to_index(self, action):
        # Convert action tuple ((start_row, start_col), (end_row, end_col)) to single index
        (start_row, start_col), (end_row, end_col) = action
        return start_row * 512 + start_col * 64 + end_row * 8 + end_col

    def index_to_action(self, index):
        # Convert index back to action tuple
        start_row = index // 512
        start_col = (index % 512) // 64
        end_row = (index % 64) // 8
        end_col = index % 8
        return ((start_row, start_col), (end_row, end_col))

    def remember(self, state, action, reward, value, action_prob, done):
        self.states.append(state)
        self.action_indices.append(self.action_to_index(action))
        self.rewards.append(reward)
        self.values.append(value)
        self.action_probs.append(action_prob)
        self.dones.append(done)

    def act(self, state, valid_moves):
        if not valid_moves:  # If no valid moves
            return None
        
        try:
            state_tensor = self.preprocess_state(state)
            
            with torch.no_grad():
                policy, value = self.network(state_tensor)
            
            # Create mask for valid moves
            valid_indices = [self.action_to_index(move) for move in valid_moves]
            mask = torch.zeros_like(policy)
            mask[valid_indices] = 1
            masked_policy = policy * mask
            
            # Check if any valid moves have non-zero probability
            if masked_policy.sum().item() <= 1e-10:
                # If all probabilities are zero, use uniform distribution over valid moves
                masked_policy = mask / len(valid_moves)
            else:
                # Normalize the masked policy
                masked_policy = masked_policy / masked_policy.sum()
            
            # Sample action from the masked policy
            dist = Categorical(masked_policy)
            action_idx = dist.sample()
            
            # Convert index back to move
            for move, idx in zip(valid_moves, valid_indices):
                if idx == action_idx.item():
                    return move, value.item(), masked_policy[action_idx].item()
            
            # If no move found (shouldn't happen, but just in case)
            return None, None, None
            
        except Exception as e:
            print(f"Error in act method: {e}")
            return None, None, None

    def compute_gae(self, next_value):
        gae = 0
        returns = []
        
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_return = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_return = self.values[step + 1]
                
            delta = self.rewards[step] + self.gamma * next_return * next_non_terminal - self.values[step]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae
            returns.insert(0, gae + self.values[step])
            
        return returns

    def update(self, next_value):
        if len(self.states) == 0:
            return
            
        returns = self.compute_gae(next_value)
        
        states = torch.FloatTensor(np.array([s.flatten() / 4.0 for s in self.states])).to(self.device)
        actions = torch.LongTensor(self.action_indices).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_probs = torch.FloatTensor(self.action_probs).to(self.device)
        
        for _ in range(self.epochs):
            # Get current policy and values
            policy, values = self.network(states)
            values = values.squeeze()
            
            # Calculate ratio and surrogate loss
            dist = Categorical(policy)
            new_probs = torch.exp(dist.log_prob(actions))
            ratio = new_probs / (old_probs + 1e-10)
            
            # Calculate surrogate losses
            advantages = returns - values.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = 0.5 * (returns - values).pow(2).mean()
            
            # Calculate total loss
            total_loss = policy_loss + value_loss
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        self.clear_memory() 