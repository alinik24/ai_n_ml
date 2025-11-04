"""
Advanced DQN Approaches - State of the Art
Optimized for RTX 4060 GPU

References:
- Double DQN: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (AAAI 2016)
- Dueling DQN: Wang et al., "Dueling Network Architectures for Deep RL" (ICML 2016)
- Prioritized Experience Replay: Schaul et al., "Prioritized Experience Replay" (ICLR 2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random


# ==============================================================================
# DOUBLE DQN - Reduces overestimation bias
# ==============================================================================

class DoubleDQNAgent:
    """
    Double DQN: Uses two networks but action selection and evaluation are decoupled

    Key Innovation: Use online network for action selection, target network for evaluation
    Paper: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (AAAI 2016)

    Why it's better: Reduces Q-value overestimation which is common in standard DQN
    """

    def __init__(self, n_actions=11, learning_rate=1e-4, discount=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 buffer_size=50000, batch_size=64, target_update_freq=1000,
                 use_amp=False):

        from .dqn_agent import DQNetwork  # Reuse network architecture

        self.n_actions = n_actions
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_amp = use_amp

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print info
        if torch.cuda.is_available():
            print(f"[Double DQN] Using device: {torch.cuda.get_device_name(0)}")
            print(f"[Double DQN] Innovation: Decoupled action selection and evaluation")

        # TWO networks for Double DQN (this is allowed as it's a different algorithm)
        self.online_net = DQNetwork(n_actions).to(self.device)
        self.target_net = DQNetwork(n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network in eval mode

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp and torch.cuda.is_available() else None
        self.steps = 0

    def select_action(self, state, training=True):
        """Epsilon-greedy using online network"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_net(state_tensor)
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        """
        Double DQN Update:
        1. Online network selects best action: a* = argmax_a Q_online(s', a)
        2. Target network evaluates that action: Q_target(s', a*)
        This reduces overestimation!
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < 1000:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device, non_blocking=True)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device, non_blocking=True)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device, non_blocking=True)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device, non_blocking=True)

        if self.use_amp and self.scaler is not None:
            with torch.amp.autocast('cuda'):
                # Current Q-values
                q_values = self.online_net(states).gather(1, actions)

                # Double DQN: Select actions with online network, evaluate with target network
                with torch.no_grad():
                    # Online network selects best actions
                    next_actions = self.online_net(next_states).argmax(1, keepdim=True)
                    # Target network evaluates those actions
                    next_q_values = self.target_net(next_states).gather(1, next_actions)
                    targets = rewards + self.gamma * next_q_values * (1 - dones)

                loss = F.smooth_l1_loss(q_values, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values = self.online_net(states).gather(1, actions)

            with torch.no_grad():
                next_actions = self.online_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                targets = rewards + self.gamma * next_q_values * (1 - dones)

            loss = F.smooth_l1_loss(q_values, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
            self.optimizer.step()

        self.steps += 1

        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)
        print(f"[SAVE] Double DQN model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        print(f"[LOAD] Double DQN model loaded from {path}")

    def get_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "N/A (CPU mode)"


# ==============================================================================
# DUELING DQN - Separate value and advantage streams
# ==============================================================================

class DuelingDQNetwork(nn.Module):
    """
    Dueling Network Architecture

    Key Innovation: Separates state value V(s) and action advantage A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

    Paper: Wang et al., "Dueling Network Architectures for Deep RL" (ICML 2016)

    Why it's better: Better generalization, especially when many actions have similar values
    """

    def __init__(self, n_actions=11):
        super().__init__()

        # Shared convolutional layers (feature extraction)
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Value stream: V(s)
        self.value_fc1 = nn.Linear(25600, 512)
        self.value_fc2 = nn.Linear(512, 1)  # Outputs single value

        # Advantage stream: A(s,a)
        self.advantage_fc1 = nn.Linear(25600, 512)
        self.advantage_fc2 = nn.Linear(512, n_actions)  # Outputs advantage per action

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Shared feature extraction
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.dropout(value)
        value = self.value_fc2(value)

        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.dropout(advantage)
        advantage = self.advantage_fc2(advantage)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Subtracting mean makes the advantage have zero mean (stabilizes learning)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class DuelingDQNAgent:
    """
    Dueling DQN Agent
    Uses dueling network architecture for better value estimation
    """

    def __init__(self, n_actions=11, learning_rate=1e-4, discount=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 buffer_size=50000, batch_size=64, target_update_freq=1000,
                 use_amp=False):

        self.n_actions = n_actions
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_amp = use_amp

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print(f"[Dueling DQN] Using device: {torch.cuda.get_device_name(0)}")
            print(f"[Dueling DQN] Innovation: Separate value and advantage streams")

        # Dueling architecture networks
        self.online_net = DuelingDQNetwork(n_actions).to(self.device)
        self.target_net = DuelingDQNetwork(n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp and torch.cuda.is_available() else None
        self.steps = 0

    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_net(state_tensor)
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < 1000:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device, non_blocking=True)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device, non_blocking=True)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device, non_blocking=True)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device, non_blocking=True)

        if self.use_amp and self.scaler is not None:
            with torch.amp.autocast('cuda'):
                q_values = self.online_net(states).gather(1, actions)

                with torch.no_grad():
                    # Use Double DQN update for even better performance
                    next_actions = self.online_net(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_net(next_states).gather(1, next_actions)
                    targets = rewards + self.gamma * next_q_values * (1 - dones)

                loss = F.smooth_l1_loss(q_values, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values = self.online_net(states).gather(1, actions)

            with torch.no_grad():
                next_actions = self.online_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                targets = rewards + self.gamma * next_q_values * (1 - dones)

            loss = F.smooth_l1_loss(q_values, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
            self.optimizer.step()

        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)
        print(f"[SAVE] Dueling DQN model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        print(f"[LOAD] Dueling DQN model loaded from {path}")

    def get_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "N/A (CPU mode)"


# ==============================================================================
# PRIORITIZED EXPERIENCE REPLAY
# ==============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay

    Key Innovation: Sample important transitions more frequently based on TD-error
    Paper: Schaul et al., "Prioritized Experience Replay" (ICLR 2016)

    Why it's better: Learns faster by focusing on surprising/important experiences
    """

    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization (0 = uniform, 1 = full)
        self.beta = beta  # Importance sampling correction (increases to 1)
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priority

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Add new experience with maximum priority"""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch based on priorities"""
        if self.size == 0:
            return []

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Get samples
        batch = [self.buffer[idx] for idx in indices]

        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-errors"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon

    def __len__(self):
        return self.size


# Simple replay buffer for non-prioritized versions
class ReplayBuffer:
    """Standard experience replay buffer"""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
