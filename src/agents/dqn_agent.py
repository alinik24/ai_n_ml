import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

class DQNetwork(nn.Module):
    """Deep Q-Network (MANDATORY)"""
    
    def __init__(self, n_actions=11):
        super().__init__()
        
        # Convolutional layers for grid processing
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate flattened size: 20x20x64 = 25600
        self.fc1 = nn.Linear(25600, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x shape: (batch, 20, 20, 12) -> (batch, 12, 20, 20)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten - use reshape instead of view for non-contiguous tensors
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent - Simplest Form with SINGLE Neural Network
    (Requirement: use single NN, not two!)
    """

    def __init__(self, n_actions=11, learning_rate=1e-4, discount=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 buffer_size=50000, batch_size=64,
                 use_amp=False):

        self.n_actions = n_actions
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.use_amp = use_amp  # Automatic Mixed Precision for faster training

        # Device setup with detailed info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print GPU info
        if torch.cuda.is_available():
            print(f"[GPU] Using device: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] CUDA Version: {torch.version.cuda}")
            print(f"[GPU] Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            if self.use_amp:
                print(f"[GPU] Mixed Precision Training: ENABLED")
        else:
            print(f"[CPU] No GPU available, using CPU")

        # SINGLE NETWORK (as per requirement!)
        self.q_network = DQNetwork(n_actions).to(self.device)
        print(f"[DQN] Using SINGLE network (requirement compliant)")

        # Enable cuDNN optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # For mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp and torch.cuda.is_available() else None

        self.steps = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection using SINGLE network"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)  # Single network!
                return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """
        Train using SINGLE network (simplest DQN form)
        No target network as per requirements!
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < 1000:
            return

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Transfer to GPU with non_blocking for faster transfers
        states = torch.FloatTensor(np.array(states)).to(self.device, non_blocking=True)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device, non_blocking=True)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device, non_blocking=True)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device, non_blocking=True)

        # Mixed precision training
        if self.use_amp and self.scaler is not None:
            with torch.amp.autocast('cuda'):
                # Compute Q(s,a) using SINGLE network
                q_values = self.q_network(states).gather(1, actions)

                # Compute target using SAME network (not separate target network!)
                with torch.no_grad():
                    next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1)
                    targets = rewards + self.gamma * next_q_values * (1 - dones)

                # Huber loss
                loss = F.smooth_l1_loss(q_values, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training with SINGLE network
            q_values = self.q_network(states).gather(1, actions)

            # Use SAME network for next Q-values (simplest DQN)
            with torch.no_grad():
                next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1)
                targets = rewards + self.gamma * next_q_values * (1 - dones)

            loss = F.smooth_l1_loss(q_values, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

        self.steps += 1
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """Save model (SINGLE network)"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)
        print(f"[SAVE] DQN model (single network) saved to {path}")

    def load(self, path):
        """Load model with proper device mapping"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        print(f"[LOAD] DQN model loaded from {path}")

    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "N/A (CPU mode)"