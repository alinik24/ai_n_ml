"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      ✅ MANDATORY IMPLEMENTATION #2 ✅                      ║
║                                                                              ║
║                  DEEP Q-NETWORK (DQN) - SINGLE NETWORK                       ║
║                                                                              ║
║  This is one of the TWO REQUIRED algorithms for the project.                 ║
║                                                                              ║
║  Project Requirements:                                                       ║
║    1. ✅ Tabular Q-Learning (q_learning_agent.py)                           ║
║    2. ✅ Simple DQN with SINGLE network (THIS FILE)                         ║
║                                                                              ║
║                                                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Deep Q-Network (DQN) Agent Implementation - SINGLE NETWORK VERSION

This module implements the simplest form of Deep Q-Network as required by the project.
It uses a SINGLE neural network to approximate the Q-function.

Key Differences from Tabular Q-Learning:
    1. Neural network approximates Q(s,a) instead of table lookup
    2. Can handle high-dimensional state spaces (20x20x6 grid)
    3. Uses experience replay for stable learning
    4. Gradient descent instead of direct value updates

Algorithm:
    1. Store experience (s, a, r, s') in replay buffer
    2. Sample random minibatch from buffer
    3. Compute target: y = r + γ max_a' Q(s', a')  [SAME network for both!]
    4. Update network to minimize: L = (Q(s,a) - y)²
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

class DQNetwork(nn.Module):
    """
    Deep Q-Network Architecture

    Network Structure:
        Input: (batch, 6, 20, 20) - 6-channel grid observation
        Conv Layers: Extract spatial features from warehouse layout
        FC Layers: Map features to Q-values for each action

    Architecture:
        - Conv1: 6 -> 32 channels (3x3 kernel)
        - Conv2: 32 -> 64 channels (3x3 kernel)
        - Conv3: 64 -> 64 channels (3x3 kernel)
        - Flatten: 20x20x64 = 25,600 features
        - FC1: 25,600 -> 512
        - FC2: 512 -> 256
        - FC3: 256 -> 11 (Q-values for 11 actions)
    """

    def __init__(self, n_actions=11):
        """
        Initialize DQN architecture

        Args:
            n_actions (int): Number of actions in action space (11)
        """
        super().__init__()

        # Convolutional layers for spatial feature extraction
        # These learn to recognize patterns like walls, zones, obstacles
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)   # Input channels: 6
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Increase capacity
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Deep features

        # Fully connected layers for action value computation
        # Calculate flattened size: 20x20x64 = 25600
        self.fc1 = nn.Linear(25600, 512)       # Feature extraction
        self.fc2 = nn.Linear(512, 256)          # Abstract representation
        self.fc3 = nn.Linear(256, n_actions)    # Q-value outputs

        # Dropout for regularization (prevent overfitting)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input observation (batch, 20, 20, 6)

        Returns:
            torch.Tensor: Q-values for each action (batch, n_actions)
        """
        # Reshape from (batch, height, width, channels) to (batch, channels, height, width)
        # PyTorch Conv2D expects channels first
        x = x.permute(0, 3, 1, 2)

        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))  # Extract low-level features
        x = F.relu(self.conv2(x))  # Extract mid-level features
        x = F.relu(self.conv3(x))  # Extract high-level features

        # Flatten spatial dimensions: (batch, 64, 20, 20) -> (batch, 25600)
        # Use reshape instead of view for better compatibility with non-contiguous tensors
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))    # Dense feature representation
        x = self.dropout(x)         # Regularization during training
        x = F.relu(self.fc2(x))    # Abstract features
        q_values = self.fc3(x)      # Output Q-values for each action

        return q_values


class ReplayBuffer:
    """
    Experience Replay Buffer

    Stores past experiences and samples random minibatches for training.
    This breaks correlation between consecutive samples and improves stability.

    Why Experience Replay?
        1. Breaks temporal correlations in observation sequence
        2. Allows reuse of past experiences (data efficiency)
        3. Smooths out learning (reduces variance)

    The buffer uses a deque with fixed capacity, automatically discarding
    oldest experiences when full (FIFO).
    """

    def __init__(self, capacity=50000):
        """
        Initialize replay buffer

        Args:
            capacity (int): Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer

        Args:
            state: Current state observation
            action (int): Action taken
            reward (float): Reward received
            next_state: Next state observation
            done (bool): Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample random minibatch from buffer

        Args:
            batch_size (int): Number of experiences to sample

        Returns:
            list: Random sample of experiences
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent - SINGLE NETWORK IMPLEMENTATION (Simplest Form)

    *** IMPORTANT: This is NOT Double DQN ***
    As per project requirements, this uses only ONE neural network for both:
        1. Action selection: Q(s, a)
        2. Target computation: max_a' Q(s', a')

    This can lead to overestimation of Q-values but is the simplest DQN form.

    Key Components:
        - Single Q-Network: Approximates Q(s,a) for all actions
        - Experience Replay: Stores and reuses past experiences
        - Epsilon-Greedy: Balances exploration vs exploitation
        - Gradient Descent: Updates network weights via backpropagation

    Attributes:
        q_network (DQNetwork): Single neural network (NOT two networks!)
        replay_buffer (ReplayBuffer): Experience replay memory
        optimizer (Adam): Network optimizer
        scaler (GradScaler): For mixed precision training (optional)
    """

    def __init__(self, n_actions=11, learning_rate=1e-4, discount=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 buffer_size=50000, batch_size=64,
                 use_amp=False):
        """
        Initialize DQN Agent with SINGLE network

        Args:
            n_actions (int): Number of actions in action space
            learning_rate (float): Learning rate for Adam optimizer
            discount (float): Discount factor γ for future rewards
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay factor per episode
            epsilon_min (float): Minimum epsilon value
            buffer_size (int): Capacity of replay buffer
            batch_size (int): Minibatch size for training
            use_amp (bool): Use automatic mixed precision (faster on modern GPUs)
        """
        self.n_actions = n_actions
        self.gamma = discount              # Discount factor γ
        self.epsilon = epsilon             # Exploration rate ε
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.use_amp = use_amp  # Automatic Mixed Precision for faster training

        # Device setup (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print hardware info
        if torch.cuda.is_available():
            print(f"[GPU] Using device: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] CUDA Version: {torch.version.cuda}")
            print(f"[GPU] Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            if self.use_amp:
                print(f"[GPU] Mixed Precision Training: ENABLED")
        else:
            print(f"[CPU] No GPU available, using CPU")

        # *** SINGLE NETWORK (as required) ***
        # No separate target network in this implementation
        self.q_network = DQNetwork(n_actions).to(self.device)

        # Enable cuDNN auto-tuner for faster convolutions
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Adam optimizer: adaptive learning rates for each parameter
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Mixed precision training scaler (for GPU acceleration)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp and torch.cuda.is_available() else None

        self.steps = 0  # Training step counter
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection using SINGLE network

        Similar to tabular Q-learning, but uses neural network for Q-values.

        Args:
            state (np.ndarray): Current state observation
            training (bool): If True, uses epsilon-greedy; if False, greedy only

        Returns:
            int: Selected action
        """
        # Exploration: random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        # Exploitation: action with highest Q-value from network
        else:
            with torch.no_grad():  # No gradient computation for inference
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)  # Get Q(s,a) for all actions
                return q_values.argmax().item()  # Return action with max Q-value
    
    def update(self, state, action, reward, next_state, done):
        """
        DQN Update using SINGLE network (simplest form)

        *** KEY POINT: Uses SAME network for both Q(s,a) and max Q(s',a') ***
        This is different from Double DQN which uses two networks.

        Training Process:
            1. Store experience in replay buffer
            2. Sample random minibatch (breaks correlation)
            3. Compute current Q-values: Q(s,a)
            4. Compute target: y = r + γ max_a' Q(s',a')  [using SAME network]
            5. Minimize loss: L = (Q(s,a) - y)²
            6. Update weights via backpropagation

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        # Store experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Wait until we have enough experiences
        if len(self.replay_buffer) < 1000:
            return

        # Sample random minibatch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and transfer to device (GPU/CPU)
        # non_blocking=True allows asynchronous GPU transfer
        states = torch.FloatTensor(np.array(states)).to(self.device, non_blocking=True)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device, non_blocking=True)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device, non_blocking=True)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device, non_blocking=True)

        # Two training paths: with/without mixed precision
        # Mixed precision = faster training on modern GPUs, same results

        if self.use_amp and self.scaler is not None:
            # MIXED PRECISION TRAINING PATH
            with torch.amp.autocast('cuda'):  # Use FP16 for forward pass
                # Step 1: Compute current Q-values: Q(s,a) using SINGLE network
                q_values = self.q_network(states).gather(1, actions)

                # Step 2: Compute targets: y = r + γ max_a' Q(s',a')
                # *** Uses SAME network (not a separate target network!) ***
                with torch.no_grad():  # No gradients for target computation
                    next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1)
                    targets = rewards + self.gamma * next_q_values * (1 - dones)

                # Step 3: Compute loss (Huber loss = smooth L1)
                # More robust to outliers than MSE
                loss = F.smooth_l1_loss(q_values, targets)

            # Step 4: Backpropagation with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()  # Scale loss for FP16
            self.scaler.unscale_(self.optimizer)  # Unscale for gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Stabilize training
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # STANDARD TRAINING PATH (no mixed precision)
            # Step 1: Compute Q(s,a) using SINGLE network
            q_values = self.q_network(states).gather(1, actions)

            # Step 2: Compute targets using SAME network (simplest DQN form)
            with torch.no_grad():
                next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1)
                targets = rewards + self.gamma * next_q_values * (1 - dones)

            # Step 3: Compute loss
            loss = F.smooth_l1_loss(q_values, targets)

            # Step 4: Backpropagation
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