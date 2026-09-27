"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      ✅ MANDATORY IMPLEMENTATION #1 ✅                      ║
║                                                                              ║
║                        TABULAR Q-LEARNING AGENT                              ║
║                                                                              ║
║  This is one of the TWO REQUIRED algorithms for the project.                 ║
║                                                                              ║
║  Project Requirements:                                                       ║
║    1. ✅ Tabular Q-Learning (THIS FILE)                                      ║
║    2. ✅ Simple DQN with single network (dqn_agent.py)                       ║
║                                                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Tabular Q-Learning Agent Implementation

This module implements the classic tabular Q-learning algorithm, which is one of the
fundamental reinforcement learning algorithms. Q-learning is a model-free, off-policy
TD (Temporal Difference) learning algorithm.

Algorithm:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

Where:
    - Q(s,a): Quality value of taking action a in state s
    - α (alpha): Learning rate (how much new information overrides old)
    - γ (gamma): Discount factor (importance of future rewards)
    - r: Immediate reward
    - s': Next state
"""

import numpy as np
from collections import defaultdict

def _default_q_values(n_actions):
    """
    Helper function to create default Q-values (picklable)

    Initializes Q-values optimistically (10.0) to encourage exploration
    of new state-action pairs during early training.

    Args:
        n_actions (int): Number of possible actions

    Returns:
        dict: Dictionary mapping actions to initial Q-values
    """
    return {a: 10.0 for a in range(n_actions)}

class QLearningAgent:
    """
    Tabular Q-Learning Agent (MANDATORY REQUIREMENT)

    This agent uses a table to store Q-values for each state-action pair.
    The table is built incrementally as the agent explores the environment.

    Key Features:
        - Epsilon-greedy exploration strategy
        - State discretization for continuous observations
        - Off-policy learning (learns optimal policy while following ε-greedy)

    Attributes:
        n_actions (int): Number of possible actions (11 in warehouse environment)
        alpha (float): Learning rate [0,1]
        gamma (float): Discount factor [0,1]
        epsilon (float): Exploration rate [0,1]
        q_table (dict): State-action value table
    """

    def __init__(self, n_actions=11, learning_rate=0.1, discount=0.95,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        """
        Initialize Q-Learning Agent

        Args:
            n_actions (int): Number of actions in the action space
            learning_rate (float): Learning rate (α) - controls update magnitude
            discount (float): Discount factor (γ) - importance of future rewards
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Factor to decay epsilon after each episode
            epsilon_min (float): Minimum epsilon value (ensures some exploration)
        """
        self.n_actions = n_actions
        self.alpha = learning_rate      # α: controls how much we update Q-values
        self.gamma = discount            # γ: balances immediate vs future rewards
        self.epsilon = epsilon           # ε: probability of random action
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> {action: q_value}
        # Using a regular dict instead of defaultdict for pickle compatibility
        # This will be populated as the agent explores the environment
        self.q_table = {}
    
    def discretize_state(self, observation):
        """
        Convert continuous observation to discrete state tuple

        Tabular Q-learning requires discrete states. This method extracts the most
        important features from the high-dimensional observation (20x20x12) and
        converts them into a manageable discrete state representation.

        State Features:
            1. Robot X position (0-19)
            2. Robot Y position (0-19)
            3. Battery level bin (0-4): [0-20%, 20-40%, 40-60%, 60-80%, 80-100%]
            4. Human nearby flag (0 or 1): Safety consideration

        This creates a maximum of 20 × 20 × 5 × 2 = 4,000 possible states,
        which is manageable for tabular methods.

        Args:
            observation (np.ndarray): Environment observation (grid_size, grid_size, 12)

        Returns:
            tuple: Discrete state representation (x, y, battery_bin, human_nearby)
        """
        # Channel 0: Robot position - find where robot is located on grid
        robot_channel = observation[:, :, 0]
        robot_pos = tuple(np.argwhere(robot_channel == 1.0)[0]) if np.any(robot_channel) else (0, 0)

        # Channel 1: Battery level - discretize into 5 bins for manageability
        battery_val = observation[0, 0, 1]  # Uniform across grid
        battery_bin = int(battery_val * 5)  # 0-4 (5 bins)

        # Channel 3: Human nearby - binary flag for safety
        # Checks if any humans are present in the environment
        human_channel = observation[:, :, 3]
        human_nearby = int(np.any(human_channel > 0))

        # Simplified state representation: (x, y, battery_bin, human_nearby)
        state = (robot_pos[0], robot_pos[1], battery_bin, human_nearby)
        return state
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection

        This implements the exploration-exploitation trade-off:
        - With probability ε: explore (random action)
        - With probability (1-ε): exploit (greedy action based on Q-values)

        The epsilon value decays over time, shifting from exploration to exploitation.

        Args:
            state: Current state (can be raw observation or discrete state)
            training (bool): If True, uses epsilon-greedy; if False, always greedy

        Returns:
            int: Selected action index (0-10)
        """
        # Auto-discretize if raw observation is passed
        if isinstance(state, np.ndarray) and state.ndim == 3:
            state = self.discretize_state(state)

        # Exploration: Random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        # Exploitation: Best action according to Q-table
        else:
            # Initialize state if not seen before (optimistic initialization)
            if state not in self.q_table:
                self.q_table[state] = _default_q_values(self.n_actions)
            # Return action with highest Q-value
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update rule (core algorithm)

        Updates Q-value for the state-action pair using the Bellman equation:
            Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        This is an off-policy algorithm - it learns the optimal policy (greedy)
        while following an exploratory policy (epsilon-greedy).

        Args:
            state: Current state
            action (int): Action taken
            reward (float): Reward received
            next_state: Resulting state
            done (bool): Whether episode terminated

        The update has three components:
            1. TD target: r + γ max_a' Q(s',a')
            2. TD error: TD_target - Q(s,a)
            3. Update: Q(s,a) += α * TD_error
        """
        # Auto-discretize if raw observations are passed
        if isinstance(state, np.ndarray) and state.ndim == 3:
            state = self.discretize_state(state)
        if isinstance(next_state, np.ndarray) and next_state.ndim == 3:
            next_state = self.discretize_state(next_state)

        # Initialize states if not seen before
        if state not in self.q_table:
            self.q_table[state] = _default_q_values(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = _default_q_values(self.n_actions)

        # Find best action in next state (greedy policy)
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)

        # Temporal Difference (TD) target: r + γ max_a' Q(s',a')
        # If done, there is no next state, so future reward is 0
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)

        # TD error: How much our estimate was wrong
        td_error = td_target - self.q_table[state][action]

        # Update Q-value: move towards the target by learning rate α
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """
        Decay exploration rate

        Gradually reduces epsilon to shift from exploration to exploitation.
        Ensures epsilon never goes below epsilon_min to maintain minimal exploration.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """
        Save Q-Learning model to disk

        Saves the Q-table and hyperparameters using pickle serialization.
        The Q-table can be quite large depending on state space exploration.

        Args:
            path (str): File path to save the model (.pkl extension)
        """
        import pickle
        with open(path, 'wb') as f:
            model_data = {
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'n_actions': self.n_actions,
                'alpha': self.alpha,
                'gamma': self.gamma
            }
            pickle.dump(model_data, f)
        print(f"[SAVE] Saving Q-Learning model to {path}...")
        print(f"[SAVE] Model saved successfully! Q-table size: {len(self.q_table)} states")

    def load(self, path):
        """
        Load Q-Learning model from disk

        Restores the Q-table and hyperparameters from a saved file.

        Args:
            path (str): File path to load the model from
        """
        import pickle
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.q_table = model_data['q_table']
        self.epsilon = model_data.get('epsilon', self.epsilon_min)
        self.n_actions = model_data.get('n_actions', self.n_actions)
        print(f"[LOAD] Q-Learning model loaded from {path}")
        print(f"       Q-table size: {len(self.q_table)} states")