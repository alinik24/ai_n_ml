import numpy as np
from collections import defaultdict

def _default_q_values(n_actions):
    """Helper function to create default Q-values (picklable)"""
    return {a: 10.0 for a in range(n_actions)}

class QLearningAgent:
    """Tabular Q-Learning Agent (MANDATORY)"""

    def __init__(self, n_actions=11, learning_rate=0.1, discount=0.95,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> {action: q_value}
        # Using a regular dict instead of defaultdict for pickle compatibility
        self.q_table = {}
    
    def discretize_state(self, observation):
        """
        Convert continuous observation to discrete state tuple
        Extract key features from observation
        """
        # Channel 0: Robot position
        robot_channel = observation[:, :, 0]
        robot_pos = tuple(np.argwhere(robot_channel == 1.0)[0]) if np.any(robot_channel) else (0, 0)
        
        # Channel 1: Battery level (discretize into 5 bins)
        battery_val = observation[0, 0, 1]  # Uniform across grid
        battery_bin = int(battery_val * 5)  # 0-4
        
        # Channel 3: Human nearby (check adjacent cells)
        human_channel = observation[:, :, 3]
        human_nearby = int(np.any(human_channel > 0))
        
        # Simplified state representation
        state = (robot_pos[0], robot_pos[1], battery_bin, human_nearby)
        return state
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Initialize state if not seen before
            if state not in self.q_table:
                self.q_table[state] = _default_q_values(self.n_actions)
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        # Initialize states if not seen before
        if state not in self.q_table:
            self.q_table[state] = _default_q_values(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = _default_q_values(self.n_actions)

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save Q-Learning model"""
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
        print(f"[SAVE] Q-Learning model saved to {path}")

    def load(self, path):
        """Load Q-Learning model"""
        import pickle
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.q_table = model_data['q_table']
        self.epsilon = model_data.get('epsilon', self.epsilon_min)
        self.n_actions = model_data.get('n_actions', self.n_actions)
        print(f"[LOAD] Q-Learning model loaded from {path}")
        print(f"       Q-table size: {len(self.q_table)} states")