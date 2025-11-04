# ============================================================================
# COMPLETE PROJECT STRUCTURE FOR WAREHOUSE ROBOT RL PROJECT
# ============================================================================

"""
Directory Structure:
====================

matricola-lastname/
│
├── report.pdf                    # 15-20 page report (use template provided)
│
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── warehouse_env.py      # Main Gymnasium environment
│   │   ├── entities.py           # Human, Forklift, Robot classes
│   │   ├── zones.py              # Zone definitions and checks
│   │   ├── battery_system.py    # Battery management logic
│   │   └── task_generator.py    # Task creation logic
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── q_learning_agent.py  # Tabular Q-Learning (MANDATORY)
│   │   ├── dqn_agent.py         # Deep Q-Network (MANDATORY)
│   │   ├── replay_buffer.py     # Experience replay for DQN
│   │   └── networks.py          # Neural network architectures
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_qlearning.py   # Q-Learning training script
│   │   ├── train_dqn.py         # DQN training script
│   │   └── curriculum.py        # Progressive difficulty
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py          # Run evaluation on trained agents
│   │   ├── metrics.py           # Calculate all metrics
│   │   └── visualize.py         # Generate plots
│   │
│   ├── experiments/
│   │   ├── exp1_baseline.py
│   │   ├── exp2_battery.py
│   │   ├── exp3_scalability.py
│   │   └── exp4_generalization.py
│   │
│   ├── results/
│   │   ├── figures/             # Save plots here
│   │   ├── metrics/             # CSV files
│   │   └── models/              # Trained weights
│   │
│   ├── requirements.txt
│   ├── README.md
│   └── config.yaml              # Hyperparameters
│
└── (optional) colab_notebook.ipynb  # Can link to this

"""

# ============================================================================
# FILE 1: src/environment/warehouse_env.py
# ============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
from typing import Tuple, Dict, List, Optional

class PackageType(Enum):
    NONE = 0
    STANDARD = 1
    FRAGILE = 2
    HEAVY = 3
    URGENT = 4

class ZoneType(Enum):
    CORRIDOR = 0
    STORAGE = 1
    RESTRICTED = 2
    CHARGING = 3
    PACKING = 4
    SHIPPING = 5
    RECEIVING = 6

class WarehouseAMREnv(gym.Env):
    """
    Enhanced Warehouse Environment with Real-World Constraints
    - Battery management
    - Multi-stage tasks
    - Zone authorization
    - Realistic human movement
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Configuration
        config = config or {}
        self.grid_size = config.get('grid_size', 20)
        self.max_steps = config.get('max_steps', 600)
        self.n_humans = config.get('n_humans', 3)
        self.n_forklifts = config.get('n_forklifts', 1)
        self.n_other_robots = config.get('n_other_robots', 2)
        
        # Battery parameters
        self.battery_max = 100
        self.battery_drain_move = 1.0
        self.battery_drain_heavy = 2.0
        self.battery_drain_idle = 0.5
        self.battery_charge_rate = 5.0
        self.battery_critical = 20
        
        # Define zones (each as (min_corner, max_corner))
        self.zones = {
            'storage_a': ((2, 2), (5, 5)),
            'storage_b': ((2, 8), (5, 11)),
            'storage_c': ((2, 14), (5, 17)),
            'receiving_1': ((0, 0), (1, 3)),
            'receiving_2': ((0, 17), (1, 19)),
            'packing_1': ((15, 2), (17, 5)),
            'packing_2': ((15, 8), (17, 11)),
            'shipping_local': ((18, 0), (19, 4)),
            'shipping_express': ((18, 7), (19, 11)),
            'shipping_intl': ((18, 14), (19, 19)),
            'charging_main': ((10, 0), (11, 2)),
            'charging_emergency': ((10, 17), (11, 19)),
            'restricted_office': ((7, 7), (9, 9)),
            'restricted_break': ((12, 7), (14, 9)),
            'restricted_qc': ((7, 14), (9, 16))
        }
        
        # Task library (pickup_zone -> delivery_zone)
        self.task_library = [
            {'pickup': 'receiving_1', 'delivery': 'storage_a', 'type': PackageType.STANDARD},
            {'pickup': 'receiving_2', 'delivery': 'storage_b', 'type': PackageType.HEAVY},
            {'pickup': 'storage_a', 'delivery': 'packing_1', 'type': PackageType.STANDARD},
            {'pickup': 'storage_b', 'delivery': 'packing_2', 'type': PackageType.FRAGILE},
            {'pickup': 'packing_1', 'delivery': 'shipping_local', 'type': PackageType.STANDARD},
            {'pickup': 'packing_2', 'delivery': 'shipping_express', 'type': PackageType.FRAGILE},
            {'pickup': 'storage_a', 'delivery': 'storage_c', 'type': PackageType.STANDARD},
            {'pickup': 'storage_c', 'delivery': 'shipping_express', 'type': PackageType.URGENT},
        ]
        
        # Spaces
        self.action_space = spaces.Discrete(11)  # 11 actions
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_size, self.grid_size, 12),
            dtype=np.float32
        )
        
        # State variables (initialized in reset())
        self.robot_pos = None
        self.battery = None
        self.carrying = None
        self.is_charging = False
        self.current_task = None
        self.task_stage = None
        self.steps = 0
        self.humans = []
        self.forklifts = []
        self.other_robots = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Robot state
        self.robot_pos = np.array([10, 10], dtype=int)
        self.battery = 100.0
        self.carrying = PackageType.NONE
        self.is_charging = False
        
        # Task
        self.current_task = self._generate_task()
        self.task_stage = 'pickup'
        self.steps = 0
        
        # Initialize entities
        self.humans = self._init_humans()
        self.forklifts = self._init_forklifts()
        self.other_robots = self._init_other_robots()
        
        observation = self._get_observation()
        info = {'battery': self.battery, 'task': self.current_task}
        
        return observation, info
    
    def step(self, action: int):
        self.steps += 1
        reward = 0.0
        
        # 1. Execute robot action
        action_reward = self._execute_action(action)
        reward += action_reward
        
        # 2. Update battery
        battery_drain = self._calculate_battery_drain(action)
        self.battery = max(0, self.battery - battery_drain)
        
        # Battery critical checks
        if self.battery == 0:
            reward -= 300
            terminated = True
            truncated = False
            obs = self._get_observation()
            info = {'reason': 'battery_depleted'}
            return obs, reward, terminated, truncated, info
        elif self.battery < self.battery_critical:
            reward -= 20
        
        # 3. Update all entities
        self._update_humans()
        self._update_forklifts()
        self._update_other_robots()
        
        # 4. Check collisions
        collision_penalty = self._check_collisions()
        reward += collision_penalty
        
        # Critical collision terminates episode
        terminated = collision_penalty <= -150
        
        # 5. Check zone violations
        if self._in_restricted_zone():
            reward -= 50
        
        # 6. Check task progress
        task_reward = self._check_task_progress()
        reward += task_reward
        
        # 7. Time penalty
        reward -= 1
        
        # 8. Check truncation
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = {
            'battery': self.battery,
            'task_stage': self.task_stage,
            'collision': collision_penalty < 0
        }
        
        return observation, reward, terminated, truncated, info
    
    def _generate_task(self) -> Dict:
        """Generate a random task from library"""
        task = self.np_random.choice(self.task_library).copy()
        # 30% chance for urgent
        if self.np_random.random() < 0.3:
            task['type'] = PackageType.URGENT
            task['deadline'] = self.steps + 200
        return task
    
    def _init_humans(self) -> List[Dict]:
        """Initialize human workers"""
        humans = []
        for i in range(self.n_humans):
            zone = self.np_random.choice(['storage_a', 'storage_b', 'packing_1', 'shipping_local'])
            pos = self._random_pos_in_zone(zone)
            humans.append({
                'id': i,
                'pos': pos,
                'target': None,
                'state': 'working',
                'idle_counter': 0
            })
        return humans
    
    def _init_forklifts(self) -> List[Dict]:
        """Initialize forklifts on patrol routes"""
        forklifts = []
        for i in range(self.n_forklifts):
            patrol_points = [(5, 10), (10, 10), (15, 10), (10, 10)]
            forklifts.append({
                'id': i,
                'pos': np.array(patrol_points[0]),
                'patrol': patrol_points,
                'patrol_idx': 0
            })
        return forklifts
    
    def _init_other_robots(self) -> List[Dict]:
        """Initialize other AMRs"""
        robots = []
        for i in range(self.n_other_robots):
            robots.append({
                'id': i,
                'pos': np.array([5, 5]),
                'target': np.array([15, 15])
            })
        return robots
    
    def _execute_action(self, action: int) -> float:
        """Execute robot action and return reward"""
        reward = 0.0
        
        # Movement actions (0-3)
        if action in [0, 1, 2, 3]:
            directions = [
                np.array([-1, 0]),  # North
                np.array([1, 0]),   # South
                np.array([0, 1]),   # East
                np.array([0, -1])   # West
            ]
            new_pos = self.robot_pos + directions[action]
            
            if self._is_valid_pos(new_pos):
                self.robot_pos = new_pos
                if self._moving_toward_goal(directions[action]):
                    reward += 3
            else:
                reward -= 10
        
        # Wait (4)
        elif action == 4:
            if not self._should_wait():
                reward -= 5
        
        # Pickup (5)
        elif action == 5:
            if self._at_pickup() and self.carrying == PackageType.NONE:
                self.carrying = self.current_task['type']
                self.task_stage = 'delivery'
                reward += 60
            else:
                reward -= 15
        
        # Drop (6)
        elif action == 6:
            if self._at_delivery() and self.carrying != PackageType.NONE:
                self.carrying = PackageType.NONE
                self.task_stage = 'completed'
                reward += 150
                # Generate new task
                self.current_task = self._generate_task()
                self.task_stage = 'pickup'
            else:
                reward -= 15
        
        # Start charging (7)
        elif action == 7:
            if self._at_charging_station():
                self.is_charging = True
                reward += 5
            else:
                reward -= 10
        
        # Stop charging (8)
        elif action == 8:
            if self.is_charging:
                self.is_charging = False
            else:
                reward -= 5
        
        return reward
    
    def _calculate_battery_drain(self, action: int) -> float:
        """Calculate battery consumption"""
        if self.is_charging:
            return -self.battery_charge_rate  # Negative = charging
        elif action in [0, 1, 2, 3]:  # Movement
            if self.carrying == PackageType.HEAVY:
                return self.battery_drain_heavy
            else:
                return self.battery_drain_move
        else:
            return self.battery_drain_idle
    
    def _update_humans(self):
        """Update human positions with realistic behavior"""
        shift_phase = self._get_shift_phase()
        
        for human in self.humans:
            rand = self.np_random.random()
            
            if shift_phase == 'lunch':
                if rand < 0.7:
                    human['target'] = self._get_zone_center('restricted_break')
                    human['state'] = 'walking'
            else:
                if rand < 0.6:  # Purposeful
                    zone = self.np_random.choice(['storage_a', 'packing_1', 'shipping_local'])
                    human['target'] = self._get_zone_center(zone)
                elif rand < 0.85:  # Continue
                    pass
                else:  # Random/chat
                    human['state'] = 'chatting'
                    human['idle_counter'] = 5
            
            # Move if walking
            if human['state'] == 'walking' and human['target'] is not None:
                if human['idle_counter'] > 0:
                    human['idle_counter'] -= 1
                else:
                    direction = human['target'] - human['pos']
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        human['pos'] = human['pos'] + np.round(direction).astype(int)
                        human['pos'] = np.clip(human['pos'], 0, self.grid_size - 1)
    
    def _update_forklifts(self):
        """Update forklift patrol"""
        for forklift in self.forklifts:
            target = forklift['patrol'][forklift['patrol_idx']]
            direction = np.array(target) - forklift['pos']
            
            if np.linalg.norm(direction) < 1:
                forklift['patrol_idx'] = (forklift['patrol_idx'] + 1) % len(forklift['patrol'])
            else:
                direction = direction / np.linalg.norm(direction)
                forklift['pos'] = forklift['pos'] + 2 * np.round(direction).astype(int)
                forklift['pos'] = np.clip(forklift['pos'], 0, self.grid_size - 1)
    
    def _update_other_robots(self):
        """Update other AMRs"""
        for robot in self.other_robots:
            direction = robot['target'] - robot['pos']
            if np.linalg.norm(direction) < 1:
                robot['target'] = self.np_random.integers(0, self.grid_size, size=2)
            else:
                direction = direction / np.linalg.norm(direction)
                robot['pos'] = robot['pos'] + np.round(direction).astype(int)
                robot['pos'] = np.clip(robot['pos'], 0, self.grid_size - 1)
    
    def _check_collisions(self) -> float:
        """Check for collisions and return penalty"""
        penalty = 0.0
        
        # Humans
        for human in self.humans:
            if np.array_equal(self.robot_pos, human['pos']):
                penalty -= 200  # CRITICAL
            elif np.linalg.norm(self.robot_pos - human['pos']) < 1.5:
                penalty -= 30
        
        # Forklifts
        for forklift in self.forklifts:
            if np.array_equal(self.robot_pos, forklift['pos']):
                penalty -= 150
            elif np.linalg.norm(self.robot_pos - forklift['pos']) < 3:
                penalty -= 5
        
        # Other robots
        for robot in self.other_robots:
            if np.array_equal(self.robot_pos, robot['pos']):
                penalty -= 40
        
        return penalty
    
    def _check_task_progress(self) -> float:
        """Check if task progressing"""
        reward = 0.0
        
        if self.task_stage == 'completed':
            reward += 150
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Generate multi-channel observation"""
        obs = np.zeros((self.grid_size, self.grid_size, 12), dtype=np.float32)
        
        # Channel 0: Robot position
        obs[self.robot_pos[0], self.robot_pos[1], 0] = 1.0
        
        # Channel 1: Battery level
        obs[:, :, 1] = self.battery / 100.0
        
        # Channel 2: Current task path
        if self.task_stage == 'pickup':
            zone = self.current_task['pickup']
        else:
            zone = self.current_task['delivery']
        self._mark_zone_in_obs(obs, zone, channel=2)
        
        # Channel 3: Humans
        for human in self.humans:
            x, y = human['pos']
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                obs[x, y, 3] = 1.0
        
        # Channel 4: Forklifts
        for forklift in self.forklifts:
            x, y = forklift['pos']
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                obs[x, y, 4] = 1.0
        
        # Channel 5: Other robots
        for robot in self.other_robots:
            x, y = robot['pos']
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                obs[x, y, 5] = 1.0
        
        # Channels 6-11: Additional features (zones, charging, etc.)
        # ... (implement as needed)
        
        return obs
    
    # Helper methods
    def _is_valid_pos(self, pos: np.ndarray) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
    
    def _in_restricted_zone(self) -> bool:
        for zone_name, (min_c, max_c) in self.zones.items():
            if 'restricted' in zone_name:
                if (min_c[0] <= self.robot_pos[0] <= max_c[0] and
                    min_c[1] <= self.robot_pos[1] <= max_c[1]):
                    return True
        return False
    
    def _at_pickup(self) -> bool:
        pickup_zone = self.current_task['pickup']
        return self._in_zone(pickup_zone)
    
    def _at_delivery(self) -> bool:
        delivery_zone = self.current_task['delivery']
        return self._in_zone(delivery_zone)
    
    def _at_charging_station(self) -> bool:
        return self._in_zone('charging_main') or self._in_zone('charging_emergency')
    
    def _in_zone(self, zone_name: str) -> bool:
        if zone_name not in self.zones:
            return False
        min_c, max_c = self.zones[zone_name]
        return (min_c[0] <= self.robot_pos[0] <= max_c[0] and
                min_c[1] <= self.robot_pos[1] <= max_c[1])
    
    def _get_shift_phase(self) -> str:
        if 200 <= self.steps <= 250:
            return 'lunch'
        elif self.steps < 200:
            return 'morning'
        else:
            return 'afternoon'
    
    def _random_pos_in_zone(self, zone_name: str) -> np.ndarray:
        min_c, max_c = self.zones[zone_name]
        x = self.np_random.integers(min_c[0], max_c[0] + 1)
        y = self.np_random.integers(min_c[1], max_c[1] + 1)
        return np.array([x, y])
    
    def _get_zone_center(self, zone_name: str) -> np.ndarray:
        min_c, max_c = self.zones[zone_name]
        return np.array([(min_c[0] + max_c[0]) // 2, (min_c[1] + max_c[1]) // 2])
    
    def _mark_zone_in_obs(self, obs: np.ndarray, zone_name: str, channel: int):
        min_c, max_c = self.zones[zone_name]
        obs[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, channel] = 1.0
    
    def _moving_toward_goal(self, direction: np.ndarray) -> bool:
        if self.task_stage == 'pickup':
            target = self._get_zone_center(self.current_task['pickup'])
        else:
            target = self._get_zone_center(self.current_task['delivery'])
        
        distance_before = np.linalg.norm(self.robot_pos - direction - target)
        distance_after = np.linalg.norm(self.robot_pos - target)
        return distance_after < distance_before
    
    def _should_wait(self) -> bool:
        # Wait if human or forklift blocking path
        for human in self.humans:
            if np.linalg.norm(self.robot_pos - human['pos']) < 2:
                return True
        for forklift in self.forklifts:
            if np.linalg.norm(self.robot_pos - forklift['pos']) < 3:
                return True
        return False
    
    def render(self):
        """Visualize environment"""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw zones
        for zone_name, (min_c, max_c) in self.zones.items():
            color = 'red' if 'restricted' in zone_name else 'green' if 'charging' in zone_name else 'gray'
            width = max_c[1] - min_c[1] + 1
            height = max_c[0] - min_c[0] + 1
            rect = patches.Rectangle(
                (min_c[1], min_c[0]), width, height,
                linewidth=1, edgecolor='black',
                facecolor=color, alpha=0.2
            )
            ax.add_patch(rect)
        
        # Draw humans
        for human in self.humans:
            circle = plt.Circle((human['pos'][1], human['pos'][0]), 0.3, color='blue')
            ax.add_patch(circle)
        
        # Draw forklifts
        for forklift in self.forklifts:
            rect = patches.Rectangle(
                (forklift['pos'][1] - 0.4, forklift['pos'][0] - 0.4),
                0.8, 0.8, color='red'
            )
            ax.add_patch(rect)
        
        # Draw robot
        robot_color = 'green' if self.battery > 50 else 'orange' if self.battery > 20 else 'red'
        circle = plt.Circle((self.robot_pos[1], self.robot_pos[0]), 0.4, color=robot_color)
        ax.add_patch(circle)
        
        plt.title(f"Warehouse AMR | Battery: {self.battery:.1f}% | Step: {self.steps}")
        plt.show()


# ============================================================================
# FILE 2: src/agents/q_learning_agent.py
# ============================================================================

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
        from collections import defaultdict
        self.q_table = defaultdict(lambda: {a: 10.0 for a in range(n_actions)})
    
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
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================================
# FILE 3: src/agents/dqn_agent.py
# ============================================================================

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
        
        x = x.view(x.size(0), -1)  # Flatten
        
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
    """DQN Agent"""
    
    def __init__(self, n_actions=11, learning_rate=1e-4, discount=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 buffer_size=50000, batch_size=64, target_update_freq=500):
        
        self.n_actions = n_actions
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQNetwork(n_actions).to(self.device)
        self.target_net = DQNetwork(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.steps = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """Store transition and train"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < 1000:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)
        
        # Compute target: r + gamma * max_a' Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Huber loss
        loss = F.smooth_l1_loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ============================================================================
# FILE 4: src/training/train_qlearning.py
# ============================================================================

def train_q_learning(env, agent, n_episodes=3000, save_path='models/q_learning_best.pkl'):
    """Train Tabular Q-Learning Agent"""
    import pickle
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print("Training Q-Learning Agent...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        state = agent.discretize_state(obs)
        
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not done and not truncated:
            action = agent.select_action(state, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if episode_reward > 100:  # Successful episode
            success_count += 1
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = success_count / 100
            print(f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.2%} | Epsilon: {agent.epsilon:.3f}")
            success_count = 0
    
    # Save model
    with open(save_path, 'wb') as f:
        pickle.dump(agent.q_table, f)
    
    return episode_rewards, episode_lengths


# ============================================================================
# FILE 5: src/training/train_dqn.py
# ============================================================================

def train_dqn(env, agent, n_episodes=5000, save_path='models/dqn_best.pth'):
    """Train DQN Agent"""
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print("Training DQN Agent...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not done and not truncated:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            
            agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if episode_reward > 100:
            success_count += 1
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = success_count / 100
            print(f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.2%} | Epsilon: {agent.epsilon:.3f}")
            success_count = 0
            
            # Save best model
            if success_rate > 0.7:
                agent.save(save_path)
    
    return episode_rewards, episode_lengths


# ============================================================================
# FILE 6: requirements.txt
# ============================================================================

"""
gymnasium>=0.29.0
numpy>=1.24.0
torch>=2.0.0
matplotlib>=3.7.0
pandas>=2.0.0
scipy>=1.10.0
pyyaml>=6.0
tensorboard>=2.13.0
"""

# ============================================================================
# FILE 7: src/main.py - Complete Training Script
# ============================================================================

def main():
    """Main training script"""
    
    # Create environment
    env = WarehouseAMREnv(config={'n_humans': 3, 'n_forklifts': 1, 'n_other_robots': 2})
    
    print("=" * 60)
    print("WAREHOUSE AMR RL PROJECT")
    print("=" * 60)
    
    # Train Q-Learning
    print("\n[1/2] Training Tabular Q-Learning...")
    q_agent = QLearningAgent()
    q_rewards, q_lengths = train_q_learning(env, q_agent, n_episodes=3000)
    
    # Train DQN
    print("\n[2/2] Training Deep Q-Network...")
    dqn_agent = DQNAgent()
    dqn_rewards, dqn_lengths = train_dqn(env, dqn_agent, n_episodes=5000)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Q-Learning
    ax1.plot(q_rewards, alpha=0.3)
    ax1.plot(pd.Series(q_rewards).rolling(100).mean(), label='Q-Learning')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Q-Learning Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot DQN
    ax2.plot(dqn_rewards, alpha=0.3)
    ax2.plot(pd.Series(dqn_rewards).rolling(100).mean(), label='DQN')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('DQN Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_comparison.png', dpi=300)
    print("\nTraining complete! Results saved to results/figures/")

if __name__ == '__main__':
    main()


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
STEP 1: Setup
-------------
1. Create directory structure as shown above
2. Create virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. Install dependencies:
   pip install gymnasium numpy torch matplotlib pandas scipy

STEP 2: Run Training
--------------------
python src/main.py

STEP 3: Evaluation
------------------
- Trained models saved in models/
- Metrics saved in results/metrics/
- Plots saved in results/figures/

STEP 4: Write Report
--------------------
- Use the report template provided
- Insert your experimental results in marked sections
- Generate all required plots
- Analyze and discuss findings

STEP 5: Submission
------------------
1. Compress everything:
   zip -r matricola-lastname.zip src/ report.pdf requirements.txt README.md

2. Submit via course form before deadline

3. Prepare for oral discussion (all team members must attend)
"""