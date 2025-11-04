# ML Project Report Template
## Autonomous Warehouse Robot with Battery Management and Multi-Stage Tasks

**Student Name:** [Your Name]  
**Matricola:** [Your Matricola]  
**Academic Year:** 2024-2025  
**Course:** Machine Learning

---

## Abstract (150 words max)

This project addresses the challenge of autonomous navigation for warehouse robots operating in dynamic environments with human workers, industrial vehicles, and complex operational constraints. We implement two reinforcement learning approaches—**Tabular Q-Learning** and **Deep Q-Networks (DQN)**—to train an Autonomous Mobile Robot (AMR) that must:
1. Navigate safely around freely moving humans
2. Manage battery charge autonomously
3. Execute multi-stage pickup and delivery tasks
4. Respect zone authorizations and traffic rules
5. Coordinate with other robots and yield to forklifts

The robot operates in a 20×20 grid warehouse with realistic constraints including battery depletion, multiple pickup/delivery stations, package types (standard/fragile/heavy/urgent), and restricted zones. Results demonstrate that [**INSERT SUMMARY: e.g., "DQN achieves 85% success rate with 99% safety compliance, outperforming tabular Q-learning by 23% in complex scenarios"**].

---

## 1. Introduction (2 pages)

### 1.1 Motivation

Modern warehouses increasingly deploy Autonomous Mobile Robots (AMRs) to automate material handling. However, these robots must operate safely alongside human workers while meeting operational efficiency targets. Key challenges include:

- **Dynamic Human Behavior**: Workers move freely, stop unpredictably, form groups
- **Operational Constraints**: Battery management, zone restrictions, priority handling
- **Safety Requirements**: Maintaining 1.5m safety distance, yielding to forklifts
- **Task Complexity**: Multi-stage workflows (receiving → storage → packing → shipping)

This project simulates a realistic warehouse environment to evaluate reinforcement learning approaches for safe and efficient autonomous navigation.

### 1.2 Problem Statement

**Objective**: Train an AMR agent to maximize task completion while ensuring zero safety violations.

**Core Challenges**:
1. Navigate 20×20 warehouse with moving obstacles
2. Maintain battery charge (0-100%, depletes with movement)
3. Execute complex tasks: pickup from zone A → deliver to zone B
4. Avoid collisions with humans (-200 penalty), forklifts (-150), other robots (-40)
5. Respect restricted zones (office, break room, quality control)
6. Handle 4 package types with different properties (standard, fragile, heavy, urgent)

**State Space Complexity**: ~500,000 discrete states (tabular) or 4,820-dimensional continuous (DQN)

**Episode Success Criteria**:
- Task completed: Package delivered to correct zone
- Safety maintained: Zero critical collisions
- Battery managed: Robot didn't run out of power (0% = mission failure)

### 1.3 Contributions

1. Custom Gymnasium environment modeling realistic warehouse operations
2. Comparative study of Tabular Q-Learning vs DQN on multi-constraint navigation
3. Analysis of safety-efficiency trade-offs under battery constraints
4. Evaluation of generalization to unseen human behavior patterns
5. Industrial applicability assessment for real-world deployment

---

## 2. Background & Related Work (2 pages)

### 2.1 Reinforcement Learning Fundamentals

**Markov Decision Process (MDP)**: Defined as tuple $(S, A, P, R, \gamma)$

- State space $S$: Robot position, battery, task status, obstacles
- Action space $A$: {N, S, E, W, Wait, Pickup, Drop, Charge, StopCharge, RequestRightOfWay, EmergencyStop}
- Transition function $P(s'|s,a)$: Stochastic (humans move unpredictably)
- Reward function $R(s,a,s')$: Safety-weighted (Section 3.3)
- Discount factor $\gamma = 0.95$: Long-term planning

**Q-Learning**: Off-policy TD control algorithm
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**Deep Q-Network**: Function approximation with neural networks
$$\mathcal{L}(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$$

### 2.2 Warehouse Robotics Literature

- **Wurman et al. (2008)**: Coordinating hundreds of warehouse robots
- **Fragapane et al. (2021)**: AMR planning and control in intralogistics
- **Villani & Sabattini (2018)**: Safety metrics for human-robot collaboration

### 2.3 Relevant Safety Standards

- ANSI/RIA R15.08-2020: Industrial mobile robot safety
- ISO 3691-4: Automated guided vehicles safety requirements
- OSHA guidelines: 1.5m minimum safety distance for collaborative robots

---

## 3. Environment Design (3 pages)

### 3.1 Warehouse Layout

**Grid**: 20×20 cells (each cell = 1m × 1m = real-world scale)

**Zones** (15 total):
- **Storage**: 3 zones (Electronics, Textiles, Tools)
- **Receiving**: 2 stations (Truck dock, Rail dock)
- **Packing**: 2 stations (Standard, Fragile handling)
- **Shipping**: 3 zones (Local, Express, International)
- **Charging**: 2 stations (Main bay, Emergency point)
- **Restricted**: 3 zones (Office, Break room, QC lab)

```
Visual Layout:
┌─────────────────────────────────────────────┐
│  CHG  │  Storage A  │  Storage B  │ Stor C │ 
│       │ (Electron.) │ (Textiles)  │ (Tools)│
├───────┼─────────────┼─────────────┼────────┤
│ Recv1 │             Corridor              │
├───────┼──────────────────────────────┼─────┤
│       │   RESTRICTED OFFICE AREA     │ Rec2│
│ Pack1 │         (No Entry)            │     │
├───────┼──────────────────────────────┼─────┤
│ Pack2 │      Main Corridor           │ CHG │
│       │   (Forklift Patrol Route)    │ Emer│
├───────┼──────────────────────────────┼─────┤
│Ship_L │Ship_Expr│ Break  │Ship_Intl │ QC  │
└─────────────────────────────────────────────┘
```

### 3.2 Dynamic Entities

#### 3.2.1 Autonomous Robot (Our Agent)
- **Speed**: 1 cell/step (1 m/s) or 0.5 cells/step when carrying heavy
- **Battery**: 100% max, drains 1%/move, 2%/heavy move, 0.5%/idle
- **Charging Rate**: +5%/step at charging station
- **Package Capacity**: 1 package at a time

#### 3.2.2 Human Workers (4-5 active)
**Behavior Model**:
```python
Movement Distribution:
- 60%: Purposeful (moving to workstation/package)
- 25%: Social (chatting, irregular stops)
- 15%: Random walk
```

**Shift Schedule**:
- Morning (steps 0-200): 4-5 workers in storage zones
- Lunch (steps 200-250): 2-3 workers, mostly in break room
- Afternoon (steps 250-500): 3-4 workers in shipping zones

**Unpredictability**: Sudden direction changes, group formations, prolonged stops

#### 3.2.3 Forklifts (1-2 active)
- **Speed**: 2 cells/step (faster than robot)
- **Patrol**: Predefined routes on main corridors
- **Right-of-Way**: Always has priority (robot must yield)
- **Danger Zone**: 3-cell radius marked as high risk

#### 3.2.4 Other AMRs (1-2 active)
- **Speed**: 1 cell/step
- **Behavior**: Scripted pickup-delivery routes
- **Coordination**: Can request/grant right-of-way

### 3.3 Reward Function Design

**Philosophy**: Safety-first with efficiency incentives

```python
Reward Components:

# SAFETY PENALTIES (Highest Priority)
collision_with_human:         -200  # CRITICAL
collision_with_forklift:      -150  # Equipment damage
entered_restricted_zone:       -50  # Authorization violation
within_safety_zone (< 1.5m):   -30  # Too close warning
collision_with_other_robot:    -40  # Coordination failure

# BATTERY MANAGEMENT
battery_depleted (0%):        -300  # Mission failure (terminates episode)
battery_critical (< 20%):      -20/step if not heading to charger
charged_appropriately:         +5   # Started charging when needed
charged_unnecessarily:        -10   # Charged at > 70% battery

# TASK COMPLETION
successful_pickup:            +60
successful_delivery:         +150  # Main objective
completed_urgent_on_time:    +100  # Bonus for deadline met
failed_urgent_deadline:       -80  # Missed priority task

# EFFICIENCY
moving_toward_goal:            +3   # Progress reward
unnecessary_wait:              -5   # Time wasting
time_step:                     -1   # Encourage speed

# OPERATIONAL COMPLIANCE
yielded_to_forklift:           +8   # Safety behavior
used_designated_paths:         +2   # Following rules
avoided_congestion:            +5   # Smart navigation
```

**Rationale**: Heavy penalties for safety violations ensure learned policy prioritizes zero collisions over speed. Battery management rewards encourage proactive charging behavior.

### 3.4 Task Specification

**Task Types** (9 common workflows):

1. **Inbound**: Receiving → Storage
2. **Outbound**: Storage → Packing → Shipping
3. **Transfer**: Storage A → Storage B
4. **Urgent**: Any route with 200-step deadline

**Package Properties**:
- **Standard**: Normal speed, any route
- **Fragile**: Avoid high-traffic corridors, careful handling
- **Heavy**: 50% speed reduction, 2× battery drain
- **Urgent**: Time bonus/penalty, higher priority

**Episode Structure**:
1. Robot spawns at (10, 10) with 100% battery
2. Random task assigned: {pickup_zone, delivery_zone, package_type}
3. Agent must navigate: Start → Pickup → Delivery (→ Charging if needed)
4. Success: Package delivered + battery > 0%
5. Failure: Collision, battery depleted, or timeout (600 steps)

### 3.5 State Space Definition

#### Tabular Q-Learning (Discretized)

**State Tuple** (13 features):
```python
s = (robot_x, robot_y,                  # Position: 20×20
     battery_bin,                        # [0-20%, 20-40%, 40-60%, 60-80%, 80-100%]: 5
     carrying_package,                   # [None, Standard, Fragile, Heavy, Urgent]: 5
     task_stage,                         # [Idle, Pickup, Delivery, ChargingNeeded]: 4
     pickup_zone_id,                     # 6 possible zones
     delivery_zone_id,                   # 6 possible zones
     task_urgency,                       # [None, Standard, Urgent]: 3
     human_adjacent,                     # [No, N, S, E, W, Multiple]: 6
     forklift_nearby,                    # [No, Approaching, Crossing]: 3
     other_robot_adjacent,               # [No, Yes]: 2
     current_zone_type)                  # [Corridor, Storage, Restricted, etc.]: 6

Total states = 400 × 5 × 5 × 4 × 6 × 6 × 3 × 6 × 3 × 2 × 6 
             ≈ 466,560 states
```

**Discretization Strategy**:
- Battery: 20% bins (coarse but sufficient for charging decisions)
- Nearby obstacles: Binary presence in adjacent cells (reduces dimensionality)
- Zone type: Categorical (affects allowed actions)

#### Deep Q-Network (Continuous)

**Multi-Channel Tensor**: 20×20×12
```python
Channel  0: Robot position (1 at location, 0 elsewhere)
Channel  1: Battery level gradient (0.0-1.0 normalized)
Channel  2: Current task path visualization
Channel  3: Human positions and movement vectors
Channel  4: Forklift positions with danger zones (3-cell radius)
Channel  5: Other AMRs with predicted paths
Channel  6: Package locations (color-coded by type)
Channel  7: Zone authorization map (1=allowed, 0=restricted)
Channel  8: Charging station availability
Channel  9: Pickup/delivery station status
Channel 10: Static obstacles + dynamic blockages
Channel 11: Time-of-day zones (shift schedule)
```

**Additional Features** (20 scalars appended):
```python
[battery_percentage,              # 0.0-1.0
 carrying_package_onehot,          # 5 values
 package_weight_factor,            # 0.5 or 1.0
 urgent_steps_remaining,           # 0-30 normalized
 dist_to_pickup,                   # Euclidean, normalized
 dist_to_delivery,                 # Euclidean, normalized
 dist_to_nearest_charger,          # Euclidean, normalized
 current_zone_authorized,          # Boolean
 forklift_collision_risk,          # 0.0-1.0
 human_density_nearby,             # Count/10
 shift_phase_onehot]               # 3 values (morning/lunch/afternoon)
```

**Final Input**: Flatten(20×20×12) + 20 = 4,820 features

### 3.6 Action Space

**Discrete(11)**:
```python
0: Move North      (△y = -1)
1: Move South      (△y = +1)
2: Move East       (△x = +1)
3: Move West       (△x = -1)
4: Wait            (Stay in place, useful when blocked)
5: Pickup Package  (Only valid at pickup zones)
6: Drop Package    (Only valid at delivery zones)
7: Start Charging  (Only valid at charging stations)
8: Stop Charging   (Leave charging station)
9: Request Right-of-Way (Signal to other robots for coordination)
10: Emergency Stop (Immediate halt for safety)
```

**Action Constraints**:
- Cannot pickup if battery < 30% and package is heavy
- Cannot move if battery = 0% (episode terminates)
- Cannot charge if not at charging station
- Wait consumes 0.5% battery (idle power draw)

### 3.7 Gymnasium Implementation

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WarehouseAMREnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None):
        super().__init__()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(20, 20, 12),
            dtype=np.float32
        )
        
        # Initialize environment parameters
        self.grid_size = 20
        self.max_steps = 600
        self.battery_max = 100
        # ... (full implementation in artifact)
    
    def reset(self, seed=None, options=None):
        # Initialize robot, humans, forklifts, tasks
        return observation, info
    
    def step(self, action):
        # Execute action, update entities, calculate reward
        return observation, reward, terminated, truncated, info
    
    def render(self):
        # Matplotlib visualization
        pass
```

**Key Implementation Details**:
- Uses `gymnasium.Env` base class
- Compatible with stable-baselines3 (if desired for comparison)
- Includes comprehensive logging for analysis
- Configurable difficulty levels (1-5)

---

## 4. Algorithmic Approaches (3 pages)

### 4.1 Tabular Q-Learning

#### 4.1.1 Algorithm Description

Classical temporal-difference learning with discrete state-action space.

**Update Rule**:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

Where:
- $\alpha$: Learning rate (0.1, decayed)
- $\gamma$: Discount factor (0.95)
- $r_{t+1}$: Immediate reward
- $s_t, a_t$: Current state-action
- $s_{t+1}$: Next state

#### 4.1.2 Q-Table Structure

**Initialization**: Sparse dictionary with optimistic values
```python
Q_table = defaultdict(lambda: {a: 10.0 for a in range(11)})
# Optimistic initialization encourages exploration
```

**Memory Requirement**: ~466K states × 11 actions × 8 bytes = ~41 MB

#### 4.1.3 Exploration Strategy

**Epsilon-Greedy**:
$$a_t = \begin{cases} 
\text{argmax}_a Q(s_t, a) & \text{with probability } 1-\epsilon \\
\text{random action} & \text{with probability } \epsilon
\end{cases}$$

**Decay Schedule**:
$$\epsilon(t) = \max(0.01, 1.0 - \frac{t}{1000})$$
- Start: ε = 1.0 (full exploration)
- End: ε = 0.01 (1% exploration)
- Decay: Linear over 1000 episodes

#### 4.1.4 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate α | 0.1 → 0.01 | Decays as Q-values stabilize |
| Discount γ | 0.95 | Long-term planning for multi-stage tasks |
| Exploration ε | 1.0 → 0.01 | Standard decay schedule |
| Episodes | 3000 | Convergence observed by episode 2000-2500 |

#### 4.1.5 Implementation Pseudocode

```python
def train_q_learning(env, episodes=3000):
    Q = defaultdict(lambda: {a: 10.0 for a in range(11)})
    
    for episode in range(episodes):
        state = env.reset()
        epsilon = max(0.01, 1.0 - episode/1000)
        alpha = 0.1 / (1 + episode/500)  # Decay learning rate
        
        for step in range(600):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = max(Q[state], key=Q[state].get)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            # Q-learning update
            best_next_action = max(Q[next_state], key=Q[next_state].get)
            td_target = reward + 0.95 * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state = next_state
            if done or truncated:
                break
    
    return Q
```

### 4.2 Deep Q-Network (DQN)

#### 4.2.1 Network Architecture

**Convolutional Neural Network** for spatial feature extraction:

```
Input: (20, 20, 12) tensor
    ↓
Conv2D: 32 filters, 3×3 kernel, ReLU
    → Output: (18, 18, 32)
    ↓
Conv2D: 64 filters, 3×3 kernel, ReLU
    → Output: (16, 16, 64)
    ↓
Conv2D: 64 filters, 3×3 kernel, ReLU
    → Output: (14, 14, 64)
    ↓
Flatten: 14×14×64 = 12,544 features
    ↓
Concatenate with 20 scalar features
    → Total: 12,564 features
    ↓
Dense: 512 units, ReLU, Dropout(0.2)
    ↓
Dense: 256 units, ReLU
    ↓
Dense: 128 units, ReLU
    ↓
Output: 11 units, Linear (Q-values for 11 actions)
```

**PyTorch Implementation**:
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(12564, 512)  # 12544 + 20
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 11)  # 11 actions
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, grid_input, scalar_features):
        # Process grid through conv layers
        x = F.relu(self.conv1(grid_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Concatenate with scalar features
        x = torch.cat([x, scalar_features], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)  # No activation (regression)
        
        return q_values
```

#### 4.2.2 Experience Replay

**Replay Buffer**: Store transitions for decorrelation

```python
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=64):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
```

**Benefits**:
- Breaks temporal correlation in training data
- Enables off-policy learning
- Improves sample efficiency

#### 4.2.3 Target Network

**Dual Network Strategy**:
- **Policy Network** ($\theta$): Updated every step
- **Target Network** ($\theta^-$): Updated every 500 steps

**Loss Function (Huber Loss)**:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \mathcal{L}_\delta(y - Q(s,a;\theta)) \right]$$

Where:
$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

$$\mathcal{L}_\delta(x) = \begin{cases}
\frac{1}{2}x^2 & \text{if } |x| \leq \delta \\
\delta(|x| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

(Huber loss = MSE for small errors, MAE for large → robust to outliers)

#### 4.2.4 Training Procedure

```python
def train_dqn(env, episodes=5000):
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=50000)
    
    epsilon = 1.0
    global_step = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(600):
            # Epsilon-greedy with decay
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    grid, scalars = preprocess_state(state)
                    q_values = policy_net(grid, scalars)
                    action = q_values.argmax().item()
            
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if buffer has enough samples
            if len(replay_buffer) > 1000:
                batch = replay_buffer.sample(64)
                loss = compute_loss(batch, policy_net, target_net)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            
            # Update target network
            if global_step % 500 == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            state = next_state
            episode_reward += reward
            global_step += 1
            
            if done or truncated:
                break
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.9995)
    
    return policy_net
```

#### 4.2.5 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 1e-4 | Standard for Adam with conv nets |
| Batch size | 64 | Balance memory/computation |
| Replay buffer | 50,000 | 80+ full episodes stored |
| Target update freq | 500 steps | Stable learning (not too frequent) |
| Epsilon decay | 0.9995 per episode | Slower decay for DQN (needs more exploration) |
| Discount γ | 0.95 | Same as Q-learning |
| Huber delta | 1.0 | Standard value |
| Gradient clipping | 1.0 | Prevents exploding gradients |

---

## 5. Experimental Setup (2 pages)

### 5.1 Research Questions

**RQ1**: How do Tabular Q-Learning and DQN compare in task success rate and safety compliance?

**RQ2**: Can the agent learn effective battery management without explicit hand-coded heuristics?

**RQ3**: How does performance scale with environment complexity (number of humans, obstacles)?

**RQ4**: Does DQN generalize to unseen human behavior patterns not encountered during training?

**RQ5**: What is the trade-off between task completion speed and safety compliance?

### 5.2 Evaluation Metrics

#### Safety Metrics (Primary)
```python
collision_rate = (episodes_with_collision / total_episodes) × 100%
human_safety_violations = (safety_zone_entries / total_steps) × 100%
critical_incidents = count(collisions with humans or forklifts)
safety_score = 100 - (collision_rate + 10×critical_incident_rate)
```

#### Task Performance
```python
success_rate = (completed_deliveries / total_episodes) × 100%
avg_steps_to_completion = mean(steps per successful episode)
throughput = packages_delivered / 1000_steps
battery_efficiency = tasks_completed_per_100%_battery
```

#### Learning Efficiency
```python
episodes_to_convergence = first_episode(success_rate > 80%)
sample_efficiency = cumulative_reward / training_samples
training_time = wall_clock_hours
```

#### Operational Compliance
```python
charging_efficiency = (charges_when_needed / charges_total) × 100%
zone_violation_rate = (restricted_entries / total_steps) × 100%
forklift_yield_rate = (times_yielded / forklift_encounters) × 100%
```

### 5.3 Experimental Scenarios

#### Experiment 1: Baseline Performance (Levels 1-5)

**Configuration**:
| Level | Humans | Forklifts | Other Robots | Description |
|-------|--------|-----------|--------------|-------------|
| 1 | 1 | 0 | 1 | Basic navigation |
| 2 | 2 | 1 | 1 | Forklift avoidance |
| 3 | 3 | 1 | 2 | Multi-agent coordination |
| 4 | 4 | 2 | 2 | Full complexity |
| 5 | 5 | 2 | 3 | Stress test |

**Procedure**:
1. Train Q-Learning for 3000 episodes on each level
2. Train DQN for 5000 episodes on each level
3. Evaluate trained agents for 100 episodes (different seeds)
4. Record all metrics

**Expected Output**: Learning curves, performance tables, comparison plots

---

#### Experiment 2: Battery Management Analysis

**Variants**:
- **A**: Default rewards (includes battery penalties)
- **B**: No battery rewards (agent must discover charging necessity)
- **C**: Heavy battery penalties (10× multiplier)

**Hypothesis**: Agents with battery rewards will learn proactive charging behavior.

**Metrics to Compare**:
- Battery depletion rate (episodes failing due to 0% battery)
- Average battery level during task completion
- Charging efficiency (charges when < 30% vs > 70%)

[**INSERT RESULTS SECTION 6.2**]

---

#### Experiment 3: Scalability Study

**Procedure**:
1. Train DQN on Level 2 (moderate complexity)
2. Evaluate on Levels 1-5 without retraining
3. Measure performance degradation

**Key Question**: At what complexity does learned policy break down?

[**INSERT RESULTS SECTION 6.3**]

---

#### Experiment 4: Human Behavior Generalization

**Training Setup**: Level 3 with scripted human patterns (60% purposeful, 25% social, 15% random)

**Testing Variants**:
- **Test A**: 80% purposeful (more predictable)
- **Test B**: 50% random (more chaotic)
- **Test C**: Group formations (3+ humans moving together)
- **Test D**: Rush hour (7 humans active simultaneously)

**Hypothesis**: DQN should maintain > 70% performance on unseen patterns.

[**INSERT RESULTS SECTION 6.4**]

---

#### Experiment 5: Safety-Speed Trade-off Analysis

**Reward Multipliers**:
- Variant 1: Safety penalties × 0.5 (lenient)
- Variant 2: Safety penalties × 1.0 (default)
- Variant 3: Safety penalties × 2.0 (strict)
- Variant 4: Safety penalties × 5.0 (ultra-conservative)

**Measured Trade-offs**:
- Success rate vs collision rate
- Average completion time vs safety score
- Efficiency vs compliance

[**INSERT RESULTS SECTION 6.5**]

---

### 5.4 Statistical Methodology

**Repeated Trials**: 10 independent runs per configuration (different random seeds)

**Significance Testing**:
- Wilcoxon signed-rank test (non-parametric, pairwise comparisons)
- Kruskal-Wallis test (multi-group comparisons)
- Bonferroni correction for multiple comparisons

**Confidence Intervals**: Report 95% CI for all metrics

**Reproducibility**: Seeds logged, code versioned, hyperparameters documented

---

## 6. Results & Analysis (4-5 pages)

### 6.1 Baseline Performance Comparison

[**INSERT TABLES AND FIGURES HERE**]

#### Table 1: Performance Summary Across Difficulty Levels

| Method | Level | Success Rate (%) | Collision Rate (%) | Avg Steps | Training Episodes |
|--------|-------|------------------|--------------------|-----------|--------------------|
| Q-Learning | 1 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 3000 |
| Q-Learning | 2 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 3000 |
| Q-Learning | 3 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 3000 |
| Q-Learning | 4 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 3000 |
| Q-Learning | 5 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 3000 |
| DQN | 1 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 5000 |
| DQN | 2 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 5000 |
| DQN | 3 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 5000 |
| DQN | 4 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 5000 |
| DQN | 5 | [XX.X ± Y.Y] | [X.X ± Y.Y] | [XXX ± YY] | 5000 |

**Statistical Significance**: [Report p-values from Wilcoxon test]

---

#### Figure 1: Learning Curves (Cumulative Reward vs Episodes)

[**INSERT PLOT**: Two subplots showing Q-Learning and DQN learning curves for each level. X-axis: Episodes, Y-axis: Average cumulative reward (smoothed with moving average). Include error bands (std dev).]

**Observations**:
- [Describe convergence behavior]
- [Compare learning speed between methods]
- [Note any instabilities or plateaus]

---

#### Figure 2: Success Rate Evolution During Training

[**INSERT PLOT**: Success rate over episodes for both methods across all levels. Show how quickly each method achieves 80% success threshold.]

---

#### Figure 3: Collision Heatmap

[**INSERT HEATMAP**: 20×20 grid showing collision frequency at each location. Use color intensity to indicate danger zones. Generate separate heatmaps for Q-Learning vs DQN.]

**Insights**:
- [Identify high-risk zones]
- [Compare collision patterns between methods]

---

### 6.2 Battery Management Analysis

[**INSERT RESULTS FOR EXPERIMENT 2**]

#### Table 2: Battery Management Performance

| Variant | Battery Depletion Failures (%) | Avg Battery at Completion (%) | Charging Efficiency (%) |
|---------|--------------------------------|-------------------------------|-------------------------|
| A (Default) | [X.X ± Y.Y] | [XX.X ± Y.Y] | [XX.X ± Y.Y] |
| B (No Battery Reward) | [X.X ± Y.Y] | [XX.X ± Y.Y] | [XX.X ± Y.Y] |
| C (Heavy Penalty) | [X.X ± Y.Y] | [XX.X ± Y.Y] | [XX.X ± Y.Y] |

#### Figure 4: Battery Level Distribution

[**INSERT PLOT**: Histogram of battery levels when agent initiates charging. Compare variants A, B, C.]

**Analysis**:
- [Did agent learn proactive charging in variant A?]
- [What happened in variant B without explicit rewards?]
- [Was variant C too conservative?]

---

### 6.3 Scalability Analysis

[**INSERT RESULTS FOR EXPERIMENT 3**]

#### Figure 5: Performance Degradation Across Complexity Levels

[**INSERT PLOT**: Line graph showing success rate (Y-axis) vs difficulty level (X-axis) for Q-Learning and DQN. Include error bars.]

**Key Findings**:
- Q-Learning breaking point: Level [X]
- DQN breaking point: Level [X]
- Performance gap increases with complexity: [Describe trend]

---

### 6.4 Generalization to Unseen Human Behaviors

[**INSERT RESULTS FOR EXPERIMENT 4**]

#### Table 3: Transfer Performance

| Test Scenario | Training Success (%) | Test Success (%) | Performance Drop (%) |
|---------------|----------------------|------------------|----------------------|
| Test A (Predictable) | [XX.X] | [XX.X ± Y.Y] | [±X.X] |
| Test B (Chaotic) | [XX.X] | [XX.X ± Y.Y] | [±X.X] |
| Test C (Groups) | [XX.X] | [XX.X ± Y.Y] | [±X.X] |
| Test D (Rush Hour) | [XX.X] | [XX.X ± Y.Y] | [±X.X] |

**Interpretation**:
- [How robust is the policy?]
- [Which scenario was most challenging?]
- [Evidence of overfitting or good generalization?]

---

### 6.5 Safety-Speed Trade-off

[**INSERT RESULTS FOR EXPERIMENT 5**]

#### Figure 6: Pareto Frontier (Safety vs Speed)

[**INSERT PLOT**: Scatter plot with Success Rate (X-axis) vs Safety Score (Y-axis). Each point represents a reward multiplier variant. Show Pareto optimal points.]

**Findings**:
- Optimal balance: Safety penalty multiplier = [X.X]
- [Describe trade-off curve]

---

### 6.6 Qualitative Analysis

#### Learned Behaviors (DQN)

[**DESCRIBE OBSERVED BEHAVIORS**:]
1. **Proactive Human Avoidance**: Agent learned to predict human trajectories and preemptively alter course
2. **Forklift Yielding**: Consistently stopped and waited when forklift approached
3. **Battery-Aware Planning**: Detoured to charging station when battery < 25%
4. **Corridor Preference**: Heavy packages routed through low-traffic corridors
5. **Deadlock Resolution**: Coordinated with other robots using right-of-way requests

#### Failure Modes

[**DOCUMENT FAILURE CASES**:]
1. **Overcautious Behavior**: Sometimes waited excessively near humans (efficiency loss)
2. **Rare Corner Cases**: Struggled with simultaneous forklift + human encounters
3. **Battery Miscalculation**: Occasionally initiated long tasks with insufficient charge

#### Trajectory Visualizations

[**INSERT FIGURES**: Sample trajectories showing successful and failed episodes. Use arrows for robot path, markers for human positions over time.]

---

## 7. Discussion (2 pages)

### 7.1 Summary of Key Findings

1. **DQN Outperforms Tabular Q-Learning**: DQN achieved [XX%] higher success rate on Level 4+, demonstrating superior handling of high-dimensional state spaces.

2. **Safety-Critical Behavior Emerges**: Both methods learned to prioritize safety (collision rate < [X%]), validating the reward structure design.

3. **Battery Management is Learnable**: Agents successfully learned proactive charging without hand-coded rules (depletion failures < [X%]).

4. **Generalization Varies by Complexity**: DQN maintained [XX%] performance on unseen human behaviors, while Q-Learning dropped to [XX%].

5. **Trade-off is Real**: Increasing safety penalties by 2× reduced collisions by [X%] but increased completion time by [Y%].

### 7.2 Comparison of Approaches

#### Tabular Q-Learning

**Strengths**:
- Simple to implement and debug
- Guaranteed convergence (under assumptions)
- Fast training on low-complexity levels (< 1 hour)
- Interpretable policy (can inspect Q-table)

**Limitations**:
- State space explosion beyond Level 3
- Poor generalization to unseen states
- Cannot handle continuous observations (e.g., precise distances)
- Memory requirements grow exponentially

**Best Use Case**: Simplified warehouse environments with < 3 dynamic obstacles

#### Deep Q-Network

**Strengths**:
- Scales to high-dimensional state spaces
- Learns spatial relationships via convolutions
- Generalizes to unseen scenarios ([XX%] transfer success)
- Can handle continuous state features

**Limitations**:
- Longer training time (5-10× slower than Q-Learning)
- Requires careful hyperparameter tuning
- Risk of catastrophic forgetting without replay buffer
- Less interpretable (black-box neural network)

**Best Use Case**: Realistic warehouse environments with 5+ dynamic entities

### 7.3 Practical Implications for Real-World Deployment

#### Industrial Applicability

**Feasibility**: The trained DQN policy could serve as a baseline for real-world AMR systems with:
- LiDAR sensors (replacing grid observation)
- RTLS tracking (for human/forklift positions)
- Battery management system integration

**Remaining Gaps**:
- Sim-to-real transfer requires domain randomization
- Real sensors have noise not modeled in simulation
- Legal/regulatory certification for safety-critical systems

#### Safety Considerations

**Learned Policy Limitations**:
- No guarantee of 100% safety (achieved [XX%])
- Rare failure modes still observed
- Requires redundant safety systems (emergency stops, physical barriers)

**Recommendations**:
1. Deploy with conservative speed limits initially
2. Use RL policy as "suggestion" with human override
3. Implement safety monitors checking learned decisions
4. Gradual rollout with extensive real-world testing

### 7.4 Limitations of Current Work

1. **Simplified Physics**: Grid-based movement doesn't capture continuous dynamics (acceleration, momentum, turning radius)
2. **Perfect Sensing**: Real robots face sensor noise, occlusions, latency
3. **Single Robot Focus**: Real warehouses coordinate fleets of 20+ robots
4. **Static Environment**: Layout changes (new obstacles, reorganization) not addressed
5. **Idealized Communication**: Assumed perfect coordination signals between robots

### 7.5 Future Work Directions

1. **Advanced RL Algorithms**:
   - Double DQN (reduce overestimation bias)
   - Dueling DQN (separate value/advantage functions)
   - Prioritized Experience Replay (focus on critical scenarios)
   - Rainbow DQN (combine multiple improvements)

2. **Multi-Agent Learning**:
   - Train all robots simultaneously (MARL)
   - Decentralized policies with communication
   - Emergent coordination protocols

3. **Hierarchical Control**:
   - High-level task planner (which package next?)
   - Low-level motion controller (path execution)
   - Temporal abstractions (options framework)

4. **Sim-to-Real Transfer**:
   - Domain randomization (vary lighting, sensor noise)
   - System identification (learn real-world dynamics)
   - Transfer learning from simulation to real robot

5. **Human Behavior Modeling**:
   - LSTM/Transformer for trajectory prediction
   - Inverse RL to learn from human operators
   - Adaptive policies that learn online

---

## 8. Conclusion (1 page)

This project successfully developed and evaluated two reinforcement learning approaches—Tabular Q-Learning and Deep Q-Networks—for autonomous warehouse robot navigation under realistic operational constraints including battery management, multi-stage tasks, zone authorizations, and complex human-robot interaction.

### Contributions

1. **Realistic Environment**: Implemented a comprehensive Gymnasium environment (`WarehouseAMREnv`) modeling battery dynamics, 15 zones, 4 package types, and unpredictable human movement patterns.

2. **Comparative Analysis**: Demonstrated that DQN achieves [XX%] higher success rate than Q-Learning on complex scenarios (Level 4+), with [X.X%] collision rate maintained across all difficulty levels.

3. **Emergent Safety Behaviors**: Validated that safety-critical behaviors (human avoidance, forklift yielding, proactive battery charging) can be learned through reward shaping without explicit hand-coded rules.

4. **Generalization Study**: Showed DQN maintains [XX%] performance when tested on unseen human behavior patterns, indicating robust policy learning.

5. **Trade-off Quantification**: Identified optimal safety penalty multiplier ([X.X]) that balances task efficiency and collision prevention.

### Lessons Learned

- **State Representation Matters**: Convolutional channels capturing spatial relationships (danger zones, traffic patterns) significantly improved DQN performance over flattened representations.

- **Reward Engineering is Critical**: Heavy penalties for safety violations (200× task reward) were necessary to prioritize collision avoidance during exploration.

- **Curriculum Learning Helps**: Starting training on simpler levels (fewer obstacles) then gradually increasing complexity accelerated convergence by [XX%].

- **Sample Efficiency Gap**: DQN required [X]× more training episodes than Q-Learning to reach comparable performance on simple tasks, but this gap reversed on complex scenarios.

### Final Remarks

While significant challenges remain for real-world deployment (sensor noise, continuous dynamics, legal certification), this work demonstrates that modern RL techniques can learn safe and efficient navigation policies for collaborative warehouse robots. The key insight is that **safety-first reward design combined with rich state representations enables autonomous systems to operate alongside humans without compromising on efficiency**.

Future work should focus on multi-agent coordination, sim-to-real transfer, and integration with actual warehouse management systems to move from simulation to production-ready AMR fleets.

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

3. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI Conference on Artificial Intelligence*.

4. Wurman, P. R., et al. (2008). Coordinating hundreds of cooperative, autonomous vehicles in warehouses. *AI Magazine*, 29(1), 9-20.

5. Fragapane, G., et al. (2021). Planning and control of autonomous mobile robots for intralogistics. *European Journal of Operational Research*, 294(2), 405-426.

6. Villani, V., & Sabattini, L. (2018). Safety in human-robot collaborative manufacturing environments: Metrics and control. *IEEE Transactions on Automation Science and Engineering*, 15(4), 1882-1894.

7. ANSI/RIA R15.08-2020. Industrial Mobile Robots—Safety Requirements.

8. ISO 3691-4:2020. Industrial trucks—Safety requirements—Part 4: Driverless industrial trucks and their systems.

9. Brockman, G., et al. (2016). OpenAI Gym. *arXiv preprint arXiv:1606.01540*.

10. Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.

[Continue with 20+ more references covering RL theory, warehouse robotics, safety standards, human-robot interaction, etc.]

---

**End of Report Template**

**Instructions for Completing Results Sections:**
1. Run all experiments as described in Section 5
2. Generate tables and figures as specified in Section 6
3. Replace `[INSERT RESULTS]` placeholders with actual data
4. Replace `[XX.X ± Y.Y]` with real numbers and confidence intervals
5. Add interpretations and insights in observation paragraphs
6. Ensure all claims are supported by experimental evidence

**Tip**: Use `matplotlib.pyplot` for professional-quality plots with:
- Clear axis labels and units
- Legends identifying each method/variant
- Error bars or confidence bands (95% CI)
- Consistent color scheme (blue for Q-Learning, orange for DQN)
- Grid lines for readability
- Figure captions explaining what's shown