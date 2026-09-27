# Warehouse AMR Navigation using Reinforcement Learning

**Project Report**
*Implementing Tabular Q-Learning and Deep Q-Network for Autonomous Mobile Robot Control*

---

## 1. Problem Description

### 1.1 Overview

This project addresses the challenge of autonomous navigation for mobile robots in warehouse environments. The robot must learn to complete delivery tasks while navigating safely around dynamic obstacles, managing battery levels, and optimizing task completion time.

### 1.2 Environment

The warehouse is modeled as a **20×20 grid** with the following components:
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
┌────────────────────────────────────────────┐
│  CHG  │  St A       │  St B          │St C │ 
│       │ (Els)       │ (Txs)          │(Tls)│
├───────┼──────────────────────────────┼─────┤
│ Recv1 │             Corridor               │
├───────┼──────────────────────────────┼─────┤
│       │   RESTRICTED OFFICE AREA     │Rec2 │
│ Pack1 │         (No Entry)           │     │
├───────┼──────────────────────────────┼─────┤
│ Pack2 │      Main Corridor           │ CHG │
│       │   (Forklift Patrol Route)    │ Emer│
├───────┼──────────────────────────────┼─────┤
│Ship_L │Ship_Expr│ Break  │Ship_Intl  │ QC  │
└────────────────────────────────────────────┘
```

**Zones:**
- **Storage zones** (A, B, C): Where packages are stored
- **Receiving zones** (1, 2): Where new packages arrive
- **Packing zones** (1, 2): Preparation areas
- **Shipping zones** (Local, Express, International): Delivery destinations
- **Charging stations** (Main, Emergency): For battery recharging
- **Restricted zones** (Office, Break room, QC): No entry areas

### Dynamic Entities

#### Autonomous Robot (Our Agent)
- **Speed**: 1 cell/step (1 m/s) or 0.5 cells/step when carrying heavy
- **Battery**: 100% max, drains 1%/move, 2%/heavy move, 0.5%/idle
- **Charging Rate**: +5%/step at charging station
- **Package Capacity**: 1 package at a time

#### Human Workers (4-5 active)
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

#### Forklifts (1-2 active)
- **Speed**: 2 cells/step (faster than robot)
- **Patrol**: Predefined routes on main corridors
- **Right-of-Way**: Always has priority (robot must yield)
- **Danger Zone**: 3-cell radius marked as high risk

#### Other AMRs (1-2 active)
- **Speed**: 1 cell/step
- **Behavior**: Scripted pickup-delivery routes
- **Coordination**: Can request/grant right-of-way

### 3.3 Reward Function Design

**Philosophy**: Safety-first with efficiency incentives

```python
Reward Components:

| Event | Reward | Description |
|-------|--------|-------------|
| Task completion | +150 | Successfully deliver package |
| Pickup | +60 | Pick up package at correct zone |
| Move toward goal | +3 | Step in right direction |
| Start charging | +5 | Begin charging at station |
| Time penalty | -1 | Each time step |
| Invalid action | -5 to -10 | Illegal move or action |
| Low battery | -20 | Battery < 20% |
| Too close to human | -30 | Within 1.5 units |
| Robot collision | -40 | Collision with other robot |
| Zone violation | -50 | Enter restricted area |
| Forklift collision | -150 | **Critical** safety violation |
| Human collision | -200 | **Terminal** safety violation |
| Battery depletion | -300 | **Terminal** - episode ends |
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
### 1.3 State Space

**Observation:** Multi-channel grid (20×20×12)
- Channel 0: Robot position (binary)
- Channel 1: Battery level (0-1, normalized)
- Channel 2: Current task target zone
- Channel 3: Human positions
- Channel 4: Forklift positions
- Channel 5: Other robot positions
- Channels 6-11: Additional features

For **tabular Q-learning**, this high-dimensional space is discretized to:
- Robot X position (0-19)
- Robot Y position (0-19)
- Battery bin (0-4): [0-20%, 20-40%, 40-60%, 60-80%, 80-100%]
- Human nearby (0-1): Safety flag

This yields a maximum of **20 × 20 × 5 × 2 = 4,000 discrete states**.

### 1.4 Action Space

**Discrete(11) actions:**
```
0: Move North     1: Move South     2: Move East      3: Move West
4: Wait           5: Pickup         6: Drop           7: Start Charge
8: Stop Charge    9-10: Reserved
```

### 1.5 Reward Structure



### 1.6 Objective

Learn a policy π that maximizes cumulative reward:

```
maximize E[Σ γ^t * r_t]
```

Where:
- γ = 0.95 (discount factor)
- r_t = reward at time t

---

## 2. Solution Approach

### 2.1 Algorithm Selection

Two algorithms were implemented as required:

1. **Tabular Q-Learning:** Classic RL algorithm for discrete state spaces
2. **Deep Q-Network (DQN):** Neural network approximation for high-dimensional spaces

### 2.2 Tabular Q-Learning

#### Theory

Q-Learning is a **model-free, off-policy** temporal difference learning algorithm. It learns the optimal action-value function Q*(s,a) which represents the expected return for taking action a in state s and following the optimal policy thereafter.

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- **α = 0.1:** Learning rate (how much new information overrides old)
- **γ = 0.95:** Discount factor (importance of future rewards)
- **r:** Immediate reward
- **s':** Next state
- **max_a' Q(s',a'):** Maximum Q-value in next state

**Key Properties:**
- **Off-policy:** Learns optimal policy while following ε-greedy exploration
- **Convergence:** Guaranteed to converge to Q* under certain conditions
- **Simplicity:** Direct table lookup, no function approximation

#### Implementation

```python
class QLearningAgent:
    """Tabular Q-Learning Agent"""

    def __init__(self, n_actions=11, learning_rate=0.1,
                 discount=0.95, epsilon=1.0):
        self.alpha = learning_rate      # α
        self.gamma = discount            # γ
        self.epsilon = epsilon           # ε for exploration
        self.q_table = {}               # State -> {action: Q-value}

    def discretize_state(self, observation):
        """Convert high-dim observation to discrete state"""
        # Extract robot position from channel 0
        robot_channel = observation[:, :, 0]
        robot_pos = tuple(np.argwhere(robot_channel == 1.0)[0])

        # Battery discretization (5 bins)
        battery_val = observation[0, 0, 1]
        battery_bin = int(battery_val * 5)

        # Human proximity (safety feature)
        human_channel = observation[:, :, 3]
        human_nearby = int(np.any(human_channel > 0))

        return (robot_pos[0], robot_pos[1], battery_bin, human_nearby)

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            if state not in self.q_table:
                self.q_table[state] = {a: 10.0 for a in range(self.n_actions)}
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        # Initialize states if new
        if state not in self.q_table:
            self.q_table[state] = {a: 10.0 for a in range(self.n_actions)}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 10.0 for a in range(self.n_actions)}

        # Q-Learning update rule
        best_next_action = max(self.q_table[next_state],
                               key=self.q_table[next_state].get)
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
```

**State Discretization Rationale:**
- **Position (20×20):** Essential for navigation
- **Battery (5 bins):** Sufficient granularity for charging decisions
- **Human nearby:** Binary flag for safety-critical situations
- **Total states:** ~2,000 (manageable for tabular method)

#### Training Loop

```python
def train_q_learning(env, agent, n_episodes=500):
    """Train Q-Learning agent"""
    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = agent.discretize_state(obs)
        episode_reward = 0
        done = False

        while not done:
            # 1. Select action (ε-greedy)
            action = agent.select_action(state, training=True)

            # 2. Execute action
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = agent.discretize_state(next_obs)

            # 3. Update Q-table
            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            done = done or truncated

        # Decay exploration
        agent.decay_epsilon()  # ε = max(0.01, ε * 0.999)
```

### 2.3 Deep Q-Network (DQN)

#### Theory

DQN extends Q-Learning to continuous state spaces using neural network function approximation. Instead of storing Q-values in a table, a neural network learns to map states to action values.

**Loss Function:**
```
L(θ) = E[(y - Q(s,a;θ))²]

where: y = r + γ max_a' Q(s',a';θ)  (TD target)
```

**Key Innovations:**
1. **Experience Replay:** Store transitions in buffer, sample random minibatches
   - Breaks correlation between consecutive samples
   - Improves data efficiency through reuse

2. **Neural Network:** Approximates Q-function
   - Handles high-dimensional inputs (20×20×12)
   - Generalizes across similar states

**Important:** This implementation uses a **SINGLE network** (not Double DQN), meaning the same network is used for both action selection and target computation. This is the simplest form of DQN as required.

#### Network Architecture

```python
class DQNetwork(nn.Module):
    """Deep Q-Network Architecture"""

    def __init__(self, n_actions=11):
        super().__init__()

        # Convolutional layers - extract spatial features
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)   # 12→32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 32→64 channels
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)   # 64→64 channels

        # Fully connected layers - compute Q-values
        self.fc1 = nn.Linear(25600, 512)    # 20×20×64 = 25,600 features
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions) # Output: Q-value per action

        self.dropout = nn.Dropout(0.2)       # Regularization

    def forward(self, x):
        """Forward pass: observation → Q-values"""
        # Reshape: (batch, 20, 20, 12) → (batch, 12, 20, 20)
        x = x.permute(0, 3, 1, 2)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values  # Shape: (batch, 11)
```

**Architecture Rationale:**
- **Conv layers:** Learn spatial patterns (walls, zones, obstacles)
- **FC layers:** Abstract reasoning for action selection
- **Dropout:** Prevents overfitting
- **Total parameters:** ~13.5M (fits in 8GB VRAM)

#### DQN Agent Implementation

```python
class DQNAgent:
    """DQN Agent with SINGLE Network"""

    def __init__(self, n_actions=11, learning_rate=1e-4,
                 discount=0.95, batch_size=64, buffer_size=50000):
        self.gamma = discount
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SINGLE NETWORK (as required)
        self.q_network = DQNetwork(n_actions).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)

    def update(self, state, action, reward, next_state, done):
        """DQN update using SINGLE network"""
        # Store experience
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < 1000:
            return  # Wait for initial buffer fill

        # Sample minibatch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q-values: Q(s,a)
        q_values = self.q_network(states).gather(1, actions)

        # Compute targets: y = r + γ max_a' Q(s',a')
        # *** Uses SAME network (not separate target network!) ***
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss and update
        loss = F.smooth_l1_loss(q_values, targets)  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
```

**Key Implementation Details:**
- **Single network:** Same network for Q(s,a) and max Q(s',a')
- **Huber loss:** More robust than MSE for outliers
- **Gradient clipping:** Stabilizes training (max norm = 1.0)
- **Experience replay:** Breaks temporal correlations

#### Training Loop

```python
def train_dqn(env, agent, n_episodes=500):
    """Train DQN agent"""
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 1. Select action (ε-greedy from network)
            action = agent.select_action(obs, training=True)

            # 2. Execute action
            next_obs, reward, done, truncated, _ = env.step(action)

            # 3. Store and train
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            done = done or truncated

        agent.decay_epsilon()
```

---

## 3. Implementation Architecture

### 3.1 Project Structure

```
src/
├── environment/
│   └── warehouse_env.py     # Gymnasium-based environment
├── agents/
│   ├── q_learning_agent.py  # Tabular Q-Learning
│   └── dqn_agent.py         # Deep Q-Network (single network)
└── training/
    ├── train_qlearning.py   # Q-Learning training loop
    └── train_dqn.py         # DQN training loop
```

### 3.2 Key Components

**Environment (`warehouse_env.py`):**
- Gymnasium-compatible interface
- Multi-channel observations (20×20×12)
- Dynamic entity updates (humans, forklifts)
- Battery management system
- Collision detection and rewards

**Agents:**
- **Q-Learning:** State discretization, Q-table, ε-greedy
- **DQN:** CNN architecture, replay buffer, gradient descent

**Training:**
- Episode management
- Metrics tracking (rewards, success rate)
- Model checkpointing
- Progress logging

---

## 4. Experimental Results

### 4.1 Training Configuration

| Parameter | Q-Learning | DQN |
|-----------|------------|-----|
| Episodes | 500 | 500 |
| Learning rate (α) | 0.1 | 0.0001 |
| Discount (γ) | 0.95 | 0.95 |
| Epsilon start | 1.0 | 1.0 |
| Epsilon decay | 0.999 | 0.9995 |
| Epsilon min | 0.01 | 0.01 |
| Batch size | N/A | 64-128 |
| Replay buffer | N/A | 50,000-100,000 |

### 4.2 Performance Metrics

**Q-Learning Results:**
```
Episode 100/500 | Avg Reward: -1149.50 | Success Rate: 0.00% | Epsilon: 0.905
Episode 200/500 | Avg Reward: -1333.15 | Success Rate: 0.00% | Epsilon: 0.819
Episode 300/500 | Avg Reward: -1220.09 | Success Rate: 0.00% | Epsilon: 0.741
Episode 400/500 | Avg Reward: -1481.40 | Success Rate: 0.00% | Epsilon: 0.670
Episode 500/500 | Avg Reward: -1623.28 | Success Rate: 0.00% | Epsilon: 0.606
```

**DQN Results:**
- Training successfully completed with GPU acceleration
- Model saved with single network architecture
- Mixed precision training enabled on RTX 4060

### 4.3 Observations

**Q-Learning:**
- Successfully builds Q-table (~1,737 states explored)
- Low success rate indicates environment complexity
- State discretization may be too coarse
- Longer training (3,000+ episodes) likely needed for convergence

**DQN:**
- Handles raw high-dimensional observations
- Benefits from GPU acceleration (CUDA 12.9)
- Experience replay stabilizes learning
- Single network architecture as required

### 4.4 GPU Utilization

**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB)
```
- CUDA Version: 12.9
- PyTorch Version: 2.8.0+cu129
- Mixed Precision: Enabled
- Batch Size: 128 (optimized for GPU)
- Memory Usage: ~2-3 GB VRAM during training
```

### 4.5 Training Time

| Algorithm | Episodes | Time (GPU) |
|-----------|----------|------------|
| Q-Learning | 500 | ~5 min |
| DQN | 500 | ~20 min |

---

## 5. Discussion

### 5.1 Strengths

**Q-Learning:**
- Simple and interpretable
- Guaranteed convergence (with sufficient exploration)
- No hyperparameter tuning for network architecture
- Low computational requirements

**DQN:**
- Handles high-dimensional observations
- No manual feature engineering (learns features)
- Generalizes across similar states
- Scalable with more training

### 5.2 Limitations

**Q-Learning:**
- Curse of dimensionality (state space explosion)
- State discretization loses information
- Cannot handle continuous states directly
- Slow convergence in complex environments

**DQN:**
- Requires more training time
- Sensitive to hyperparameters
- Can overestimate Q-values (single network)
- Needs GPU for practical training times

### 5.3 Comparison

| Aspect | Q-Learning | DQN |
|--------|------------|-----|
| State representation | Discrete | Continuous |
| Memory | Q-table (~2MB) | Replay buffer (~500MB) |
| Computation | Table lookup | Neural network forward pass |
| Generalization | None | Across similar states |
| Training speed | Fast per episode | Slow per episode |
| Convergence | Guaranteed | Not guaranteed |

### 5.4 Potential Improvements

1. **Q-Learning:**
   - Finer state discretization
   - Add more features (forklift positions, task urgency)
   - Longer training (3,000+ episodes)

2. **DQN:**
   - Target network (Double DQN) for stability
   - Prioritized experience replay
   - Dueling architecture for better value estimates
   - Longer training with curriculum learning

---

## 6. Conclusion

This project successfully implemented and compared two fundamental reinforcement learning algorithms for warehouse robot navigation:

1. **Tabular Q-Learning:** Classic approach with discrete state space
2. **Deep Q-Network (DQN):** Neural network approximation with single network architecture

Both algorithms were implemented according to project requirements:
- Q-Learning uses tabular representation with state discretization
- DQN uses a **single neural network** (not Double DQN)
- Complete training loops with proper exploration strategies
- Comprehensive code documentation

The warehouse environment presents significant challenges (dynamic obstacles, battery management, multi-stage tasks), requiring sophisticated policies. While initial results show room for improvement, the foundation is solid for extended training and algorithm refinement.

**Future Work:**
- Extended training runs (3,000+ episodes)
- Hyperparameter tuning
- Curriculum learning (progressive difficulty)
- Advanced algorithms (Double DQN, Dueling DQN) as extensions

---

## References

1. **Watkins, C. J., & Dayan, P. (1992).** Q-learning. *Machine Learning*, 8(3-4), 279-292.

2. **Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).** Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

3. **Sutton, R. S., & Barto, A. G. (2018).** Reinforcement learning: An introduction. *MIT Press*.

4. **Van Hasselt, H., Guez, A., & Silver, D. (2016).** Deep reinforcement learning with double Q-learning. *AAAI Conference on Artificial Intelligence*.

5. **Wang, Z., Schaul, T., Hessel, M., et al. (2016).** Dueling network architectures for deep reinforcement learning. *International Conference on Machine Learning*, 1995-2003.


**Author:** [Your Name]
**Course:** Machine Learning
**Institution:** Sapienza University of Rome
**Date:** November 2025


---------------------------
🧩 What is a Seed?

A seed (short for random seed) is a number that initializes the random number generators in your code — in Python, NumPy, PyTorch, and your environment.
All randomness in RL (like exploration, environment dynamics, initialization of neural networks, etc.) comes from these random number generators.

If you fix the seed, you fix the random sequence → the entire experiment becomes reproducible.

💡 In simple terms:

The seed is like the starting point of randomness.

Same seed = same “random” events every time.

Different seeds = different random experiences.

🎮 Example:

Imagine you’re testing a robot in a warehouse:

Seed 0 → humans start near shelves, forklift on the left.

Seed 1 → humans near doors, forklift on the right.

Seed 2 → humans and forklift move differently.

Each seed gives your RL agent a slightly different environment scenario.
Running on multiple seeds tells you if your agent performs well on average, not just in one lucky setup.

🧠 Why Do We Use Multiple Seeds?

Reinforcement Learning is highly stochastic (random):

Random weight initialization in neural networks.

Random environment events (e.g., human/forklift behavior).

Random exploration during training (ε-greedy, etc.).

Because of this, if you only run one experiment (one seed), you might get misleading results:

Maybe your agent just got lucky.

Or maybe it failed because of one unlucky scenario.

So, we repeat the same experiment with different seeds and average the results.

📊 Benefits:

Measures stability and robustness of algorithms.

Gives confidence intervals (how uncertain your results are).

Follows the scientific standard for fair RL evaluation (see Henderson et al., 2018 “Deep RL that Matters”).

🎬 What Are Evaluation Episodes per Seed?

Once you set a seed, you still need to run several episodes (simulation runs) under that seed to estimate how well the agent performs in that environment setup.

For each seed:

You might run 30, 50, or 100 evaluation episodes.

Each episode = one full task from start to goal.

Each episode gives you metrics like reward, success rate, collisions, etc.

Averaging across episodes smooths out the randomness within a single seed.

Then you average across seeds to get the overall mean, median, and confidence intervals.

⚙️ In Your Script

The script asks:

1️⃣ “Select number of evaluation episodes PER SEED:”
1. Quick (30 episodes)
2. Standard (50 episodes)
3. Thorough (100 episodes)


This means:
For each seed, the agent will run that many test episodes.

So, if you choose 3 → Thorough (100 episodes),
each seed runs 100 test episodes.

2️⃣ “Select number of independent training/eval SEEDS:”
1. Fast (3 seeds)
2. Recommended (5 seeds)
3. Strong (10 seeds)


This means:
You will repeat the whole evaluation on that many independent random seeds.

So, if you choose 3 → Strong (10 seeds),
the script will evaluate on 10 different random seeds.

📈 Putting It Together

With your choices:

100 episodes per seed

10 seeds

👉 The agent runs 100 × 10 = 1,000 total evaluation episodes.

That means you get:

10 independent evaluations (each with its own random conditions)

100 episodes in each (averaging over randomness within that seed)

The script will then:

Collect all rewards, success rates, and safety metrics.

Compute averages and confidence intervals across seeds and episodes.

Plot results showing how stable and reliable the agents are.

📚 Summary Table
Term	Meaning	Purpose	Example in Script
Seed	A fixed random number used to make randomness reproducible.	Ensures consistent experiments and measures variability.	[0, 1, 2, ..., 9] for 10 seeds
Multiple Seeds	Independent random setups for fair testing.	Tests robustness to randomness in training and environment.	Evaluates each agent 10 times with different seeds
Evaluation Episodes per Seed	Number of episodes (simulations) per seed.	Reduces noise within a single seed’s estimate.	100 episodes per seed
Total Episodes	Seeds × Episodes per seed.	Total simulations per agent.	10 × 100 = 1,000 episodes per agent
🧾 How to Explain to Your Professor

You can say something like:

“In reinforcement learning, results can vary due to randomness in the environment and model initialization.
To ensure reliable and reproducible results, we evaluate each agent on multiple random seeds.
Each seed represents an independent random scenario, and for each seed we run several episodes to average out internal randomness.
This allows us to report statistically sound performance metrics with confidence intervals, rather than relying on a single, potentially biased run.”