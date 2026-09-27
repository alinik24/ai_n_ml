"""
Deep Q-Network (DQN) Training Loop - SINGLE NETWORK VERSION

This module implements the training procedure for the DQN agent using
a SINGLE neural network (not Double DQN).

Key Differences from Tabular Q-Learning:
    1. Uses raw observations (no discretization needed)
    2. Neural network approximates Q(s,a)
    3. Experience replay buffer stores past experiences
    4. Minibatch updates for gradient descent
    5. GPU acceleration available

Training Loop:
    For each episode:
        1. Reset environment
        2. While episode not done:
            a. Select action using ε-greedy (from neural network Q-values)
            b. Execute action in environment
            c. Store experience (s, a, r, s') in replay buffer
            d. Sample minibatch and update network via gradient descent
        3. Decay epsilon
        4. Track metrics and save best model
"""

import numpy as np
import torch

def train_dqn(env, agent, n_episodes=5000, save_path='models/dqn_best.pth'):
    """
    Train DQN Agent with SINGLE Network

    This function implements the complete DQN training loop using
    experience replay and neural network function approximation.

    Key Features:
        - Works with raw high-dimensional observations (20x20x12)
        - Experience replay for stable learning
        - GPU acceleration (if available)
        - Automatic model checkpointing

    Args:
        env: Warehouse environment
        agent: DQN agent instance (with single Q-network)
        n_episodes (int): Number of training episodes
        save_path (str): Path to save best model

    Returns:
        tuple: (episode_rewards, episode_lengths) - Training history
    """

    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    best_avg_reward = -float('inf')  # For saving best model

    print("Training DQN Agent...")
    print(f"Device: {agent.device}")

    # Main training loop
    for episode in range(n_episodes):
        # Reset environment - get initial observation
        obs, info = env.reset()

        episode_reward = 0
        done = False
        truncated = False
        steps = 0

        # Episode loop - agent interacts with environment
        while not done and not truncated:
            # 1. Select action from neural network Q-values (ε-greedy)
            action = agent.select_action(obs, training=True)

            # 2. Execute action in environment
            next_obs, reward, done, truncated, info = env.step(action)

            # 3. Store experience in replay buffer and train network
            # This internally:
            #   - Adds (s,a,r,s',done) to buffer
            #   - Samples random minibatch
            #   - Computes loss and updates weights via backprop
            agent.update(obs, action, reward, next_obs, done)

            # 4. Move to next state
            obs = next_obs
            episode_reward += reward
            steps += 1

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Track success
        if episode_reward > 100:
            success_count += 1

        # Print progress and save best model every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = success_count / 100

            # GPU memory info (if available)
            gpu_mem = ""
            if torch.cuda.is_available():
                gpu_mem = f" | GPU Mem: {agent.get_memory_usage()}"

            print(f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.2%} | Epsilon: {agent.epsilon:.3f}{gpu_mem}")

            success_count = 0

            # Save best model (checkpointing)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(save_path)
                print(f"  [NEW BEST] Saved model with avg reward: {avg_reward:.2f}")

    return episode_rewards, episode_lengths