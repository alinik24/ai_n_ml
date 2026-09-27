"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠️  OPTIONAL IMPLEMENTATION ⚠️                        ║
║                    ADVANCED DQN VARIANTS TRAINING LOOPS                      ║
║                                                                              ║
║  This file contains training functions for ADVANCED algorithms NOT          ║
║  required for the project:                                                  ║
║    🔬 Double DQN - Reduces Q-value overestimation                           ║
║    🔬 Dueling DQN - Separate value and advantage streams                    ║
║                                                                              ║
║  These are state-of-the-art extensions for comparison purposes only.        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Double DQN Training:
    Key difference from Simple DQN:
        - Uses TWO networks: online network and target network
        - Online network selects actions: a* = argmax_a Q_online(s',a)
        - Target network evaluates actions: Q_target(s', a*)
        - Reduces overestimation bias in Q-value updates

Dueling DQN Training:
    Key difference from Simple DQN:
        - Network architecture splits into value and advantage streams
        - Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        - Better generalization across actions
        - Learns which states are valuable independent of action
"""

import numpy as np
import torch


def train_double_dqn(env, agent, n_episodes=5000, save_path='models/double_dqn_best.pth'):
    """
    Train Double DQN Agent

    Double DQN improves upon Simple DQN by using separate networks for
    action selection and evaluation, reducing Q-value overestimation.

    Key Features:
        - Two networks: online (action selection) + target (evaluation)
        - Decoupled action selection from value estimation
        - Reduced overestimation bias
        - More stable learning than Simple DQN

    Algorithm:
        1. Select action: a* = argmax_a Q_online(s', a)
        2. Evaluate action: y = r + γ Q_target(s', a*)
        3. Update online network to minimize (Q_online(s,a) - y)²

    Args:
        env: Warehouse environment
        agent: DoubleDQNAgent instance (with online + target networks)
        n_episodes (int): Number of training episodes
        save_path (str): Path to save best model

    Returns:
        tuple: (episode_rewards, episode_lengths) - Training history
    """

    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    best_avg_reward = -float('inf')

    print("\n[TRAINING] Double DQN...")
    print(f"  Device: {agent.device}")

    # Main training loop
    for episode in range(n_episodes):
        obs, info = env.reset()

        episode_reward = 0
        done = False
        truncated = False
        steps = 0

        # Episode loop - agent interacts with environment
        while not done and not truncated:
            # Select action (from online network)
            action = agent.select_action(obs, training=True)

            # Execute action
            next_obs, reward, done, truncated, info = env.step(action)

            # Update networks (Double DQN update rule)
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            steps += 1

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if episode_reward > 100:
            success_count += 1

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = success_count / 100

            # GPU memory info
            gpu_mem = ""
            if torch.cuda.is_available():
                gpu_mem = f" | GPU Mem: {agent.get_memory_usage()}"

            print(f"  Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.2%} | Epsilon: {agent.epsilon:.3f}{gpu_mem}")

            success_count = 0

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(save_path)
                print(f"  [NEW BEST] Saved model with avg reward: {avg_reward:.2f}")

    return episode_rewards, episode_lengths


def train_dueling_dqn(env, agent, n_episodes=5000, save_path='models/dueling_dqn_best.pth'):
    """
    Train Dueling DQN Agent

    Dueling DQN uses a specialized network architecture that separates
    the estimation of state value and action advantages, leading to
    better generalization.

    Key Features:
        - Dual-stream architecture: value stream V(s) + advantage stream A(s,a)
        - Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        - Better at learning which states are valuable
        - More stable training with sparse rewards

    Architecture:
        Shared CNN → Split into:
            1. Value stream: V(s) - "How good is this state?"
            2. Advantage stream: A(s,a) - "How much better is action a?"
        Combine: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    Args:
        env: Warehouse environment
        agent: DuelingDQNAgent instance (with dueling architecture)
        n_episodes (int): Number of training episodes
        save_path (str): Path to save best model

    Returns:
        tuple: (episode_rewards, episode_lengths) - Training history
    """

    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    best_avg_reward = -float('inf')

    print("\n[TRAINING] Dueling DQN...")
    print(f"  Device: {agent.device}")

    # Main training loop
    for episode in range(n_episodes):
        obs, info = env.reset()

        episode_reward = 0
        done = False
        truncated = False
        steps = 0

        # Episode loop - agent interacts with environment
        while not done and not truncated:
            # Select action (Q-values from dueling architecture)
            action = agent.select_action(obs, training=True)

            # Execute action
            next_obs, reward, done, truncated, info = env.step(action)

            # Update network (dueling architecture)
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            steps += 1

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if episode_reward > 100:
            success_count += 1

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = success_count / 100

            # GPU memory info
            gpu_mem = ""
            if torch.cuda.is_available():
                gpu_mem = f" | GPU Mem: {agent.get_memory_usage()}"

            print(f"  Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.2%} | Epsilon: {agent.epsilon:.3f}{gpu_mem}")

            success_count = 0

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(save_path)
                print(f"  [NEW BEST] Saved model with avg reward: {avg_reward:.2f}")

    return episode_rewards, episode_lengths
