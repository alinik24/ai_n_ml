"""
Tabular Q-Learning Training Loop

This module implements the training procedure for the tabular Q-learning agent.

Training Loop:
    For each episode:
        1. Reset environment and get initial state
        2. While episode not done:
            a. Select action using ε-greedy policy
            b. Execute action in environment
            c. Observe reward and next state
            d. Update Q-table: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        3. Decay epsilon (reduce exploration over time)
        4. Track episode metrics

The agent gradually learns optimal Q-values through repeated interaction
with the environment, balancing exploration and exploitation.
"""

import numpy as np
import pickle

def train_q_learning(env, agent, n_episodes=3000, save_path='models/q_learning_best.pkl'):
    """
    Train Tabular Q-Learning Agent

    This function implements the complete training loop for Q-learning.
    The agent learns by trial and error, gradually building up a Q-table
    that maps states to optimal actions.

    Args:
        env: Warehouse environment
        agent: Q-Learning agent instance
        n_episodes (int): Number of training episodes
        save_path (str): Path to save trained model

    Returns:
        tuple: (episode_rewards, episode_lengths) - Training history
    """

    # Metrics tracking
    episode_rewards = []   # Cumulative reward per episode
    episode_lengths = []   # Steps taken per episode
    success_count = 0      # Episodes with reward > 100

    print("Training Q-Learning Agent...")

    # Main training loop - each episode is one complete task sequence
    for episode in range(n_episodes):
        # Reset environment to initial state
        obs, info = env.reset()
        state = agent.discretize_state(obs)  # Convert to discrete state

        episode_reward = 0
        done = False        # Task completion flag
        truncated = False   # Max steps reached flag
        steps = 0

        # Episode loop - interact with environment until done
        while not done and not truncated:
            # 1. Select action using epsilon-greedy policy
            action = agent.select_action(state, training=True)

            # 2. Execute action and observe result
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_obs)

            # 3. Update Q-table using Q-learning update rule
            # Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
            agent.update(state, action, reward, next_state, done)

            # 4. Move to next state
            state = next_state
            episode_reward += reward
            steps += 1

        # Decay exploration rate after each episode
        # Gradually shift from exploration to exploitation
        agent.decay_epsilon()

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Track successful episodes (reward > 100 indicates task completion)
        if episode_reward > 100:
            success_count += 1

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = success_count / 100
            print(f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.2%} | Epsilon: {agent.epsilon:.3f}")
            success_count = 0
    
    # Save model with metadata
    print(f"\n[SAVE] Saving Q-Learning model to {save_path}...")
    with open(save_path, 'wb') as f:
        model_data = {
            'q_table': agent.q_table,
            'epsilon': agent.epsilon,
            'n_actions': agent.n_actions,
            'q_table_size': len(agent.q_table)
        }
        pickle.dump(model_data, f)
    print(f"[SAVE] Model saved successfully! Q-table size: {len(agent.q_table)} states")

    return episode_rewards, episode_lengths