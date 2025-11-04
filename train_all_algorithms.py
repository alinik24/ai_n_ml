"""
Train All RL Algorithms - Comprehensive Comparison
Trains: Q-Learning, DQN, Double DQN, Dueling DQN
Optimized for RTX 4060 GPU
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from datetime import datetime

from src.environment import WarehouseAMREnv
from src.agents import QLearningAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent
from src.training import train_q_learning, train_dqn

def train_advanced_dqn(env, agent, n_episodes=5000, save_path='models/advanced_dqn.pth', agent_name="Advanced DQN"):
    """
    Train advanced DQN variants (Double DQN, Dueling DQN)
    Reuses DQN training logic
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    best_avg_reward = -float('inf')

    print(f"Training {agent_name}...")
    print(f"Device: {agent.device}")

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

            # GPU memory info
            gpu_mem = ""
            if torch.cuda.is_available():
                gpu_mem = f" | GPU Mem: {agent.get_memory_usage()}"

            print(f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.2%} | Epsilon: {agent.epsilon:.3f}{gpu_mem}")

            success_count = 0

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(save_path)
                print(f"  [NEW BEST] Saved model with avg reward: {avg_reward:.2f}")

    return episode_rewards, episode_lengths


def main():
    """
    Train all algorithms and save results
    """

    print("=" * 80)
    print("COMPREHENSIVE RL ALGORITHM TRAINING")
    print("Training: Q-Learning, DQN, Double DQN, Dueling DQN")
    print("=" * 80)

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

    # GPU info
    if torch.cuda.is_available():
        print(f"\n[GPU INFO]")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        use_gpu = True
    else:
        print(f"\n[CPU MODE] No GPU detected")
        use_gpu = False

    # Create environment
    env = WarehouseAMREnv(config={'n_humans': 3, 'n_forklifts': 1, 'n_other_robots': 2})

    results = {}

    # 1. Train Q-Learning
    print("\n" + "=" * 80)
    print("[1/4] Training Tabular Q-Learning")
    print("=" * 80)
    q_agent = QLearningAgent()
    q_rewards, q_lengths = train_q_learning(env, q_agent, n_episodes=3000)
    results['Q-Learning'] = {'rewards': q_rewards, 'lengths': q_lengths}

    # 2. Train Simple DQN (mandatory - single network)
    print("\n" + "=" * 80)
    print("[2/4] Training Simple DQN (Single Network - Mandatory)")
    print("=" * 80)
    use_amp = use_gpu and torch.cuda.get_device_capability()[0] >= 7
    dqn_agent = DQNAgent(use_amp=use_amp)
    dqn_rewards, dqn_lengths = train_dqn(env, dqn_agent, n_episodes=4000)
    results['DQN'] = {'rewards': dqn_rewards, 'lengths': dqn_lengths}

    # 3. Train Double DQN
    print("\n" + "=" * 80)
    print("[3/4] Training Double DQN (Reduces Overestimation)")
    print("=" * 80)
    double_dqn_agent = DoubleDQNAgent(use_amp=use_amp, target_update_freq=1000)
    double_rewards, double_lengths = train_advanced_dqn(
        env, double_dqn_agent, n_episodes=4000,
        save_path='models/double_dqn_best.pth',
        agent_name="Double DQN"
    )
    results['Double DQN'] = {'rewards': double_rewards, 'lengths': double_lengths}

    # 4. Train Dueling DQN
    print("\n" + "=" * 80)
    print("[4/4] Training Dueling DQN (Separate Value/Advantage)")
    print("=" * 80)
    dueling_dqn_agent = DuelingDQNAgent(use_amp=use_amp, target_update_freq=1000)
    dueling_rewards, dueling_lengths = train_advanced_dqn(
        env, dueling_dqn_agent, n_episodes=4000,
        save_path='models/dueling_dqn_best.pth',
        agent_name="Dueling DQN"
    )
    results['Dueling DQN'] = {'rewards': dueling_rewards, 'lengths': dueling_lengths}

    # Save results to CSV
    print("\n" + "=" * 80)
    print("Saving Training Results")
    print("=" * 80)

    for algo_name, data in results.items():
        df = pd.DataFrame({
            'episode': range(len(data['rewards'])),
            'reward': data['rewards'],
            'length': data['lengths']
        })
        filename = f"results/metrics/{algo_name.lower().replace(' ', '_')}_training.csv"
        df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")

    # Generate comparison plot
    generate_comparison_plot(results)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run evaluation: python src/evaluation/evaluate_all.py")
    print("  2. Check results: results/figures/")
    print("  3. Review metrics: results/metrics/")


def generate_comparison_plot(results):
    """Generate learning curves comparison"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {
        'Q-Learning': '#1f77b4',
        'DQN': '#ff7f0e',
        'Double DQN': '#2ca02c',
        'Dueling DQN': '#d62728'
    }

    # Plot 1: Rewards (raw)
    ax1 = axes[0, 0]
    for algo_name, data in results.items():
        ax1.plot(data['rewards'], alpha=0.3, color=colors[algo_name])

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Raw Episode Rewards')
    ax1.grid(True, alpha=0.3)
    ax1.legend(results.keys())

    # Plot 2: Smoothed rewards (100-episode moving average)
    ax2 = axes[0, 1]
    for algo_name, data in results.items():
        smoothed = pd.Series(data['rewards']).rolling(100, min_periods=1).mean()
        ax2.plot(smoothed, label=algo_name, linewidth=2, color=colors[algo_name])

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward (100-ep window)')
    ax2.set_title('Smoothed Learning Curves')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Plot 3: Episode lengths
    ax3 = axes[1, 0]
    for algo_name, data in results.items():
        smoothed = pd.Series(data['lengths']).rolling(100, min_periods=1).mean()
        ax3.plot(smoothed, label=algo_name, linewidth=2, color=colors[algo_name])

    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length (steps)')
    ax3.set_title('Episode Length Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Success rate (reward > 100)
    ax4 = axes[1, 1]
    for algo_name, data in results.items():
        success = [1 if r > 100 else 0 for r in data['rewards']]
        success_rate = pd.Series(success).rolling(100, min_periods=1).mean() * 100
        ax4.plot(success_rate, label=algo_name, linewidth=2, color=colors[algo_name])

    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Task Success Rate (100-ep window)')
    ax4.set_ylim([-5, 105])
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('results/figures/all_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n[SAVE] Comparison plot saved to results/figures/all_algorithms_comparison.png")
    plt.close()


if __name__ == '__main__':
    # Start training
    start_time = datetime.now()
    print(f"\nTraining started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    main()

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nTraining completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
