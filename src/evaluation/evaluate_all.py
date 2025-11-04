"""
Comprehensive Evaluation Script
Evaluates all agents and generates comparison data for the report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import WarehouseAMREnv
from agents import QLearningAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent
from agents.baseline_agents import RandomAgent, GreedyAgent, RuleBasedAgent, evaluate_agent

def evaluate_all_agents(n_eval_episodes=100):
    """
    Evaluate all agents and generate comprehensive results
    """

    print("=" * 70)
    print("COMPREHENSIVE AGENT EVALUATION")
    print("=" * 70)

    # Create environment
    env = WarehouseAMREnv(config={'n_humans': 3, 'n_forklifts': 1, 'n_other_robots': 2})

    results = {}

    # 1. Evaluate Random Agent (Baseline)
    print("\n[1/5] Evaluating Random Agent...")
    random_agent = RandomAgent()
    results['Random'] = evaluate_agent(env, random_agent, n_episodes=n_eval_episodes)
    print(f"  Mean Reward: {results['Random']['mean_reward']:.2f}")
    print(f"  Success Rate: {results['Random']['success_rate']:.2%}")

    # 2. Evaluate Greedy Agent (Baseline)
    print("\n[2/5] Evaluating Greedy Agent...")
    greedy_agent = GreedyAgent()
    results['Greedy'] = evaluate_agent(env, greedy_agent, n_episodes=n_eval_episodes)
    print(f"  Mean Reward: {results['Greedy']['mean_reward']:.2f}")
    print(f"  Success Rate: {results['Greedy']['success_rate']:.2%}")

    # 3. Evaluate Rule-Based Agent (Baseline)
    print("\n[3/5] Evaluating Rule-Based Agent...")
    rule_agent = RuleBasedAgent()
    results['Rule-Based'] = evaluate_agent(env, rule_agent, n_episodes=n_eval_episodes)
    print(f"  Mean Reward: {results['Rule-Based']['mean_reward']:.2f}")
    print(f"  Success Rate: {results['Rule-Based']['success_rate']:.2%}")

    # 4. Evaluate Q-Learning (if model exists)
    print("\n[4/5] Evaluating Q-Learning Agent...")
    q_agent = QLearningAgent()
    if os.path.exists('models/q_learning_best.pkl'):
        q_agent.load('models/q_learning_best.pkl')
        results['Q-Learning'] = evaluate_agent(env, q_agent, n_episodes=n_eval_episodes)
        print(f"  Mean Reward: {results['Q-Learning']['mean_reward']:.2f}")
        print(f"  Success Rate: {results['Q-Learning']['success_rate']:.2%}")
    else:
        print("  Model not found. Train first with: python main.py")
        results['Q-Learning'] = None

    # 5. Evaluate DQN (if model exists)
    print("\n[5/7] Evaluating Simple DQN (Single Network)...")
    dqn_agent = DQNAgent()
    if os.path.exists('models/dqn_best.pth'):
        dqn_agent.load('models/dqn_best.pth')
        results['DQN (Simple)'] = evaluate_agent(env, dqn_agent, n_episodes=n_eval_episodes)
        print(f"  Mean Reward: {results['DQN (Simple)']['mean_reward']:.2f}")
        print(f"  Success Rate: {results['DQN (Simple)']['success_rate']:.2%}")
    else:
        print("  Model not found. Train first.")
        results['DQN (Simple)'] = None

    # 6. Evaluate Double DQN (if model exists)
    print("\n[6/7] Evaluating Double DQN...")
    double_dqn_agent = DoubleDQNAgent()
    if os.path.exists('models/double_dqn_best.pth'):
        double_dqn_agent.load('models/double_dqn_best.pth')
        results['Double DQN'] = evaluate_agent(env, double_dqn_agent, n_episodes=n_eval_episodes)
        print(f"  Mean Reward: {results['Double DQN']['mean_reward']:.2f}")
        print(f"  Success Rate: {results['Double DQN']['success_rate']:.2%}")
    else:
        print("  Model not found. Train with: python train_all_algorithms.py")
        results['Double DQN'] = None

    # 7. Evaluate Dueling DQN (if model exists)
    print("\n[7/7] Evaluating Dueling DQN...")
    dueling_dqn_agent = DuelingDQNAgent()
    if os.path.exists('models/dueling_dqn_best.pth'):
        dueling_dqn_agent.load('models/dueling_dqn_best.pth')
        results['Dueling DQN'] = evaluate_agent(env, dueling_dqn_agent, n_episodes=n_eval_episodes)
        print(f"  Mean Reward: {results['Dueling DQN']['mean_reward']:.2f}")
        print(f"  Success Rate: {results['Dueling DQN']['success_rate']:.2%}")
    else:
        print("  Model not found. Train with: python train_all_algorithms.py")
        results['Dueling DQN'] = None

    return results


def generate_comparison_table(results):
    """Generate comparison table for report"""

    # Convert to DataFrame
    data = []
    for agent_name, metrics in results.items():
        if metrics is not None:
            data.append({
                'Agent': agent_name,
                'Mean Reward': f"{metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}",
                'Success Rate': f"{metrics['success_rate']:.2%}",
                'Avg. Episode Length': f"{metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}",
                'Collision Rate': f"{metrics['collision_rate']:.2%}",
                'Battery Depletion': f"{metrics['battery_depletion_rate']:.2%}"
            })

    df = pd.DataFrame(data)

    # Save to CSV
    os.makedirs('results/metrics', exist_ok=True)
    df.to_csv('results/metrics/agent_comparison.csv', index=False)

    print("\n" + "=" * 70)
    print("AGENT COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    return df


def generate_comparison_plots(results):
    """Generate comparison visualizations"""

    os.makedirs('results/figures', exist_ok=True)

    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("No valid results to plot")
        return

    # 1. Mean Reward Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Mean Rewards
    agents = list(valid_results.keys())
    mean_rewards = [valid_results[a]['mean_reward'] for a in agents]
    std_rewards = [valid_results[a]['std_reward'] for a in agents]

    axes[0, 0].bar(agents, mean_rewards, yerr=std_rewards, capsize=5,
                   color=['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(agents)])
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_title('Average Reward Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Plot 2: Success Rate
    success_rates = [valid_results[a]['success_rate'] * 100 for a in agents]
    axes[0, 1].bar(agents, success_rates,
                   color=['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(agents)])
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_title('Task Success Rate')
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Episode Length
    mean_lengths = [valid_results[a]['mean_length'] for a in agents]
    std_lengths = [valid_results[a]['std_length'] for a in agents]
    axes[1, 0].bar(agents, mean_lengths, yerr=std_lengths, capsize=5,
                   color=['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(agents)])
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Average Episode Length')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Collision Rate
    collision_rates = [valid_results[a]['collision_rate'] * 100 for a in agents]
    axes[1, 1].bar(agents, collision_rates,
                   color=['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(agents)])
    axes[1, 1].set_ylabel('Collision Rate (%)')
    axes[1, 1].set_title('Safety: Collision Rate')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/agent_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[SAVE] Comparison plots saved to results/figures/agent_comparison.png")

    plt.close()

    # 2. Create radar chart for multi-metric comparison
    create_radar_chart(valid_results)


def create_radar_chart(results):
    """Create radar chart for multi-dimensional comparison"""

    from math import pi

    agents = list(results.keys())

    # Normalize metrics to 0-100 scale
    metrics = {
        'Success Rate': [results[a]['success_rate'] * 100 for a in agents],
        'Reward (norm)': [(results[a]['mean_reward'] + 1500) / 30 for a in agents],  # Normalize
        'Efficiency': [100 / max(results[a]['mean_length'], 1) * 100 for a in agents],
        'Safety': [(1 - results[a]['collision_rate']) * 100 for a in agents],
        'Battery Mgmt': [(1 - results[a]['battery_depletion_rate']) * 100 for a in agents]
    }

    categories = list(metrics.keys())
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']

    for idx, agent in enumerate(agents):
        values = [metrics[cat][idx] for cat in categories]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=agent, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.title('Multi-Metric Agent Comparison', size=14, pad=20)
    plt.savefig('results/figures/radar_comparison.png', dpi=300, bbox_inches='tight')
    print("[SAVE] Radar chart saved to results/figures/radar_comparison.png")

    plt.close()


def main():
    """Main evaluation function"""

    # Evaluate all agents
    results = evaluate_all_agents(n_eval_episodes=100)

    # Generate comparison table
    df = generate_comparison_table(results)

    # Generate plots
    generate_comparison_plots(results)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - results/metrics/agent_comparison.csv")
    print(f"  - results/figures/agent_comparison.png")
    print(f"  - results/figures/radar_comparison.png")
    print(f"\nUse these results in your report!")


if __name__ == '__main__':
    main()
