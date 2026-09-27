"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             BEST-PRACTICE RL EVALUATION - ACADEMIC RIGOR                     ║
║                                                                              ║
║  Following Henderson et al. (2018), Agarwal et al. (2021) guidelines:       ║
║    • Multi-seed evaluation (3/5/10 seeds)                                   ║
║    • Deterministic test-time policy (ε=0)                                   ║
║    • Stratified bootstrap CIs over seeds                                    ║
║    • IQM (Interquartile Mean) for outlier robustness                        ║
║    • Wilson score intervals for proportions (success rate)                  ║
║    • Per-seed, per-episode data logging for reproducibility                 ║
║    • Statistical significance testing (Mann-Whitney U)                      ║
║                                                                              ║
║  References:                                                                ║
║    - Henderson et al. (2018): Deep RL Reproducibility Study                 ║
║    - Agarwal et al. (2021): Deep RL at Statistical Significance Edge        ║
║    - Clopper & Pearson (1934): Wilson score for proportions                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Optional: torch for seed control
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import WarehouseAMREnv

# ==============================================================================
# MANDATORY AGENTS
# ==============================================================================
from agents import QLearningAgent, DQNAgent

# ==============================================================================
# OPTIONAL AGENTS
# ==============================================================================
try:
    from agents import DoubleDQNAgent, DuelingDQNAgent
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

try:
    from agents.baseline_agents import RandomAgent, GreedyAgent, RuleBasedAgent, evaluate_agent
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def ask_user_choice(prompt: str, options: List[str]) -> str:
    """Interactive selection utility."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    while True:
        choice = input(f"\nEnter choice (1-{len(options)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"Invalid choice. Please enter 1-{len(options)}.")


def set_all_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Critical for fair comparison across agents with multiple runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_deterministic_eval_mode(agent):
    """
    Set agent to deterministic evaluation mode (ε=0).

    Best practice: Evaluation should use deterministic policy to simulate
    real deployment conditions (no exploration randomness).

    Reference: Henderson et al. (2018) - Section 3.2
    """
    if hasattr(agent, "epsilon"):
        try:
            agent.epsilon = 0.0
            print("    [EVAL MODE] ε=0 (deterministic policy)")
        except Exception:
            pass


def interquartile_mean(values: np.ndarray) -> float:
    """
    Interquartile Mean (IQM): Mean of middle 50% of data.

    Recommended by Agarwal et al. (2021) for RL as it's:
    - More robust to outliers than mean
    - Less affected by catastrophic failures or lucky runs
    - Better represents "typical" performance

    Reference: Agarwal et al. (2021) - Section 4.3
    """
    sorted_vals = np.sort(values.flatten())
    n = len(sorted_vals)
    q1 = int(0.25 * n)
    q3 = int(0.75 * n)
    if q3 <= q1:
        return float(np.mean(sorted_vals))
    return float(np.mean(sorted_vals[q1:q3]))


def stratified_bootstrap_ci(
    per_seed_values: List[np.ndarray],
    stat_func=np.mean,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Stratified bootstrap confidence interval over seeds.

    Resamples WITH replacement at the seed level, then aggregates episodes
    within each resampled seed. This respects the hierarchical structure
    of seed -> episodes and provides proper uncertainty quantification.

    Reference: Henderson et al. (2018) - Appendix C

    Args:
        per_seed_values: List of arrays, one per seed
        stat_func: Statistic to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (default: 0.05 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        (statistic, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n_seeds = len(per_seed_values)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample seeds with replacement
        seed_indices = rng.choice(n_seeds, size=n_seeds, replace=True)

        # Collect all episodes from resampled seeds
        resampled_data = []
        for idx in seed_indices:
            # Also resample episodes within seed
            seed_data = per_seed_values[idx]
            n_episodes = len(seed_data)
            episode_indices = rng.choice(n_episodes, size=n_episodes, replace=True)
            resampled_data.append(seed_data[episode_indices])

        # Concatenate and compute statistic
        all_data = np.concatenate(resampled_data)
        bootstrap_stats.append(stat_func(all_data))

    # Compute percentiles
    stat_value = stat_func(np.concatenate(per_seed_values))
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(stat_value), float(ci_lower), float(ci_upper)


def wilson_score_interval(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Wilson score interval for binomial proportion (success rate).

    More accurate than normal approximation for proportions, especially
    with small sample sizes or extreme proportions (near 0 or 1).

    Reference: Wilson (1927), recommended by Agresti & Coull (1998)

    Args:
        successes: Number of successes
        trials: Total number of trials
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        (proportion, ci_lower, ci_upper)
    """
    if trials == 0:
        return 0.0, 0.0, 0.0

    p = successes / trials
    z = stats.norm.ppf(1 - alpha / 2)  # z-score for desired confidence

    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) / trials + z**2 / (4 * trials**2))) / denominator

    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)

    return float(p), float(ci_lower), float(ci_upper)


def mann_whitney_u_test(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Mann-Whitney U test for statistical significance.

    Non-parametric test, doesn't assume normal distribution.
    Appropriate for RL where returns are often non-normal.

    H0: The two distributions are identical
    H1: Values in one group tend to be larger than in the other

    Args:
        group1, group2: Arrays of values to compare
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'effect_size': float(statistic / (len(group1) * len(group2))),  # rank-biserial correlation
        'interpretation': 'significant' if p_value < alpha else 'not significant'
    }


@dataclass
class AgentEvalResults:
    """Container for agent evaluation results."""
    agent: str
    seeds: List[int]
    n_episodes_per_seed: int

    # Return statistics
    mean_return: float
    median_return: float
    iqm_return: float
    std_return: float
    ci_lower_return: float
    ci_upper_return: float

    # Success rate with Wilson CI
    success_rate: float
    success_ci_lower: float
    success_ci_upper: float

    # Other metrics
    mean_length: float
    collision_rate: float
    battery_depletion_rate: float

    # Per-seed summaries for statistical testing
    per_seed_mean_returns: List[float]

    # Raw data for reproducibility
    all_returns: List[float]
    available: bool = True


# ==============================================================================
# EVALUATION ENGINE
# ==============================================================================

def evaluate_agent_multi_seed(
    agent_name: str,
    n_episodes: int,
    seeds: List[int],
    deterministic: bool = True
) -> AgentEvalResults:
    """
    Evaluate agent across multiple seeds with best practices.

    Process:
    1. Load agent and set to deterministic mode
    2. For each seed:
       - Reset environment with seed
       - Run n_episodes episodes
       - Collect per-episode metrics
    3. Aggregate across all seeds with proper statistics

    Args:
        agent_name: Agent to evaluate
        n_episodes: Episodes per seed
        seeds: List of random seeds
        deterministic: Use deterministic policy (recommended)

    Returns:
        AgentEvalResults with comprehensive statistics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {agent_name}")
    print(f"{'='*80}")

    # Instantiate agent
    agent, model_path = instantiate_agent(agent_name)
    if agent is None:
        print(f"[SKIP] Agent not available")
        return AgentEvalResults(
            agent=agent_name, seeds=[], n_episodes_per_seed=0,
            mean_return=0, median_return=0, iqm_return=0, std_return=0,
            ci_lower_return=0, ci_upper_return=0,
            success_rate=0, success_ci_lower=0, success_ci_upper=0,
            mean_length=0, collision_rate=0, battery_depletion_rate=0,
            per_seed_mean_returns=[], all_returns=[], available=False
        )

    # Load weights
    if model_path and os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"  [✓] Loaded: {model_path}")
        except Exception as e:
            print(f"  [✗] Load failed: {e}")
            return AgentEvalResults(
                agent=agent_name, seeds=[], n_episodes_per_seed=0,
                mean_return=0, median_return=0, iqm_return=0, std_return=0,
                ci_lower_return=0, ci_upper_return=0,
                success_rate=0, success_ci_lower=0, success_ci_upper=0,
                mean_length=0, collision_rate=0, battery_depletion_rate=0,
                per_seed_mean_returns=[], all_returns=[], available=False
            )
    elif model_path:
        print(f"  [✗] Model not found: {model_path}")
        print(f"      Train first: python train.py")
        return AgentEvalResults(
            agent=agent_name, seeds=[], n_episodes_per_seed=0,
            mean_return=0, median_return=0, iqm_return=0, std_return=0,
            ci_lower_return=0, ci_upper_return=0,
            success_rate=0, success_ci_lower=0, success_ci_upper=0,
            mean_length=0, collision_rate=0, battery_depletion_rate=0,
            per_seed_mean_returns=[], all_returns=[], available=False
        )

    # Set deterministic mode
    if deterministic:
        set_deterministic_eval_mode(agent)

    # Storage for seed-level results
    per_seed_returns = []
    per_seed_lengths = []
    per_seed_success_rates = []
    per_seed_collision_rates = []
    per_seed_battery_rates = []

    # Evaluate each seed
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n  [{seed_idx}/{len(seeds)}] Seed {seed}:")
        set_all_seeds(seed)

        # Create environment
        env = WarehouseAMREnv(config={'n_humans': 3, 'n_forklifts': 1, 'n_other_robots': 2})

        # Evaluate using baseline function if available
        if BASELINES_AVAILABLE:
            metrics = evaluate_agent(env, agent, n_episodes=n_episodes)

            # Extract episode-level data
            episode_returns = metrics.get('episode_rewards', [metrics.get('mean_reward', 0)] * n_episodes)
            if not isinstance(episode_returns, (list, np.ndarray)):
                episode_returns = [episode_returns] * n_episodes
            episode_returns = np.array(episode_returns, dtype=float)

            success_rate = metrics.get('success_rate', 0.0)
            collision_rate = metrics.get('collision_rate', 0.0)
            battery_rate = metrics.get('battery_depletion_rate', 0.0)

            episode_lengths = metrics.get('episode_length', [metrics.get('mean_length', 0)] * n_episodes)
            if not isinstance(episode_lengths, (list, np.ndarray)):
                episode_lengths = [episode_lengths] * n_episodes
            episode_lengths = np.array(episode_lengths, dtype=float)
        else:
            # Fallback: manual evaluation loop
            episode_returns = []
            episode_lengths = []
            successes = 0

            for ep in range(n_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False
                ep_return = 0
                steps = 0

                while not (done or truncated):
                    # Agent selects action
                    if hasattr(agent, 'select_action'):
                        action = agent.select_action(obs, training=False)
                    else:
                        action = agent.select_action(obs)

                    obs, reward, done, truncated, info = env.step(action)
                    ep_return += reward
                    steps += 1

                episode_returns.append(ep_return)
                episode_lengths.append(steps)
                if ep_return > 100:  # Task success threshold
                    successes += 1

            episode_returns = np.array(episode_returns)
            episode_lengths = np.array(episode_lengths)
            success_rate = successes / n_episodes
            collision_rate = 0.0  # Not tracked in fallback
            battery_rate = 0.0

        # Store seed results
        per_seed_returns.append(episode_returns)
        per_seed_lengths.append(episode_lengths)
        per_seed_success_rates.append(success_rate)
        per_seed_collision_rates.append(collision_rate)
        per_seed_battery_rates.append(battery_rate)

        # Print seed summary
        print(f"      Mean Return: {np.mean(episode_returns):.2f}")
        print(f"      Success Rate: {success_rate:.2%}")

    # ===========================================================================
    # AGGREGATE ACROSS SEEDS WITH PROPER STATISTICS
    # ===========================================================================

    # Concatenate all episodes from all seeds
    all_returns = np.concatenate(per_seed_returns)
    all_lengths = np.concatenate(per_seed_lengths)

    # Compute return statistics with bootstrap CI
    mean_return, ci_lower_return, ci_upper_return = stratified_bootstrap_ci(
        per_seed_returns, stat_func=np.mean, n_bootstrap=10000
    )
    median_return = float(np.median(all_returns))
    iqm_return = interquartile_mean(all_returns)
    std_return = float(np.std(all_returns))

    # Compute success rate with Wilson interval
    total_episodes = len(all_returns)
    total_successes = int(np.sum([sr * n_episodes for sr in per_seed_success_rates]))
    success_rate, success_ci_lower, success_ci_upper = wilson_score_interval(
        total_successes, total_episodes
    )

    # Other aggregates
    mean_length = float(np.mean(all_lengths))
    mean_collision = float(np.mean(per_seed_collision_rates))
    mean_battery = float(np.mean(per_seed_battery_rates))

    # Per-seed means for statistical testing
    per_seed_means = [float(np.mean(returns)) for returns in per_seed_returns]

    # Print summary
    print(f"\n  {'='*76}")
    print(f"  SUMMARY across {len(seeds)} seeds × {n_episodes} episodes = {total_episodes} total")
    print(f"  {'='*76}")
    print(f"  Mean Return:   {mean_return:8.2f}  (95% CI: [{ci_lower_return:.2f}, {ci_upper_return:.2f}])")
    print(f"  Median Return: {median_return:8.2f}")
    print(f"  IQM Return:    {iqm_return:8.2f}  (robust)")
    print(f"  Std Return:    {std_return:8.2f}")
    print(f"  Success Rate:  {success_rate:7.1%}   (95% CI: [{success_ci_lower:.1%}, {success_ci_upper:.1%}])")
    print(f"  {'='*76}")

    return AgentEvalResults(
        agent=agent_name,
        seeds=seeds,
        n_episodes_per_seed=n_episodes,
        mean_return=mean_return,
        median_return=median_return,
        iqm_return=iqm_return,
        std_return=std_return,
        ci_lower_return=ci_lower_return,
        ci_upper_return=ci_upper_return,
        success_rate=success_rate,
        success_ci_lower=success_ci_lower,
        success_ci_upper=success_ci_upper,
        mean_length=mean_length,
        collision_rate=mean_collision,
        battery_depletion_rate=mean_battery,
        per_seed_mean_returns=per_seed_means,
        all_returns=all_returns.tolist(),
        available=True
    )


def instantiate_agent(name: str) -> Tuple[Any, Optional[str]]:
    """Create agent instance by name."""
    if name == "Q-Learning":
        return QLearningAgent(), 'models/q_learning_best.pkl'
    elif name == "Simple DQN":
        return DQNAgent(), 'models/dqn_best.pth'
    elif name == "Double DQN" and ADVANCED_AVAILABLE:
        return DoubleDQNAgent(), 'models/double_dqn_best.pth'
    elif name == "Dueling DQN" and ADVANCED_AVAILABLE:
        return DuelingDQNAgent(), 'models/dueling_dqn_best.pth'
    elif name == "Random" and BASELINES_AVAILABLE:
        return RandomAgent(), None
    elif name == "Greedy" and BASELINES_AVAILABLE:
        return GreedyAgent(), None
    elif name == "Rule-Based" and BASELINES_AVAILABLE:
        return RuleBasedAgent(), None
    return None, None


# ==============================================================================
# VISUALIZATION - PUBLICATION QUALITY
# ==============================================================================

def ensure_dirs():
    """Create output directories."""
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/raw_data', exist_ok=True)


def generate_results_table(results: Dict[str, AgentEvalResults]) -> pd.DataFrame:
    """Generate comprehensive results table."""
    ensure_dirs()

    rows = []
    for name, res in results.items():
        if not res.available:
            continue

        rows.append({
            'Agent': name,
            'Mean Return': f"{res.mean_return:.2f}",
            '95% CI': f"[{res.ci_lower_return:.2f}, {res.ci_upper_return:.2f}]",
            'Median': f"{res.median_return:.2f}",
            'IQM': f"{res.iqm_return:.2f}",
            'Std': f"{res.std_return:.2f}",
            'Success Rate': f"{res.success_rate:.2%}",
            'Success 95% CI': f"[{res.success_ci_lower:.2%}, {res.success_ci_upper:.2%}]",
            'Avg Length': f"{res.mean_length:.1f}",
            'Collision Rate': f"{res.collision_rate:.2%}",
            'Seeds': len(res.seeds),
            'Episodes': res.n_episodes_per_seed * len(res.seeds)
        })

    df = pd.DataFrame(rows)
    df.to_csv('results/metrics/evaluation_results.csv', index=False)

    print("\n" + "=" * 120)
    print("EVALUATION RESULTS")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)

    return df


def plot_performance_comparison(results: Dict[str, AgentEvalResults]):
    """
    Figure 1: Performance comparison with proper CIs.

    Shows mean returns with bootstrap 95% CIs and success rates with Wilson CIs.
    """
    ensure_dirs()

    valid = {k: v for k, v in results.items() if v.available}
    if not valid:
        return

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    agents = list(valid.keys())
    colors = sns.color_palette("husl", len(agents))
    y_pos = np.arange(len(agents))

    # Panel 1: Returns with Bootstrap CI
    means = [valid[a].mean_return for a in agents]
    ci_lows = [valid[a].ci_lower_return for a in agents]
    ci_highs = [valid[a].ci_upper_return for a in agents]

    errors_low = [m - l for m, l in zip(means, ci_lows)]
    errors_high = [h - m for h, m in zip(ci_highs, means)]

    ax1.barh(y_pos, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.errorbar(means, y_pos, xerr=[errors_low, errors_high],
                fmt='none', ecolor='black', capsize=6, capthick=2.5, linewidth=2)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(agents, fontweight='bold')
    ax1.set_xlabel('Mean Return', fontweight='bold', fontsize=13)
    ax1.set_title('(A) Returns with 95% Bootstrap CI\n(stratified over seeds)',
                 fontweight='bold', fontsize=14)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, m in enumerate(means):
        label_x = m + 80 if m > 0 else m - 80
        ax1.text(label_x, i, f'{m:.0f}', va='center', ha='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Panel 2: Success Rates with Wilson CI
    success = [valid[a].success_rate * 100 for a in agents]
    success_ci_low = [valid[a].success_ci_lower * 100 for a in agents]
    success_ci_high = [valid[a].success_ci_upper * 100 for a in agents]

    # Calculate error bar magnitudes (must be non-negative)
    errors_s_low = [max(0, s - l) for s, l in zip(success, success_ci_low)]
    errors_s_high = [max(0, h - s) for h, s in zip(success_ci_high, success)]

    ax2.barh(y_pos, success, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.errorbar(success, y_pos, xerr=[errors_s_low, errors_s_high],
                fmt='none', ecolor='black', capsize=6, capthick=2.5, linewidth=2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(agents, fontweight='bold')
    ax2.set_xlabel('Success Rate (%)', fontweight='bold', fontsize=13)
    ax2.set_title('(B) Success Rates with 95% Wilson CI\n(proper for proportions)',
                 fontweight='bold', fontsize=14)
    ax2.set_xlim([0, 100])
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, s in enumerate(success):
        ax2.text(s + 3, i, f'{s:.1f}%', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[SAVED] results/figures/performance_comparison.png")
    plt.close()


def plot_robust_statistics(results: Dict[str, AgentEvalResults]):
    """Figure 2: Robust statistics - Mean vs IQM vs Median."""
    ensure_dirs()

    valid = {k: v for k, v in results.items() if v.available}
    if not valid:
        return

    agents = list(valid.keys())
    means = [valid[a].mean_return for a in agents]
    medians = [valid[a].median_return for a in agents]
    iqms = [valid[a].iqm_return for a in agents]

    x = np.arange(len(agents))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, means, width, label='Mean', alpha=0.8, color='#3498db', edgecolor='black')
    ax.bar(x, iqms, width, label='IQM (Robust)', alpha=0.8, color='#2ecc71', edgecolor='black')
    ax.bar(x + width, medians, width, label='Median', alpha=0.8, color='#e74c3c', edgecolor='black')

    ax.set_xlabel('Agent', fontweight='bold', fontsize=13)
    ax.set_ylabel('Return', fontweight='bold', fontsize=13)
    ax.set_title('Robust Aggregates Comparison\n(IQM recommended by Agarwal et al. 2021)',
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('results/figures/robust_statistics.png', dpi=300, bbox_inches='tight')
    print("[SAVED] results/figures/robust_statistics.png")
    plt.close()


def plot_statistical_significance(results: Dict[str, AgentEvalResults]):
    """
    Figure 3: Box plots showing per-seed distributions.

    Non-overlapping notches indicate statistically significant differences.
    """
    ensure_dirs()

    valid = {k: v for k, v in results.items() if v.available}
    if not valid:
        return

    agents = list(valid.keys())
    per_seed_data = [valid[a].per_seed_mean_returns for a in agents]

    fig, ax = plt.subplots(figsize=(12, 7))

    positions = np.arange(len(agents))
    bp = ax.boxplot(per_seed_data, positions=positions, widths=0.6,
                   patch_artist=True, notch=True,
                   boxprops=dict(linewidth=1.5),
                   medianprops=dict(color='red', linewidth=2.5),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))

    colors = sns.color_palette("husl", len(agents))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')

    ax.set_xticks(positions)
    ax.set_xticklabels(agents, fontweight='bold', fontsize=11)
    ax.set_ylabel('Mean Return per Seed', fontweight='bold', fontsize=13)
    ax.set_title('Statistical Significance: Per-Seed Return Distributions\n' +
                '(Notches = 95% CI around median; non-overlapping notches → significant difference)',
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)

    # Add sample size info
    n_seeds = len(valid[agents[0]].seeds)
    note = f"Based on {n_seeds} independent seeds per agent"
    ax.text(0.5, 0.98, note, transform=ax.transAxes,
           ha='center', va='top', fontsize=11, style='italic', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/figures/statistical_significance.png', dpi=300, bbox_inches='tight')
    print("[SAVED] results/figures/statistical_significance.png")
    plt.close()


def plot_absolute_metrics_table(results: Dict[str, AgentEvalResults]):
    """
    Figure 4: Absolute metrics table (REPLACES misleading radar chart).

    Shows actual values instead of normalized scores that can be misleading
    with few agents.
    """
    ensure_dirs()

    valid = {k: v for k, v in results.items() if v.available}
    if not valid:
        return

    agents = list(valid.keys())

    # Prepare data
    data = {
        'Agent': agents,
        'Mean Return': [valid[a].mean_return for a in agents],
        'IQM Return': [valid[a].iqm_return for a in agents],
        'Success (%)': [valid[a].success_rate * 100 for a in agents],
        'Avg Length': [valid[a].mean_length for a in agents],
        'Collision (%)': [valid[a].collision_rate * 100 for a in agents],
        'Std Return': [valid[a].std_return for a in agents],
    }

    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df_plot.values,
                    colLabels=df_plot.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(df_plot.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', fontsize=12)

    # Color code rows
    colors = sns.color_palette("husl", len(agents))
    for i in range(len(agents)):
        for j in range(len(df_plot.columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor((*colors[i], 0.3))
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)

            # Format numbers
            if j > 0:  # Skip agent name
                value = df_plot.iloc[i, j]
                if j in [1, 2, 6]:  # Returns and std
                    cell.get_text().set_text(f"{value:.1f}")
                elif j == 4:  # Length
                    cell.get_text().set_text(f"{value:.0f}")
                else:  # Percentages
                    cell.get_text().set_text(f"{value:.1f}")
                cell.get_text().set_fontweight('bold')

    plt.title('Absolute Performance Metrics\n(Actual values - no normalization artifacts)',
             fontweight='bold', fontsize=15, pad=20)

    # Add note
    note = "This table shows ABSOLUTE values. Avoid min-max normalization with few agents\n" + \
           "as it artificially makes one agent always 100 and the other always 0."
    plt.figtext(0.5, 0.05, note, ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results/figures/absolute_metrics.png', dpi=300, bbox_inches='tight')
    print("[SAVED] results/figures/absolute_metrics.png (REPLACES radar chart)")
    plt.close()


def plot_radar_chart(results: Dict[str, AgentEvalResults]):
    """
    Figure 5: Radar/Spider chart with MEANINGFUL baseline normalization.

    IMPORTANT: Instead of min-max between agents (which makes one always 100,
    other always 0 with only 2 agents), we use THEORETICAL baselines:
    - Success rate: 0% (worst) to 100% (best)
    - Return: Normalized by theoretical maximum achievable return
    - Collision rate: Inverted (0% collision = 100 score, 100% collision = 0 score)
    - Length efficiency: Normalized by optimal path length

    This makes the radar chart meaningful even with few agents.

    Reference: Suggested improvements addressing normalization artifacts.
    """
    ensure_dirs()

    valid = {k: v for k, v in results.items() if v.available}
    if not valid:
        return

    agents = list(valid.keys())

    # Define metrics with MEANINGFUL baselines (not agent-relative)
    # These represent performance on a 0-100 scale against theoretical limits

    metrics_data = {}
    for agent_name in agents:
        res = valid[agent_name]

        # Metric 1: Success Rate (already 0-100%)
        success_score = res.success_rate * 100

        # Metric 2: Return Normalized
        # Theoretical best: +150 (task completion) per episode
        # Theoretical worst: -300 (battery depletion)
        # Normalize to 0-100 scale
        theoretical_best = 150
        theoretical_worst = -3000  # Very bad performance
        return_norm = max(0, min(100, 100 * (res.mean_return - theoretical_worst) / (theoretical_best - theoretical_worst)))

        # Metric 3: Safety Score (inverted collision rate)
        # 0% collision = 100 score, 100% collision = 0 score
        safety_score = 100 * (1 - res.collision_rate / 100)

        # Metric 4: Efficiency Score (inverted length, normalized)
        # Theoretical optimal path: ~30 steps
        # Normalize so shorter = better
        optimal_length = 30
        max_reasonable_length = 200
        efficiency_score = max(0, min(100, 100 * (max_reasonable_length - res.mean_length) / (max_reasonable_length - optimal_length)))

        # Metric 5: Consistency Score (inverted std, normalized)
        # Lower std = higher consistency
        max_std = 300  # Theoretical maximum variation
        consistency_score = max(0, min(100, 100 * (1 - res.std_return / max_std)))

        # Metric 6: IQM-based robustness
        # Similar to return but using IQM
        iqm_norm = max(0, min(100, 100 * (res.iqm_return - theoretical_worst) / (theoretical_best - theoretical_worst)))

        metrics_data[agent_name] = [
            success_score,
            return_norm,
            safety_score,
            efficiency_score,
            consistency_score,
            iqm_norm
        ]

    # Metric labels
    categories = [
        'Success\nRate',
        'Return\nQuality',
        'Safety\nScore',
        'Path\nEfficiency',
        'Consistency',
        'Robustness\n(IQM)'
    ]

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Colors for agents
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot each agent
    for idx, agent_name in enumerate(agents):
        values = metrics_data[agent_name]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2.5, label=agent_name,
                color=colors[idx % len(colors)], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')

    # Set y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.set_rlabel_position(0)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)

    # Add title
    plt.title('Performance Radar Chart\n(Normalized against theoretical baselines, not agent-relative)',
              fontweight='bold', fontsize=14, pad=30)

    # Add explanation note
    note = ("Normalization method: Each metric scaled 0-100 against theoretical limits.\n"
            "Success Rate: actual %; Safety: 100 - collision%; Efficiency: path optimality;\n"
            "Return Quality: scaled by theoretical best/worst; Consistency: inverse of variance.\n"
            "This avoids min-max artifacts where one agent is always 100 and the other 0.")

    plt.figtext(0.5, -0.05, note, ha='center', fontsize=9, style='italic', wrap=True,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('results/figures/radar_chart_baseline_normalized.png', dpi=300, bbox_inches='tight')
    print("[SAVED] results/figures/radar_chart_baseline_normalized.png")
    plt.close()


def save_raw_data(results: Dict[str, AgentEvalResults]):
    """Save per-seed, per-episode data for reproducibility."""
    ensure_dirs()

    for name, res in results.items():
        if not res.available:
            continue

        data = asdict(res)

        with open(f'results/raw_data/{name.replace(" ", "_").lower()}_raw.json', 'w') as f:
            json.dump(data, f, indent=2)

    print("\n[SAVED] Raw per-seed data to results/raw_data/*.json")


def perform_statistical_tests(results: Dict[str, AgentEvalResults]):
    """
    Perform pairwise statistical significance tests.

    Uses Mann-Whitney U test (non-parametric, appropriate for RL).
    """
    ensure_dirs()

    valid = {k: v for k, v in results.items() if v.available}
    agents = list(valid.keys())

    if len(agents) < 2:
        return

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Mann-Whitney U)")
    print("=" * 80)

    test_results = []

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1, agent2 = agents[i], agents[j]

            # Get per-seed means
            data1 = np.array(valid[agent1].per_seed_mean_returns)
            data2 = np.array(valid[agent2].per_seed_mean_returns)

            # Perform test
            test_result = mann_whitney_u_test(data1, data2, alpha=0.05)

            test_results.append({
                'Agent 1': agent1,
                'Agent 2': agent2,
                'p-value': f"{test_result['p_value']:.4f}",
                'Significant (α=0.05)': 'Yes' if test_result['significant'] else 'No',
                'Effect Size': f"{test_result['effect_size']:.3f}",
                'Interpretation': test_result['interpretation']
            })

            print(f"\n{agent1} vs {agent2}:")
            print(f"  p-value: {test_result['p_value']:.4f}")
            print(f"  Significant: {test_result['interpretation']}")
            print(f"  Effect size: {test_result['effect_size']:.3f}")

    # Save to CSV
    df = pd.DataFrame(test_results)
    df.to_csv('results/metrics/statistical_tests.csv', index=False)
    print(f"\n[SAVED] Statistical test results to results/metrics/statistical_tests.csv")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main evaluation workflow."""
    print("\n" + "=" * 80)
    print("BEST-PRACTICE RL EVALUATION")
    print("Multi-Seed, Deterministic, with Proper Statistics")
    print("=" * 80)

    # User prompts
    include_baselines = False
    if BASELINES_AVAILABLE:
        choice = ask_user_choice(
            "\nInclude BASELINE agents (Random, Greedy, Rule-Based)?",
            ["Yes", "No - Only mandatory"]
        )
        include_baselines = (choice == "Yes")

    include_advanced = False
    if ADVANCED_AVAILABLE:
        choice = ask_user_choice(
            "\nInclude ADVANCED agents (Double DQN, Dueling DQN)?",
            ["Yes", "No - Only mandatory"]
        )
        include_advanced = (choice == "Yes")

    # Episodes per seed
    choice = ask_user_choice(
        "\nEpisodes PER SEED:",
        ["Quick (30)", "Standard (50)", "Thorough (100)"]
    )
    n_episodes = {"Quick (30)": 30, "Standard (50)": 50, "Thorough (100)": 100}[choice]

    # Number of seeds
    choice = ask_user_choice(
        "\nNumber of INDEPENDENT SEEDS:",
        ["Quick (3)", "Recommended (5)", "Rigorous (10)"]
    )
    seeds = {"Quick (3)": [0,1,2], "Recommended (5)": [0,1,2,3,4], "Rigorous (10)": list(range(10))}[choice]

    # Deterministic evaluation
    choice = ask_user_choice(
        "\nUse DETERMINISTIC policy (epsilon=0)?",
        ["Yes (recommended - simulates deployment)", "No (keep exploration)"]
    )
    deterministic = (choice == "Yes (recommended - simulates deployment)")

    # Build agent list
    agents = ["Q-Learning", "Simple DQN"]
    if include_baselines and BASELINES_AVAILABLE:
        agents = ["Random", "Greedy", "Rule-Based"] + agents
    if include_advanced and ADVANCED_AVAILABLE:
        agents += ["Double DQN", "Dueling DQN"]

    print("\n" + "=" * 80)
    print("EVALUATION PLAN")
    print("=" * 80)
    print(f"Agents: {', '.join(agents)}")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {n_episodes}")
    print(f"Total episodes per agent: {n_episodes * len(seeds)}")
    print(f"Deterministic policy: {deterministic}")
    print("=" * 80)

    # Run evaluation
    results = {}
    for idx, agent in enumerate(agents, 1):
        print(f"\n[{idx}/{len(agents)}]")
        results[agent] = evaluate_agent_multi_seed(agent, n_episodes, seeds, deterministic)

    # Generate outputs
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)

    generate_results_table(results)
    plot_performance_comparison(results)
    plot_robust_statistics(results)
    plot_statistical_significance(results)
    plot_absolute_metrics_table(results)
    plot_radar_chart(results)  # Added: baseline-normalized radar chart
    save_raw_data(results)
    perform_statistical_tests(results)

    # Save metadata
    metadata = {
        'evaluation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'seeds': seeds,
        'n_episodes_per_seed': n_episodes,
        'deterministic': deterministic,
        'agents_evaluated': [k for k, v in results.items() if v.available],
    }
    with open('results/metrics/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)
    print("\nGenerated outputs:")
    print("  📊 results/metrics/evaluation_results.csv")
    print("  📈 results/figures/performance_comparison.png")
    print("  📈 results/figures/robust_statistics.png")
    print("  📈 results/figures/statistical_significance.png")
    print("  📈 results/figures/absolute_metrics.png")
    print("  📈 results/figures/radar_chart_baseline_normalized.png (improved normalization)")
    print("  📁 results/raw_data/*.json (reproducibility)")
    print("  📊 results/metrics/statistical_tests.csv")
    print("\n💡 For your report:")
    print(f"  • Multi-seed evaluation ({len(seeds)} seeds) for statistical validity")
    print(f"  • {n_episodes * len(seeds)} total episodes per agent")
    print(f"  • Bootstrap CIs for returns, Wilson CIs for success rates")
    print(f"  • IQM for outlier-robust comparison")
    print(f"  • Deterministic evaluation (ε=0): {'Yes' if deterministic else 'No'}")
    print(f"  • Mann-Whitney U tests for significance")
    print("=" * 80)


if __name__ == '__main__':
    main()
