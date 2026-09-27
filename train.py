"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        UNIFIED TRAINING SCRIPT                               ║
║                                                                              ║
║  This script provides an interactive menu to train RL algorithms:           ║
║                                                                              ║
║  ✅ MANDATORY (Required for project):                                       ║
║     1. Tabular Q-Learning                                                   ║
║     2. Simple DQN (single network)                                          ║
║                                                                              ║
║  🔬 OPTIONAL (Advanced - NOT required):                                     ║
║     3. Double DQN                                                           ║
║     4. Dueling DQN                                                          ║
║                                                                              ║
║  Usage: python train.py                                                     ║
║  Follow interactive prompts to select algorithms and hardware.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from typing import List

# ==============================================================================
# Import environment
# ==============================================================================
from src.environment import WarehouseAMREnv

# ==============================================================================
# MANDATORY IMPORTS (Required)
# ==============================================================================
from src.agents import QLearningAgent, DQNAgent
from src.training import train_q_learning, train_dqn

# ==============================================================================
# OPTIONAL IMPORTS (Advanced - NOT required)
# ==============================================================================
try:
    from src.agents import DoubleDQNAgent, DuelingDQNAgent
    from src.training import train_double_dqn, train_dueling_dqn
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("[INFO] Advanced agents not available")


def ask_user_choice(prompt: str, options: List[str]) -> str:
    """Ask user to select from options"""
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
        print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")


def ask_multiple_choice(prompt: str, options: List[str]) -> List[str]:
    """Ask user to select multiple options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print("  0. Done selecting")

    selected = []
    while True:
        choice = input(f"\nEnter choice (0 when done): ").strip()
        try:
            idx = int(choice)
            if idx == 0:
                break
            if 1 <= idx <= len(options):
                option = options[idx - 1]
                if option not in selected:
                    selected.append(option)
                    print(f"  ✓ Added: {option}")
                else:
                    print(f"  Already selected: {option}")
            else:
                print(f"Invalid choice. Enter 0-{len(options)}.")
        except ValueError:
            print(f"Invalid input. Enter a number 0-{len(options)}.")

    return selected


def main():
    """
    Main training function with interactive prompts
    """

    print("=" * 80)
    print("WAREHOUSE AMR RL PROJECT - TRAINING SCRIPT")
    print("=" * 80)

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

    # ==============================================================================
    # STEP 1: Select algorithms to train
    # ==============================================================================
    print("\n" + "="*80)
    print("STEP 1: SELECT ALGORITHMS TO TRAIN")
    print("="*80)

    print("\n✅ MANDATORY algorithms (required for project):")
    print("  - Tabular Q-Learning")
    print("  - Simple DQN (single network)")

    if ADVANCED_AVAILABLE:
        print("\n🔬 OPTIONAL algorithms (advanced - NOT required):")
        print("  - Double DQN")
        print("  - Dueling DQN")

    available_algorithms = ["Q-Learning", "Simple DQN"]
    if ADVANCED_AVAILABLE:
        available_algorithms.extend(["Double DQN", "Dueling DQN"])

    selected_algorithms = ask_multiple_choice(
        "\nSelect algorithms to train (select multiple, enter 0 when done):",
        available_algorithms
    )

    if len(selected_algorithms) == 0:
        print("\n[ERROR] No algorithms selected. Exiting.")
        return

    print(f"\n✓ Selected algorithms: {', '.join(selected_algorithms)}")

    # ==============================================================================
    # STEP 2: Device selection (GPU/CPU)
    # ==============================================================================
    print("\n" + "="*80)
    print("STEP 2: HARDWARE SELECTION")
    print("="*80)

    use_gpu = False
    if torch.cuda.is_available():
        print(f"\n[GPU DETECTED]")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  PyTorch Version: {torch.__version__}")

        choice = ask_user_choice(
            "\nSelect device for training:",
            ["GPU (Recommended for faster training)", "CPU (Slower but works without GPU)"]
        )
        use_gpu = (choice == "GPU (Recommended for faster training)")

        if not use_gpu:
            # Force CPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.cuda.is_available = lambda: False
            print("\n[CPU MODE] CPU selected for training")
        else:
            print("\n[GPU MODE] GPU will be used for training")
    else:
        print("\n[NO GPU DETECTED]")
        print("  Training will use CPU")
        use_gpu = False

    # ==============================================================================
    # STEP 3: Training configuration
    # ==============================================================================
    print("\n" + "="*80)
    print("STEP 3: TRAINING CONFIGURATION")
    print("="*80)

    choice = ask_user_choice(
        "\nSelect training duration:",
        [
            "Quick (500 episodes) - For testing",
            "Standard (1000 episodes) - Recommended",
            "Long (3000 episodes) - Better convergence"
        ]
    )

    episode_config = {
        "Quick (500 episodes) - For testing": 500,
        "Standard (1000 episodes) - Recommended": 1000,
        "Long (3000 episodes) - Better convergence": 3000
    }
    n_episodes = episode_config[choice]

    # ==============================================================================
    # STEP 4: Start training
    # ==============================================================================
    print("\n" + "="*80)
    print(f"TRAINING {len(selected_algorithms)} ALGORITHM(S)")
    print(f"Episodes per algorithm: {n_episodes}")
    print(f"Device: {'GPU' if use_gpu else 'CPU'}")
    print("="*80)

    # Create environment
    env = WarehouseAMREnv(config={'n_humans': 3, 'n_forklifts': 1, 'n_other_robots': 2})

    # GPU optimization settings
    if use_gpu:
        # Enable mixed precision for Ampere GPUs (RTX 3000+, RTX 4000+)
        use_amp = torch.cuda.get_device_capability()[0] >= 7
        batch_size = 128
        buffer_size = 100000
        print(f"\n[GPU OPTIMIZATION]")
        print(f"  Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled (GPU too old)'}")
        print(f"  Batch Size: {batch_size} (optimized for GPU)")
        print(f"  Buffer Size: {buffer_size}")
    else:
        use_amp = False
        batch_size = 32
        buffer_size = 50000
        print(f"\n[CPU OPTIMIZATION]")
        print(f"  Batch Size: {batch_size} (CPU mode)")
        print(f"  Buffer Size: {buffer_size}")

    training_results = {}

    # Train each selected algorithm
    for idx, algorithm in enumerate(selected_algorithms, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(selected_algorithms)}] TRAINING: {algorithm}")
        print(f"{'='*80}")

        if algorithm == "Q-Learning":
            agent = QLearningAgent()
            rewards, lengths = train_q_learning(env, agent, n_episodes=n_episodes,
                                               save_path='models/q_learning_best.pkl')
            training_results['Q-Learning'] = {'rewards': rewards, 'lengths': lengths}

        elif algorithm == "Simple DQN":
            agent = DQNAgent(use_amp=use_amp, batch_size=batch_size, buffer_size=buffer_size)
            rewards, lengths = train_dqn(env, agent, n_episodes=n_episodes,
                                        save_path='models/dqn_best.pth')
            training_results['Simple DQN'] = {'rewards': rewards, 'lengths': lengths}

        elif algorithm == "Double DQN" and ADVANCED_AVAILABLE:
            agent = DoubleDQNAgent(use_amp=use_amp, batch_size=batch_size, buffer_size=buffer_size)
            rewards, lengths = train_double_dqn(env, agent, n_episodes=n_episodes,
                                               save_path='models/double_dqn_best.pth')
            training_results['Double DQN'] = {'rewards': rewards, 'lengths': lengths}

        elif algorithm == "Dueling DQN" and ADVANCED_AVAILABLE:
            agent = DuelingDQNAgent(use_amp=use_amp, batch_size=batch_size, buffer_size=buffer_size)
            rewards, lengths = train_dueling_dqn(env, agent, n_episodes=n_episodes,
                                                save_path='models/dueling_dqn_best.pth')
            training_results['Dueling DQN'] = {'rewards': rewards, 'lengths': lengths}

    # ==============================================================================
    # STEP 5: Generate training comparison plots
    # ==============================================================================
    if len(training_results) > 0:
        print(f"\n{'='*80}")
        print("GENERATING TRAINING PLOTS")
        print(f"{'='*80}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Training Comparison', fontsize=16, fontweight='bold')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Plot rewards
        for (name, data), color in zip(training_results.items(), colors):
            rewards = data['rewards']
            ax1.plot(rewards, alpha=0.3, color=color)
            smoothed = pd.Series(rewards).rolling(100, min_periods=1).mean()
            ax1.plot(smoothed, label=name, linewidth=2, color=color)

        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Cumulative Reward', fontsize=12)
        ax1.set_title('Training Rewards', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot episode lengths
        for (name, data), color in zip(training_results.items(), colors):
            lengths = data['lengths']
            smoothed = pd.Series(lengths).rolling(100, min_periods=1).mean()
            ax2.plot(smoothed, label=name, linewidth=2, color=color)

        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Episode Length (Steps)', fontsize=12)
        ax2.set_title('Episode Lengths', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/figures/training_comparison.png', dpi=300, bbox_inches='tight')
        print("[SAVED] results/figures/training_comparison.png")
        plt.close()

    # ==============================================================================
    # STEP 6: Summary
    # ==============================================================================
    print(f"\n{'='*80}")
    print("✓ TRAINING COMPLETE")
    print(f"{'='*80}")

    print(f"\nTrained algorithms ({len(selected_algorithms)}):")
    for algorithm in selected_algorithms:
        print(f"  ✓ {algorithm}")

    print(f"\nModels saved to:")
    if "Q-Learning" in selected_algorithms:
        print("  - models/q_learning_best.pkl")
    if "Simple DQN" in selected_algorithms:
        print("  - models/dqn_best.pth")
    if "Double DQN" in selected_algorithms:
        print("  - models/double_dqn_best.pth")
    if "Dueling DQN" in selected_algorithms:
        print("  - models/dueling_dqn_best.pth")

    print(f"\nTraining plot saved to:")
    print("  - results/figures/training_comparison.png")

    print(f"\nNext steps:")
    print("  1. Run evaluation: python src/evaluation/evaluate_all.py")
    print("  2. Review training plots: results/figures/training_comparison.png")
    print("  3. Check models directory: models/")
    print()


if __name__ == '__main__':
    main()
