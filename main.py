import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

# Import from src modules
from src.environment import WarehouseAMREnv
from src.agents import QLearningAgent, DQNAgent
from src.training import train_q_learning, train_dqn

def main():
    """Main training script with GPU support"""

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

    print("=" * 60)
    print("WAREHOUSE AMR RL PROJECT")
    print("=" * 60)

    # Display GPU/CPU info
    if torch.cuda.is_available():
        print(f"\n[GPU INFO]")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  PyTorch Version: {torch.__version__}")
        use_gpu = True
    else:
        print(f"\n[CPU MODE] No GPU detected, training will be slower")
        use_gpu = False

    # Create environment
    env = WarehouseAMREnv(config={'n_humans': 3, 'n_forklifts': 1, 'n_other_robots': 2})

    # Train Q-Learning
    print("\n[1/2] Training Tabular Q-Learning...")
    q_agent = QLearningAgent()
    q_rewards, q_lengths = train_q_learning(env, q_agent, n_episodes=2000)

    # Train DQN with GPU optimization
    print("\n[2/2] Training Deep Q-Network...")
    # Enable mixed precision for faster training on GPU (requires Ampere GPU or newer)
    use_amp = use_gpu and torch.cuda.get_device_capability()[0] >= 7  # Tensor Cores available
    dqn_agent = DQNAgent(use_amp=use_amp)
    dqn_rewards, dqn_lengths = train_dqn(env, dqn_agent, n_episodes=500)

    # Plot results
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