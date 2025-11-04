import numpy as np
import torch

def train_dqn(env, agent, n_episodes=5000, save_path='models/dqn_best.pth'):
    """Train DQN Agent with GPU monitoring"""

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    best_avg_reward = -float('inf')

    print("Training DQN Agent...")
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