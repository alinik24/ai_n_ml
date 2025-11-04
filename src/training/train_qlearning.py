import numpy as np
import pickle

def train_q_learning(env, agent, n_episodes=3000, save_path='models/q_learning_best.pkl'):
    """Train Tabular Q-Learning Agent"""
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print("Training Q-Learning Agent...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        state = agent.discretize_state(obs)
        
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not done and not truncated:
            action = agent.select_action(state, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if episode_reward > 100:  # Successful episode
            success_count += 1
        
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