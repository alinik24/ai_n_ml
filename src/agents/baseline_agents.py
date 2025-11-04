"""
Baseline Agents for Comparison
Used to evaluate the performance of RL agents against simple strategies
"""

import numpy as np

class RandomAgent:
    """
    Random Agent - Takes random actions
    Useful as a lower bound baseline
    """

    def __init__(self, n_actions=11):
        self.n_actions = n_actions

    def select_action(self, state, training=True):
        """Always select random action"""
        return np.random.randint(self.n_actions)

    def update(self, *args, **kwargs):
        """No learning"""
        pass

    def decay_epsilon(self):
        """No exploration decay"""
        pass


class GreedyAgent:
    """
    Greedy Agent - Always moves toward current goal
    No learning, just follows shortest path
    """

    def __init__(self, n_actions=11):
        self.n_actions = n_actions
        self.action_map = {
            0: np.array([-1, 0]),  # North
            1: np.array([1, 0]),   # South
            2: np.array([0, 1]),   # East
            3: np.array([0, -1])   # West
        }

    def select_action(self, state, training=True):
        """
        Select action that moves toward goal
        Extract robot position and goal from state (multi-channel grid)
        """
        # Find robot position (channel 0)
        robot_channel = state[:, :, 0]
        robot_pos = np.unravel_index(np.argmax(robot_channel), robot_channel.shape)

        # Find goal position (channel 2 - task path)
        goal_channel = state[:, :, 2]
        if np.any(goal_channel):
            goal_pos = np.unravel_index(np.argmax(goal_channel), goal_channel.shape)
        else:
            # No clear goal, pick random
            return np.random.randint(4)

        # Calculate direction to goal
        direction = np.array(goal_pos) - np.array(robot_pos)

        # Choose action based on largest distance component
        if abs(direction[0]) > abs(direction[1]):
            # Move vertically
            if direction[0] > 0:
                return 1  # South
            else:
                return 0  # North
        else:
            # Move horizontally
            if direction[1] > 0:
                return 2  # East
            else:
                return 3  # West

    def update(self, *args, **kwargs):
        """No learning"""
        pass

    def decay_epsilon(self):
        """No exploration decay"""
        pass


class RuleBasedAgent:
    """
    Rule-Based Agent - Follows hand-coded rules
    - Check battery first
    - Navigate to goal
    - Pickup/drop when at location
    """

    def __init__(self, n_actions=11):
        self.n_actions = n_actions
        self.last_action = 0

    def select_action(self, state, training=True):
        """
        Use rule-based logic:
        1. If battery < 30%, go to charging station
        2. If at pickup location, pickup
        3. If at delivery location, drop
        4. Otherwise, move toward goal
        """
        # Extract battery level (channel 1)
        battery_level = state[0, 0, 1] * 100  # Convert to percentage

        # Find robot position (channel 0)
        robot_channel = state[:, :, 0]
        if not np.any(robot_channel):
            return np.random.randint(4)

        robot_pos = np.unravel_index(np.argmax(robot_channel), robot_channel.shape)

        # Rule 1: Low battery - try to charge
        if battery_level < 30:
            # Try to start charging
            return 7  # Start charging action

        # Rule 2: Check if at goal location
        goal_channel = state[:, :, 2]
        if np.any(goal_channel):
            goal_pos = np.unravel_index(np.argmax(goal_channel), goal_channel.shape)

            # At goal location?
            if robot_pos == goal_pos:
                # Try pickup or drop
                if np.random.random() < 0.5:
                    return 5  # Pickup
                else:
                    return 6  # Drop

        # Rule 3: Navigate toward goal (greedy)
        if np.any(goal_channel):
            goal_pos = np.unravel_index(np.argmax(goal_channel), goal_channel.shape)
            direction = np.array(goal_pos) - np.array(robot_pos)

            # Choose action based on largest distance component
            if abs(direction[0]) > abs(direction[1]):
                if direction[0] > 0:
                    return 1  # South
                else:
                    return 0  # North
            else:
                if direction[1] > 0:
                    return 2  # East
                else:
                    return 3  # West

        # Default: random movement action
        return np.random.randint(4)

    def update(self, *args, **kwargs):
        """No learning"""
        pass

    def decay_epsilon(self):
        """No exploration decay"""
        pass


def evaluate_agent(env, agent, n_episodes=100, render=False):
    """
    Evaluate an agent on the environment

    Returns:
        dict: Performance metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    battery_depleted = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        had_collision = False

        while not done and not truncated:
            action = agent.select_action(obs, training=False)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            if info.get('collision', False):
                had_collision = True

            if render and episode == 0:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if episode_reward > 100:
            success_count += 1

        if had_collision:
            collision_count += 1

        if info.get('reason') == 'battery_depleted':
            battery_depleted += 1

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / n_episodes,
        'collision_rate': collision_count / n_episodes,
        'battery_depletion_rate': battery_depleted / n_episodes
    }
