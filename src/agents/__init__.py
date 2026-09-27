"""
Agents Module

This module contains all agent implementations, clearly separated into:
- MANDATORY: Required for project (Tabular Q-Learning, Simple DQN)
- OPTIONAL: Advanced/baseline agents for comparison and extension
"""

# ==============================================================================
# MANDATORY AGENTS (Project Requirements)
# ==============================================================================

from .q_learning_agent import QLearningAgent
from .dqn_agent import DQNAgent, DQNetwork, ReplayBuffer

# ==============================================================================
# OPTIONAL AGENTS (Advanced State-of-the-Art - NOT Required)
# ==============================================================================

from .advanced_dqn import (
    DoubleDQNAgent,      # Double DQN: Reduces overestimation
    DuelingDQNAgent,     # Dueling DQN: Separate value/advantage streams
    DuelingDQNetwork     # Dueling architecture
)

# ==============================================================================
# OPTIONAL AGENTS (Baselines for Comparison - NOT Required)
# ==============================================================================

from .baseline_agents import (
    RandomAgent,         # Random action selection
    GreedyAgent,         # Always moves toward goal
    RuleBasedAgent      # Hand-coded heuristics
)

# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # MANDATORY (Required)
    'QLearningAgent',
    'DQNAgent',
    'DQNetwork',
    'ReplayBuffer',

    # OPTIONAL (Advanced - NOT Required)
    'DoubleDQNAgent',
    'DuelingDQNAgent',
    'DuelingDQNetwork',

    # OPTIONAL (Baselines - NOT Required)
    'RandomAgent',
    'GreedyAgent',
    'RuleBasedAgent'
]
