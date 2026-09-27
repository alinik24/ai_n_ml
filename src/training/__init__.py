# ==============================================================================
# MANDATORY TRAINING FUNCTIONS (Project Requirements)
# ==============================================================================
from .train_qlearning import train_q_learning
from .train_dqn import train_dqn

# ==============================================================================
# OPTIONAL TRAINING FUNCTIONS (Advanced State-of-the-Art - NOT Required)
# ==============================================================================
try:
    from .train_advanced import train_double_dqn, train_dueling_dqn
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

# Export list
if ADVANCED_AVAILABLE:
    __all__ = ['train_q_learning', 'train_dqn', 'train_double_dqn', 'train_dueling_dqn']
else:
    __all__ = ['train_q_learning', 'train_dqn']
