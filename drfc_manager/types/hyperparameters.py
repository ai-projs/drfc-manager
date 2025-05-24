from dataclasses import dataclass
from enum import Enum


class ExplorationType(Enum):
    """
    Exploration strategies used in training algorithms.

    - `CATEGORICAL`: Used for clipped_ppo training algorithm.
    - `ADDITIVE_NOISE`: Used for sac training algorithm.
    """
    CATEGORICAL = 'categorical'
    ADDITIVE_NOISE = 'additive_noise'


class LossType(Enum):
    """
    Types of loss techniques used to minimize error.

    - `MSE`: Mean Squared Error loss.
    - `HUBER`: Huber loss.
    """
    MSE = 'mean squared error'
    HUBER = 'huber'


@dataclass
class HyperParameters:
    """
    Hyperparameters structure for DeepRacer with default values from documentation.

    Attributes:
    - `batch_size`: Batch size used in training (default: 64).
    - `beta_entropy`: Beta value for entropy regularization (default: 0.01).
    - `discount_factor`: Discount factor for future rewards (default: 0.999).
    - `e_greedy_value`: Epsilon greedy value for exploration (default: 0.05).
    - `epsilon_steps`: Number of steps for epsilon decay (default: 10000).
    - `exploration_type`: Type of exploration strategy (default: `CATEGORICAL`).
    - `loss_type`: Type of loss function used (default: `HUBER`).
    - `lr`: Learning rate (default: 0.0003).
    - `num_episodes_between_training`: Number of episodes between each training iteration (default: 40).
    - `num_epochs`: Number of epochs for training (default: 3).
    - `stack_size`: Number of frames stacked as input (default: 1).
    - `term_cond_avg_score`: Average score threshold for terminating training (default: 100000).
    - `term_cond_max_episodes`: Maximum number of episodes for terminating training (default: 100000).
    """
    batch_size: int = 64
    beta_entropy: float = 0.01
    discount_factor: float = 0.999
    e_greedy_value: float = 0.05
    epsilon_steps: int = 10000
    exploration_type: ExplorationType = ExplorationType.CATEGORICAL
    loss_type: LossType = LossType.HUBER
    lr: float = 0.0003
    num_episodes_between_training: int = 40
    num_epochs: int = 3
    stack_size: int = 1
    term_cond_avg_score: float = 100000
    term_cond_max_episodes: float = 100000
