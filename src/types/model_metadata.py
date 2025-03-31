from dataclasses import dataclass, field
from typing import List, Union
from enum import Enum


class ActionSpaceType(Enum):
    """Type of action space that will be used by the model.

    - `CONTINUOUS`: Represents continuous values on the action space, which are specified by a range.
    - `DISCRETE`: Represents discrete values on the action space, which represent specific points.
    """
    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'


class NeuralNetwork(Enum):
    """Type of Neural Network used for training the algorithm.

    - `DEEP_CONVOLUTIONAL_NETWORK_SHALLOW`: This network has no fully connected layer; the input embedder nodes
      connect directly to the action nodes.
    - `DEEP_CONVOLUTIONAL_NETWORK`: This network includes a single fully connected layer of 64 nodes.
    - `DEEP_CONVOLUTIONAL_NETWORK_DEEP`: This network includes two fully connected layers of 512 nodes each.
    """
    DEEP_CONVOLUTIONAL_NETWORK_SHALLOW = 'DEEP_CONVOLUTIONAL_NETWORK_SHALLOW'
    DEEP_CONVOLUTIONAL_NETWORK = 'DEEP_CONVOLUTIONAL_NETWORK'
    DEEP_CONVOLUTIONAL_NETWORK_DEEP = 'DEEP_CONVOLUTIONAL_NETWORK_DEEP'


class TrainingAlgorithm(Enum):
    """Training algorithms for the model.

    - `SAC`: Stochastic Actor-Critic.
    - `PPO`: Proximal Policy Optimization.
    """
    SAC = 'sac'
    PPO = 'ppo'


class Sensor(Enum):
    """Types of sensors used in the model.

    - `FRONT_FACING_CAMERA`: Front-facing camera sensor.
    - `STEREO_CAMERAS`: Stereo camera sensors.
    - `LIDAR`: Light Detection and Ranging sensor.
    """
    FRONT_FACING_CAMERA = 'FRONT_FACING_CAMERA'
    STEREO_CAMERAS = 'STEREO_CAMERAS'
    LIDAR = 'LIDAR'


@dataclass
class SteeringAngle:
    """Represents the range of steering angles."""
    high: float
    low: float


@dataclass
class Speed:
    """Represents the range of speeds."""
    high: float
    low: float


@dataclass
class ContinuousActionSpace:
    """Defines the continuous action space.

    Attributes:
    - `steering_angle`: Range of steering angles.
    - `speed`: Range of speeds.
    """
    steering_angle: SteeringAngle
    speed: Speed


class DiscreteActionSpace(Enum):
    """Defines the discrete action space."""
    steering_angle = -30.0
    speed = 0.6


@dataclass
class ModelMetadata:
    """Metadata for the model.

    Attributes:
    - `action_space_type`: Type of action space used by the model (default: `CONTINUOUS`).
    - `action_space`: Action space configuration (default: continuous action space).
    - `version`: Version of the model (default: 5).
    - `training_algorithm`: Training algorithm used (default: `PPO`).
    - `neural_network`: Type of neural network used for training (default: `DEEP_CONVOLUTIONAL_NETWORK_SHALLOW`).
    - `sensor`: List of sensors used by the model (default: `[FRONT_FACING_CAMERA]`).
    """
    action_space_type: ActionSpaceType = ActionSpaceType.CONTINUOUS
    action_space: Union[ContinuousActionSpace, List[DiscreteActionSpace]] = field(
        default_factory=lambda: ContinuousActionSpace(
            steering_angle=SteeringAngle(high=30.0, low=-30.0),
            speed=Speed(high=4.0, low=1.0))
    )
    version: int = 5
    training_algorithm: TrainingAlgorithm = TrainingAlgorithm.PPO
    neural_network: NeuralNetwork = NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW
    sensor: List[Sensor] = field(default_factory=lambda: [Sensor.FRONT_FACING_CAMERA])
