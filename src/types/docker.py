from enum import Enum

class ComposeFileType(Enum):
    """Types of compose files used in DeepRacer"""
    TRAINING = "training"
    TRAINING_SWARM = "training-swarm"

    EVAL = "eval"
    EVAL_SWARM = "eval-swarm"

    KEYS = "keys"
    ENDPOINT = "endpoint"
    MOUNT = "mount"
    AWS = "aws"
    XORG = "local-xorg"
    XORG_WSL = "local-xorg-wsl"

    METRICS = "metrics"

    ROBOMAKER_MULTI = "robomaker-multi"
    ROBOMAKER_SCRIPTS = "robomaker-scripts" 
    SIMAPP = "simapp"                   
