from enum import Enum

class ComposeFileType(Enum):
    """Types of compose files used in DeepRacer"""
    TRAINING = "training"
    KEYS = "keys"
    ENDPOINT = "endpoint"
    MOUNT = "mount"
    ROBOMAKER_MULTI = "robomaker-multi"