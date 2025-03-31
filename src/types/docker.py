from enum import Enum
from dataclasses import dataclass


@dataclass
class DockerImages(str, Enum):
    training = 'training'
    endpoint = 'endpoint'
    keys = 'keys'
