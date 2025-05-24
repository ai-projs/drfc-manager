from dataclasses import dataclass
from enum import Enum


@dataclass
class ConfigEnvs(str, Enum):
    run = 'run'
    system = 'system'