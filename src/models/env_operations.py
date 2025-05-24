# src/config/env_operations.py
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class EnvConfig:
    """Immutable environment configuration."""
    pretrained_prefix: str
    model_prefix: str
    pretrained: bool = True

def create_env_config(source_model: str, target_model: str) -> EnvConfig:
    """Pure function to create environment configuration."""
    return EnvConfig(
        pretrained_prefix=source_model,
        model_prefix=target_model,
        pretrained=True
    )

def apply_env_config(config: EnvConfig) -> None:
    """Pure function to apply environment configuration."""
    os.environ['DR_LOCAL_S3_PRETRAINED_PREFIX'] = config.pretrained_prefix
    os.environ['DR_LOCAL_S3_PRETRAINED'] = str(config.pretrained)
    os.environ['DR_LOCAL_S3_MODEL_PREFIX'] = config.model_prefix