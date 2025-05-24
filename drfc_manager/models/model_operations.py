# src/models/model_operations.py
from dataclasses import dataclass
from typing import Optional, Callable, Dict
from src.types.hyperparameters import HyperParameters
from src.types.model_metadata import ModelMetadata

@dataclass(frozen=True)
class ModelData:
    """Immutable container for model data."""
    name: str
    hyperparameters: HyperParameters
    metadata: ModelMetadata
    reward_function: Callable[[Dict], float]
    reward_code: Optional[str] = None

@dataclass(frozen=True)
class CloneConfig:
    """Immutable configuration for cloning operations."""
    source_name: str
    target_name: Optional[str]
    delimiter: str
    wipe_target: bool
    custom_hyperparameters: Optional[HyperParameters]
    custom_metadata: Optional[ModelMetadata]
    custom_reward_function: Optional[Callable[[Dict], float]]
    check_logs: bool
    skip_training: bool

def generate_model_name(source_name: str, target_name: Optional[str], delimiter: str) -> str:
    """Pure function to generate a new model name."""
    if target_name:
        return target_name
        
    import re
    match = re.search(f"{re.escape(delimiter)}([0-9]+)$", source_name)
    if match:
        current_num = int(match.group(1))
        new_num = current_num + 1
        return re.sub(f"{re.escape(delimiter)}{current_num}$", 
                     f"{delimiter}{new_num}", source_name)
    return f"{source_name}{delimiter}1"

def create_clone_config(
    source_name: str,
    target_name: Optional[str] = None,
    delimiter: str = '-',
    wipe_target: bool = False,
    custom_hyperparameters: Optional[HyperParameters] = None,
    custom_metadata: Optional[ModelMetadata] = None,
    custom_reward_function: Optional[Callable[[Dict], float]] = None,
    check_logs: bool = False,
    skip_training: bool = False
) -> CloneConfig:
    """Pure function to create a clone configuration."""
    return CloneConfig(
        source_name=source_name,
        target_name=target_name,
        delimiter=delimiter,
        wipe_target=wipe_target,
        custom_hyperparameters=custom_hyperparameters,
        custom_metadata=custom_metadata,
        custom_reward_function=custom_reward_function,
        check_logs=check_logs,
        skip_training=skip_training
    )