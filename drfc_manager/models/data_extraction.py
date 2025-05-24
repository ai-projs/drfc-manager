# src/models/data_extraction.py
from typing import Optional, Tuple, Callable, Dict
from drfc_manager.types.hyperparameters import HyperParameters
from drfc_manager.types.model_metadata import ModelMetadata
from drfc_manager.models.model_operations import ModelData
from drfc_manager.utils.minio.storage_client import StorageClient
from drfc_manager.utils.logging import logger

def extract_model_data(
    storage_client: StorageClient,
    source_model: str,
    custom_hyperparameters: Optional[HyperParameters] = None,
    custom_metadata: Optional[ModelMetadata] = None,
    custom_reward_function: Optional[Callable[[Dict], float]] = None
) -> ModelData:
    """Pure function to extract model data."""
    hyperparameters = extract_hyperparameters(storage_client, source_model, custom_hyperparameters)
    
    metadata = extract_metadata(storage_client, source_model, custom_metadata)
    
    reward_function, reward_code = extract_reward_function(
        storage_client, source_model, custom_reward_function
    )
    
    return ModelData(
        name=source_model,
        hyperparameters=hyperparameters,
        metadata=metadata,
        reward_function=reward_function,
        reward_code=reward_code
    )

def extract_hyperparameters(
    storage_client: StorageClient,
    source_model: str,
    custom_hyperparameters: Optional[HyperParameters]
) -> HyperParameters:
    """Pure function to extract hyperparameters."""
    if custom_hyperparameters:
        return custom_hyperparameters
        
    try:
        hyperparams_json = storage_client.download_json(
            f"{storage_client.config.custom_files_folder}/hyperparameters.json"
        )
        return HyperParameters(**hyperparams_json)
    except Exception:
        return HyperParameters()

def extract_metadata(
    storage_client: StorageClient,
    source_model: str,
    custom_metadata: Optional[ModelMetadata]
) -> ModelMetadata:
    """Pure function to extract model metadata."""
    if custom_metadata:
        return custom_metadata
        
    try:
        metadata_json = storage_client.download_json(
            f"{storage_client.config.custom_files_folder}/model_metadata.json"
        )
        return ModelMetadata(**metadata_json)
    except Exception:
        return ModelMetadata()

def extract_reward_function(
    storage_client: StorageClient,
    source_model: str,
    custom_reward_function: Optional[Callable[[Dict], float]]
) -> Tuple[Callable[[Dict], float], Optional[str]]:
    """
    Pure function to extract reward function and its code.
    
    Returns:
        Tuple[Callable[[Dict], float], Optional[str]]: A tuple containing:
            - The reward function (callable)
            - The reward function code (string) if available, None otherwise
    """
    if custom_reward_function:
        return custom_reward_function, None
        
    try:
        reward_code = storage_client.download_py_object(f"{source_model}/reward_function.py")
        
        namespace = {}
        exec(reward_code, namespace)
        
        reward_function = namespace.get('reward_function')
        if not callable(reward_function):
            raise ValueError("reward_function is not callable")
            
        return reward_function, reward_code
        
    except Exception as e:
        logger.warning(f"Failed to load reward function from source model: {e}")
        return create_default_reward_function(), None

def create_default_reward_function() -> Callable[[Dict], float]:
    """Pure function to create a default reward function."""
    def default_reward_function(params: Dict) -> float:
        return 1.0
    return default_reward_function