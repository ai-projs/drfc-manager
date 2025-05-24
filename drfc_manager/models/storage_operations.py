# src/storage/storage_operations.py
from typing import Any
from drfc_manager.models.model_operations import ModelData
from drfc_manager.utils.minio.storage_manager import StorageError

def check_model_exists(storage_client: Any, model_name: str) -> bool:
    """Pure function to check if a model exists."""
    try:
        objects = storage_client.client.list_objects(
            storage_client.config.bucket_name,
            prefix=f"{model_name}/",
            recursive=True
        )
        return any(True for _ in objects)
    except Exception as e:
        raise StorageError(f"Error checking if model {model_name} exists: {e}")

def delete_model(storage_client: Any, model_name: str) -> None:
    """Pure function to delete a model."""
    try:
        objects = storage_client.client.list_objects(
            storage_client.config.bucket_name,
            prefix=f"{model_name}/",
            recursive=True
        )
        for obj in objects:
            storage_client.client.remove_object(
                storage_client.config.bucket_name,
                obj.object_name
            )
    except Exception as e:
        raise StorageError(f"Error deleting model {model_name}: {e}")

def upload_model_data(storage_client: Any, model_data: ModelData) -> None:
    """Pure function to upload model data."""
    try:
        storage_client.upload_hyperparameters(model_data.hyperparameters)
        storage_client.upload_model_metadata(model_data.metadata)
        
        if model_data.reward_code:
            storage_client.upload_reward_function(model_data.reward_code)
        else:
            storage_client.upload_reward_function(model_data.reward_function)
    except Exception as e:
        raise StorageError(f"Error uploading model data: {e}")