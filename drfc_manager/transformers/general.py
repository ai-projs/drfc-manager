from typing import Any
from gloe import partial_transformer, condition, transformer

from drfc_manager.transformers.exceptions.base import BaseExceptionTransformers
from drfc_manager.config import settings
from drfc_manager.utils.minio.storage_manager import MinioStorageManager
from drfc_manager.utils.logging import logger

sagemaker_temp_dir = '/tmp/sagemaker'
work_directory = '/tmp/teste'
storage_manager = MinioStorageManager(settings)


@partial_transformer
def echo(_, data: Any, message: str) -> Any:
    """Prints a message and passes the input data through."""
    print(message)
    logger.info(message)
    return data

def log_and_passthrough(message: str):
    """Factory function that creates a transformer to log a message and pass data."""
    @transformer
    def _log(data: Any) -> Any:
        print(message)
        logger.info(message)
        return data
    _log.name = f"log: {message[:30]}..." # Optional: give the transformer a name
    return _log

@transformer
def passthrough(data: Any) -> Any:
    """A transformer that simply passes data through unchanged."""
    return data

@condition
def forward_condition(_condition: bool):
    return _condition


@partial_transformer
def copy_object(_, source_object_name: str, dest_object_name: str):
    """Copies an object within the S3 bucket using StorageManager."""
    try:
        storage_manager.copy_object(source_object_name, dest_object_name)
    except Exception as e:
        raise BaseExceptionTransformers(f"Failed to copy S3 object from {source_object_name} to {dest_object_name}", e)

@partial_transformer
def check_if_model_exists_transformer(_, model_name: str, overwrite: bool) -> bool:
    """Checks if model prefix exists and returns True if pipeline should stop."""
    prefix = f"{model_name}/"
    exists = storage_manager.object_exists(f"{prefix}model.pb")

    if exists and not overwrite:
        logger.info(f"Model prefix {prefix} exists and overwrite is False.")
        return True
    elif exists and overwrite:
        logger.info(f"Model prefix {prefix} exists but overwrite is True. Proceeding (Overwrite logic TBD).")
        return False
    else:
        logger.info(f"Model prefix {prefix} does not exist. Proceeding.")
        return False