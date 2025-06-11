from typing import Any
from gloe import transformer, condition

from drfc_manager.transformers.exceptions.base import BaseExceptionTransformers
from drfc_manager.config_env import settings
from drfc_manager.utils.minio.storage_manager import MinioStorageManager
from drfc_manager.utils.logging import logger

sagemaker_temp_dir = "/tmp/sagemaker"
work_directory = "/tmp/teste"
storage_manager = MinioStorageManager(settings)


@transformer
def echo(data):
    message = data.get("message", None)
    if message is not None:
        if callable(message):
            message = message(data)
        print(message)
        logger.info(message)
    return data


def log_and_passthrough(message: str):
    @transformer
    def _log(data: Any) -> Any:
        print(message)
        logger.info(message)
        return data

    _log.name = f"log: {message[:30]}..."  # type: ignore[attr-defined]
    return _log


@transformer
def passthrough(data: Any) -> Any:
    return data


@condition
def forward_condition(_condition: bool):
    return _condition


@transformer
def copy_object(data):
    source_object_name = data["reward_function_obj_location_custom"]
    dest_object_name = data["reward_function_obj_location_model"]
    try:
        storage_manager.copy_object(source_object_name, dest_object_name)
    except Exception as e:
        raise BaseExceptionTransformers(
            f"Failed to copy S3 object from {source_object_name} to {dest_object_name}",
            e,
        )
    return data


@transformer
def check_if_model_exists_transformer(data):
    """Check if model exists and add the result to the data."""
    model_name = data["model_name"]
    overwrite = data["overwrite"]
    prefix = f"{model_name}/"
    exists = storage_manager.object_exists(f"{prefix}model.pb")
    if exists and not overwrite:
        logger.info(f"Model prefix {prefix} exists and overwrite is False.")
        return {**data, "model_exists": True}
    elif exists and overwrite:
        logger.info(
            f"Model prefix {prefix} exists but overwrite is True. Proceeding (Overwrite logic TBD)."
        )
        return {**data, "model_exists": False}
    else:
        logger.info(f"Model prefix {prefix} does not exist. Proceeding.")
        return {**data, "model_exists": False}


@transformer
def log_message(data: Any, message: str) -> Any:
    """Transformer that logs a message and passes through the data.

    Args:
        data: The data to pass through
        message (str): The message to log
    """
    print(message)
    logger.info(message)
    return data


@transformer
def log_message_template(data: Any, template: str) -> Any:
    """Transformer that logs a message from a template and passes through the data.

    Args:
        data: The data to pass through
        template (str): The message template to log, can use data fields
    """
    message = template.format(**data)
    print(message)
    logger.info(message)
    return data


@transformer
def log_check_performed(data: Any) -> Any:
    """Log that check was performed."""
    print("Log check performed.")
    logger.info("Log check performed.")
    return data


@transformer
def log_model_exists(data: Any) -> Any:
    """Log that model exists and aborting."""
    message = f"Model prefix 's3://{data['bucket_name']}/{data['model_name']}' exists and overwrite=False. Aborting."
    print(message)
    logger.info(message)
    return data


@transformer
def log_data_uploaded(data: Any) -> Any:
    """Log that data was uploaded successfully."""
    print("Data uploaded successfully to custom files")
    logger.info("Data uploaded successfully to custom files")
    return data


@transformer
def log_reward_function_copied(data: Any) -> Any:
    """Log that reward function was copied."""
    message = f"The reward function copied successfully to models folder at {data['reward_function_obj_location_model']}"
    print(message)
    logger.info(message)
    return data


@transformer
def log_training_config_uploaded(data: Any) -> Any:
    """Log that training config was uploaded."""
    print("Upload successfully the RoboMaker training configurations")
    logger.info("Upload successfully the RoboMaker training configurations")
    return data


@transformer
def log_training_started(data: Any) -> Any:
    """Log that training has started."""
    print("Starting model training")
    logger.info("Starting model training")
    return data


@transformer
def log_stack_started(data: Any) -> Any:
    """Log that Docker stack has started."""
    print("Docker stack started.")
    logger.info("Docker stack started.")
    return data


@transformer
def log_skipping_check(data: Any) -> Any:
    """Log that log check is being skipped."""
    print("Skipping log check.")
    logger.info("Skipping log check.")
    return data
