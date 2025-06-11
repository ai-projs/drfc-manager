import os
import time
from typing import Callable, Dict, Optional
from contextlib import contextmanager

from gloe import If, transformer, condition
from drfc_manager.transformers.training import (
    create_sagemaker_temp_files,
    check_if_metadata_is_available,
    upload_hyperparameters,
    upload_metadata,
    upload_reward_function,
    upload_training_params_file,
    start_training,
    expose_config_envs_from_dataclass,
    check_training_logs_transformer,
    create_training_data_directories,
)
from drfc_manager.transformers.general import (
    check_if_model_exists_transformer,
    copy_object,
    log_check_performed,
    log_model_exists,
    log_data_uploaded,
    log_reward_function_copied,
    log_training_config_uploaded,
    log_training_started,
    log_stack_started,
    log_skipping_check,
)
from drfc_manager.types.hyperparameters import HyperParameters
from drfc_manager.types.model_metadata import ModelMetadata
from drfc_manager.config_env import settings
from drfc_manager.utils.docker.docker_manager import DockerManager
from drfc_manager.utils.docker.exceptions import DockerError, ComposeError, SwarmError
from drfc_manager.utils.minio.storage_manager import MinioStorageManager
from drfc_manager.models.model_operations import (
    create_clone_config,
    generate_model_name,
)
from drfc_manager.models.storage_operations import (
    check_model_exists,
    delete_model,
    upload_model_data,
)
from drfc_manager.models.env_operations import create_env_config, apply_env_config
from drfc_manager.models.data_extraction import extract_model_data
from drfc_manager.utils.logging import logger, setup_logging
from drfc_manager.helpers.files_manager import create_folder, delete_files_on_folder
from drfc_manager.transformers.exceptions.base import BaseExceptionTransformers

storage_manager = MinioStorageManager(settings)
docker_manager = DockerManager(settings)

sagemaker_temp_dir = os.path.expanduser("~/sagemaker_temp")
work_directory = os.path.expanduser("~/dr_work")


@transformer
def sleep_15_seconds(data):
    """Sleep for 15 seconds before checking logs"""
    time.sleep(15)
    return data


@contextmanager
def managed_docker_stack(docker_manager: DockerManager):
    """Context manager for handling Docker stack lifecycle.

    Args:
        docker_manager: DockerManager instance

    Yields:
        None

    Raises:
        DockerError: If stack operations fail
    """
    try:
        yield
    except (ComposeError, SwarmError) as e:
        logger.error(
            "docker_stack_operation_failed",
            error=str(e),
            command=e.command,
            stderr=e.stderr,
        )
        raise
    except DockerError as e:
        logger.error(
            "docker_operation_failed", error=str(e), command=e.command, stderr=e.stderr
        )
        raise
    finally:
        try:
            docker_manager.cleanup_previous_run(prune_system=False)
        except DockerError as e:
            logger.error(
                "cleanup_failed", error=str(e), command=e.command, stderr=e.stderr
            )


@transformer
def _validate_training_input(data: Dict) -> Dict:
    """Validate the training input data."""
    if not data["model_name"] or not data["model_name"].strip():
        raise ValueError("Model name cannot be empty")
    if not data["hyperparameters"]:
        raise ValueError("Hyperparameters cannot be empty")
    if not data["model_metadata"]:
        raise ValueError("Model metadata cannot be empty")
    return data


@transformer
def prepare_training_environment(data):
    """Prepare the training environment by creating necessary directories and files."""
    try:
        # Create sagemaker_temp directory
        create_folder(sagemaker_temp_dir, 0o777)

        # Create work directory
        create_folder(work_directory, 0o777)

        # Create training data directories
        data = create_training_data_directories(data)

        logger.info("Training environment prepared successfully")
    except Exception as e:
        raise BaseExceptionTransformers(
            f"Failed to prepare training environment: {e}", e
        )
    return data


@condition
def should_check_logs(data) -> bool:
    """Determine if logs should be checked after stack start."""
    return settings.deepracer.check_logs_after_start


def train_pipeline(
    model_name: str,
    hyperparameters: HyperParameters,
    model_metadata: ModelMetadata,
    reward_function: Callable[[Dict], float],
    overwrite: bool = False,
    check_logs_after_start: bool = False,
    reward_function_code: Optional[str] = None,
    quiet: bool = True,
):
    """
    Orchestrates the training pipeline (using original structure).

    Args:
        model_name (str): Name of the model to be trained.
        hyperparameters (HyperParameters): Training hyperparameters.
        model_metadata (ModelMetadata): Model metadata.
        reward_function (Callable[[Dict], float]): Reward function.
        overwrite (bool, optional): Overwrite existing model data. Defaults to False.
        check_logs_after_start (bool, optional): Check logs after stack start. Defaults to False.
        reward_function_code (Optional[str], optional): Reward function code. Defaults to None.
        quiet (bool, optional): If True, suppress console output. Defaults to True.

    Raises:
        DockerError: If Docker operations fail
        ValueError: If model data is invalid
    """
    _custom_files_folder = settings.minio.custom_files_folder
    _bucket_name = settings.minio.bucket_name

    setup_logging(run_id=settings.deepracer.run_id, model_name=model_name, quiet=quiet)

    reward_function_obj_location_custom = f"{_custom_files_folder}/reward_function.py"
    reward_function_obj_location_model = f"{model_name}/reward_function.py"

    # Prepare the input data dict
    data = {
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "model_metadata": model_metadata,
        "reward_function": reward_function,
        "overwrite": overwrite,
        "check_logs_after_start": check_logs_after_start,
        "reward_function_code": reward_function_code,
        "quiet": quiet,
        "bucket_name": _bucket_name,
        "custom_files_folder": _custom_files_folder,
        "reward_function_obj_location_custom": reward_function_obj_location_custom,
        "reward_function_obj_location_model": reward_function_obj_location_model,
    }

    # Compose the pipeline
    model_data_upload = (
        upload_hyperparameters >> upload_metadata >> upload_reward_function
    )

    check_logs_pipeline = (
        sleep_15_seconds >> check_training_logs_transformer >> log_check_performed
    )

    training_pipeline = (
        _validate_training_input
        >> prepare_training_environment
        >> create_sagemaker_temp_files
        >> check_if_metadata_is_available
        >> create_training_data_directories
        >> check_if_model_exists_transformer
        >> If(lambda d: not d["overwrite"] and d["model_exists"])
        .Then(log_model_exists)
        .Else(
            model_data_upload
            >> log_data_uploaded
            >> copy_object
            >> log_reward_function_copied
            >> upload_training_params_file
            >> log_training_config_uploaded
            >> expose_config_envs_from_dataclass
            >> log_training_started
            >> start_training
            >> log_stack_started
            >> If(lambda d: d["check_logs_after_start"])
            .Then(check_logs_pipeline)
            .Else(log_skipping_check)
        )
    )

    logger.info(
        f"Starting training pipeline for model: {model_name}, Run ID: {settings.deepracer.run_id}"
    )
    result = training_pipeline(data)
    logger.info("Training pipeline finished.")
    return result


def stop_training_pipeline():
    """
    Stops the currently running DeepRacer training Docker stack.

    Uses the run_id from the current settings (or DR_RUN_ID env var)
    to identify the correct Docker Compose project.

    Raises:
        DockerError: If Docker operations fail
    """
    logger.info("Attempting to stop training stack...")
    try:
        current_run_id = int(os.getenv("DR_RUN_ID", settings.deepracer.run_id))
        settings.deepracer.run_id = current_run_id
        logger.info(f"Targeting Run ID: {current_run_id}")

        docker_manager = DockerManager(settings)

        with managed_docker_stack(docker_manager):
            logger.info("Training stack stopped successfully.")

    except (ComposeError, SwarmError) as e:
        logger.error(
            "docker_stack_operation_failed",
            error=str(e),
            command=e.command,
            stderr=e.stderr,
        )
        raise
    except DockerError as e:
        logger.error(
            "docker_operation_failed", error=str(e), command=e.command, stderr=e.stderr
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while stopping training: {type(e).__name__} - {e}"
        )
        raise


def clone_pipeline(
    source_model_name: str,
    new_model_name: Optional[str] = None,
    delimiter: str = "-",
    wipe_target: bool = False,
    custom_hyperparameters: Optional[HyperParameters] = None,
    custom_model_metadata: Optional[ModelMetadata] = None,
    custom_reward_function: Optional[Callable[[Dict], float]] = None,
    check_logs_after_start: bool = False,
    skip_training: bool = False,
    quiet: bool = True,
) -> str:
    """Functional pipeline for cloning a model.

    Args:
        source_model_name (str): Name of the source model
        new_model_name (Optional[str], optional): Name for the new model. Defaults to None.
        delimiter (str, optional): Delimiter for model name generation. Defaults to "-".
        wipe_target (bool, optional): Whether to wipe target model if exists. Defaults to False.
        custom_hyperparameters (Optional[HyperParameters], optional): Custom hyperparameters. Defaults to None.
        custom_model_metadata (Optional[ModelMetadata], optional): Custom model metadata. Defaults to None.
        custom_reward_function (Optional[Callable[[Dict], float]], optional): Custom reward function. Defaults to None.
        check_logs_after_start (bool, optional): Whether to check logs after start. Defaults to False.
        skip_training (bool, optional): Whether to skip training. Defaults to False.
        quiet (bool, optional): Whether to suppress console output. Defaults to True.

    Returns:
        str: Name of the cloned model

    Raises:
        ValueError: If source model doesn't exist or target model exists without wipe_target
        DockerError: If Docker operations fail
    """
    config = create_clone_config(
        source_model_name,
        new_model_name,
        delimiter,
        wipe_target,
        custom_hyperparameters,
        custom_model_metadata,
        custom_reward_function,
        check_logs_after_start,
        skip_training,
    )

    target_name = generate_model_name(
        config.source_name, config.target_name, config.delimiter
    )

    if not check_model_exists(storage_manager, config.source_name):
        raise ValueError(f"Source model '{config.source_name}' does not exist")

    if check_model_exists(storage_manager, target_name):
        if not config.wipe_target:
            raise ValueError(
                f"Target model '{target_name}' exists and wipe_target=False"
            )
        delete_model(storage_manager, target_name)

    env_config = create_env_config(config.source_name, target_name)
    apply_env_config(env_config)

    model_data = extract_model_data(
        storage_manager,
        config.source_name,
        config.custom_hyperparameters,
        config.custom_metadata,
        config.custom_reward_function,
    )

    upload_model_data(storage_manager, model_data)

    if not config.skip_training:
        try:
            train_pipeline(
                model_name=target_name,
                hyperparameters=model_data.hyperparameters,
                model_metadata=model_data.metadata,
                reward_function=model_data.reward_function,
                overwrite=True,
                check_logs_after_start=config.check_logs,
                reward_function_code=model_data.reward_code,
                quiet=quiet,
            )
        except DockerError as e:
            logger.error(
                "training_failed", error=str(e), command=e.command, stderr=e.stderr
            )
            raise

    return target_name


@transformer
def start_training_stack(data):
    """Start the training stack using docker-compose."""
    try:
        # Get the compose files
        compose_files, needs_multiworker = docker_manager._prepare_compose_files()

        # Start the stack
        docker_manager.start_stack(compose_files)

        logger.info("Training stack started successfully")
        return data
    except Exception as e:
        raise BaseExceptionTransformers(f"Failed to start training stack: {e}", e)


@transformer
def stop_training_stack(data):
    """Stop the training stack."""
    try:
        docker_manager.stop_stack()
        logger.info("Training stack stopped successfully")
    except Exception as e:
        raise BaseExceptionTransformers(f"Failed to stop training stack: {e}", e)
    return data


@transformer
def cleanup_training_environment(data):
    """Clean up the training environment."""
    try:
        # Clean up sagemaker_temp directory
        delete_files_on_folder(sagemaker_temp_dir)

        # Clean up work directory
        delete_files_on_folder(work_directory)

        logger.info("Training environment cleaned up successfully")
    except Exception as e:
        raise BaseExceptionTransformers(
            f"Failed to clean up training environment: {e}", e
        )
    return data
