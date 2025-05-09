import os
import time
from typing import Callable, Dict

from gloe import If, transformer
from gloe.utils import forward

from src.transformers.training import (
    create_sagemaker_temp_files, check_if_metadata_is_available,
    upload_hyperparameters, upload_metadata, upload_reward_function,
    upload_training_params_file, start_training, expose_config_envs_from_dataclass,
    check_training_logs_transformer
)
from src.transformers.general import check_if_model_exists_transformer, copy_object, echo, forward_condition, passthrough
from src.types.hyperparameters import HyperParameters
from src.types.model_metadata import ModelMetadata
from src.config import settings
from src.utils.docker.docker_manager import DockerManager, DockerError
from src.utils.minio.storage_manager import MinioStorageManager

storage_manager = MinioStorageManager(settings)

@transformer
def sleep_15_seconds(_):
    """Sleep for 15 seconds before checking logs"""
    time.sleep(15)
    return _

def train_pipeline(
    model_name: str,
    hyperparameters: HyperParameters,
    model_metadata: ModelMetadata,
    reward_function: Callable[[Dict], float],
    overwrite: bool = False,
    check_logs_after_start: bool = False
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
    """
    settings.deepracer.run_id = int(os.getenv('DR_RUN_ID', '0'))
    settings.deepracer.local_s3_model_prefix = model_name

    _custom_files_folder = settings.minio.custom_files_folder
    _bucket_name = settings.minio.bucket_name

    reward_function_obj_location_custom = f'{_custom_files_folder}/reward_function.py'
    reward_function_obj_location_model = f'{model_name}/reward_function.py'

    model_data_to_custom_files = forward[None]() >> (
        (
            upload_hyperparameters(hyperparameters=hyperparameters),
            upload_metadata(model_metadata=model_metadata),
            upload_reward_function(reward_function=reward_function)
        )
    )

    check_logs_step = forward[None]() >> sleep_15_seconds >> check_training_logs_transformer

    training_start_pipeline = (
        create_sagemaker_temp_files >>
        check_if_metadata_is_available >>
        check_if_model_exists_transformer(model_name=model_name, overwrite=overwrite) >>
        forward_condition
        .Then(echo(data=None, message=f"Model prefix 's3://{_bucket_name}/{model_name}' exists and overwrite=False. Aborting."))
        .Else(
            model_data_to_custom_files >>
            echo(data=None, message='Data uploaded successfully to custom files') >>
            copy_object(source_object_name=reward_function_obj_location_custom, dest_object_name=reward_function_obj_location_model) >>
            echo(data=None, message=f'The reward function copied successfully to models folder at {reward_function_obj_location_model}') >>
            upload_training_params_file(model_name=model_name) >>
            echo(data=None, message='Upload successfully the RoboMaker training configurations') >>
            expose_config_envs_from_dataclass(model_name=model_name, bucket_name=_bucket_name) >>
            echo(data=None, message='Starting model training') >>
            start_training >>
            echo(data=None, message="Docker stack started.") >>
            If(lambda _: check_logs_after_start)
            .Then(check_logs_step >> echo(data=None, message="Log check performed."))
            .Else(echo(data=None, message="Skipping log check."))
        )
    )

    print(f"Starting training pipeline for model: {model_name}, Run ID: {settings.deepracer.run_id}")
    training_start_pipeline(None)
    print("Training pipeline finished.")

def stop_training_pipeline():
    """
    Stops the currently running DeepRacer training Docker stack.

    Uses the run_id from the current settings (or DR_RUN_ID env var)
    to identify the correct Docker Compose project.
    """
    print("Attempting to stop training stack...")
    try:
        current_run_id = int(os.getenv('DR_RUN_ID', settings.deepracer.run_id))
        settings.deepracer.run_id = current_run_id
        print(f"Targeting Run ID: {current_run_id}")

        docker_manager = DockerManager(settings)

        docker_manager.cleanup_previous_run(prune_system=False)

        print("Training stack stopped successfully.")
    except DockerError as e:
        print(f"Error stopping training stack: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while stopping training: {type(e).__name__} - {e}")
