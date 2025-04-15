import os
from copy import copy
from io import BytesIO
from typing import Callable, Dict

from gloe import If
from gloe.utils import debug, forward

from src.transformers.training import create_sagemaker_temp_files, check_if_metadata_is_available, \
    upload_hyperparameters, upload_metadata, upload_reward_function, upload_training_params_file, start_training, \
    expose_config_envs_from_dataclass
from src.utils.docker.server import DockerClientServer
from src.transformers.general import check_if_model_exists, echo, forward_condition, copy_object, image_tag_has_running_container
from src.types.hyperparameters import HyperParameters
from src.types.model_metadata import ModelMetadata
from src.utils.minio.server import MinioClientServer


_docker_client = DockerClientServer.get_instance()
_minio_client = MinioClientServer.get_instance()
_bucket_name = os.getenv('BUCKET_NAME')
simapp_tag = os.getenv('SIMAPP_IMAGE_REPOTAG')
_custom_files_folder = os.getenv('CUSTOM_FILES_FOLDER_PATH')

def train_pipeline(
    model_name: str,
    hyperparameters: HyperParameters,
    model_metadata: ModelMetadata,
    reward_function: Callable[[Dict], float],
    overwrite: bool = False
):
    """
    Orchestrates the training pipeline for a DeepRacer Reinforcement Learning model.

    This pipeline handles:
      - Uploading hyperparameters, metadata, and reward function to MinIO
      - Verifying container and model states to prevent conflicts
      - Copying the reward function to the model's folder
      - Uploading RoboMaker training configuration files
      - Initiating the training setup, assuming all checks pass

    Args:
        model_name (str): Name of the model to be trained.
        hyperparameters (HyperParameters): Training hyperparameters to upload.
        model_metadata (ModelMetadata): Metadata describing the model's configuration.
        reward_function (Callable[[Dict], float]): Python function defining the reward logic for training.
        overwrite (bool, optional): If True, overwrites an existing model with the same name. Defaults to False.

    Raises:
        FileUploadException: If any file upload to MinIO fails.
        FunctionConversionException: If reward function conversion to a file fails.

    Returns:
        None
    """

    reward_function_obj_location_custom = f'{_custom_files_folder}/reward_function.py'
    reward_function_obj_location_model = f'{model_name}/reward_function.py'

    model_data_to_custom_files = forward[None]() >> (
        (
            upload_hyperparameters(_minio_client, hyperparameters),
            upload_metadata(_minio_client, model_metadata),
            upload_reward_function(_minio_client, reward_function)
        )
    )

    training_start_pipeline = (
        create_sagemaker_temp_files >>
        check_if_metadata_is_available >>
        image_tag_has_running_container(_docker_client, simapp_tag) >>
        forward_condition
        .Then(echo("The training is running, please stop the train before starting a new one."))
        .Else(
            check_if_model_exists(_minio_client, model_name, overwrite) >>
            forward_condition
            .Then(echo("The model already exists, use another name or set `overwrite` param to True!"))
            .Else(
                model_data_to_custom_files >>
                echo('Data uploaded successfully to custom files') >>
                # Removing previous data not working, check if it is required
                # remove_objects_folder(_minio_client, model_name) >>
                # echo('Previous models data was successfully cleaned') >>
                copy_object(_minio_client, reward_function_obj_location_custom, reward_function_obj_location_model) >>
                echo(f'The reward function copied successfully to models folder '
                     f'at {reward_function_obj_location_model}') >>
                upload_training_params_file(_minio_client, model_name) >>
                echo('Upload successfully the RoboMaker training configurations') >>
                echo('Exposing the envs from config.env and system.env') >>
                expose_config_envs_from_dataclass(model_name, _bucket_name) >>
                echo('Starting model training') >>
                start_training
            )
        )
    )
    training_start_pipeline(None)
