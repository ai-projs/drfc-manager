from io import BytesIO
from typing import Callable, Dict

from gloe import transformer, partial_transformer
from minio import Minio as MinioClient
from minio.error import MinioException

from src.helpers.files_manager import create_folder, delete_files_on_folder
from src.transformers.exceptions.base import BaseExceptionTransformers
from src.types.config import ConfigEnvs
from src.types.hyperparameters import HyperParameters
from src.types.model_metadata import ModelMetadata
from src.utils.commands.docker_compose import DockerComposeCommands
from src.utils.minio.exceptions.file_upload_exception import FunctionConversionException
from src.utils.minio.utilities import upload_hyperparameters as _upload_hyperparameters, function_to_bytes_buffer
from src.utils.minio.utilities import upload_reward_function as _upload_reward_function
from src.utils.minio.utilities import upload_metadata as _upload_metadata
from src.utils.minio.utilities import upload_local_data as _upload_local_data
from src.types.docker import DockerImages
from src.types.env_vars import EnvVars
from src.helpers.training_params import writing_on_temp_training_yml

sagemaker_temp_dir = '/tmp/sagemaker'
work_directory = '/tmp/teste'

docker_compose = DockerComposeCommands()


@transformer
def create_sagemaker_temp_files(_) -> None:
    try:
        create_folder(sagemaker_temp_dir, 0o770)
    except PermissionError as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to create the sagemaker's temp folder", e)

@partial_transformer
def expose_config_envs_from_dataclass(_, model_name: str, bucket_name: str) -> None:
    try:
        env = EnvVars(DR_LOCAL_S3_MODEL_PREFIX=model_name, DR_LOCAL_S3_BUCKET=bucket_name)
        env.load_to_environment()
    except Exception as e:
        raise BaseExceptionTransformers(f"Failed to load environment variables via dataclass", e)

@transformer
def check_if_metadata_is_available(_) -> None:
    try:
        create_folder(work_directory)
        delete_files_on_folder(work_directory)
    except PermissionError as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to check if the metadata is available", e)
    
    
@partial_transformer
def upload_hyperparameters(_, minio_client: MinioClient, hyperparameters: HyperParameters, object_name: str = None):
    try:
        _upload_hyperparameters(minio_client, hyperparameters, object_name=object_name)
    except MinioException as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to upload the hyperparameters", e)
    
    
@partial_transformer
def upload_metadata(_, minio_client: MinioClient, model_metadata: ModelMetadata, object_name: str = None):
    try:
        _upload_metadata(minio_client, model_metadata, object_name=object_name)
    except MinioException as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to upload the model metadata", e)


@partial_transformer
def upload_reward_function(_, minio_client: MinioClient, reward_function: Callable[[Dict], float], object_name: str = None):
    try:
        reward_function_buffer = function_to_bytes_buffer(reward_function)
        _upload_reward_function(minio_client, reward_function_buffer, object_name=object_name)
    except (MinioException, FunctionConversionException) as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to upload the reward function", e)
    

def verify_object_exists(minio_client: MinioClient, object_name: str) -> bool:
    try:
        minio_client.stat_object("tcc-experiments", object_name)
        return True
    except Exception:
        return False

@partial_transformer
def upload_training_params_file(_, minio_client: MinioClient, model_name: str):
    try:
        yaml_key, local_yaml_path = writing_on_temp_training_yml(model_name)
        _upload_local_data(minio_client, local_yaml_path, yaml_key)
        if verify_object_exists(minio_client, yaml_key):
            print(f"Verified: Training params file exists at {yaml_key}")
        else:
            raise Exception(f"File not found after upload: {yaml_key}")
    except MinioException as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to upload the training parameters file", e)
    

@transformer
def start_training(_):
    try:
        images_to_start_training = [DockerImages.training, DockerImages.keys, DockerImages.endpoint]
        docker_compose.up(images_to_start_training)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to start the training", e)


@transformer
def stop_training(_):
    try:
        images_to_stop_training = [DockerImages.training, DockerImages.keys, DockerImages.endpoint]
        docker_compose.down(images_to_stop_training)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to stop the training", e)
