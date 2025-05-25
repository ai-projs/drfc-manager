from typing import Callable, Dict, Union, Optional
import os
import json

from gloe import transformer, partial_transformer
from minio import Minio as MinioClient

from drfc_manager.helpers.files_manager import create_folder, delete_files_on_folder
from drfc_manager.transformers.exceptions.base import BaseExceptionTransformers
from drfc_manager.types.hyperparameters import HyperParameters
from drfc_manager.types.model_metadata import ModelMetadata
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.helpers.training_params import writing_on_temp_training_yml

from drfc_manager.utils.docker.docker_manager import DockerManager
from drfc_manager.utils.docker.exceptions.base import DockerError
from drfc_manager.utils.minio.storage_manager import MinioStorageManager, StorageError
from drfc_manager.utils.minio.utilities import function_to_bytes_buffer
from drfc_manager.utils.minio.exceptions.file_upload_exception import FileUploadException
from drfc_manager.utils.logging import logger

from drfc_manager.config_env import settings

sagemaker_temp_dir = os.path.expanduser('~/sagemaker_temp')
work_directory = os.path.expanduser('~/dr_work')

storage_manager = MinioStorageManager(settings)
docker_manager = DockerManager(settings)

@transformer
def create_sagemaker_temp_files(_) -> None:
    try:
        create_folder(sagemaker_temp_dir, 0o770)
    except PermissionError as e:
        raise BaseExceptionTransformers(f"Permission denied creating {sagemaker_temp_dir}", e)
    except Exception as e:
        raise BaseExceptionTransformers(f"Failed to create {sagemaker_temp_dir}", e)

@transformer
def check_if_metadata_is_available(_) -> None:
    try:
        create_folder(work_directory)
        delete_files_on_folder(work_directory)
    except PermissionError as e:
        raise BaseExceptionTransformers(f"Permission denied accessing {work_directory}", e)
    except Exception as e:
        raise BaseExceptionTransformers(f"Failed to setup {work_directory}", e)
    
    
@partial_transformer
def upload_hyperparameters(_, hyperparameters: HyperParameters):
    try:
        storage_manager.upload_hyperparameters(hyperparameters)
    except Exception as e:
        raise BaseExceptionTransformers("Failed to upload hyperparameters", e)
    
    
@partial_transformer
def upload_metadata(_, model_metadata: ModelMetadata):
    try:
        storage_manager.upload_model_metadata(model_metadata)
    except Exception as e:
        raise BaseExceptionTransformers("Failed to upload model metadata", e)


@partial_transformer
def upload_reward_function(_, reward_function: Union[Callable[[Dict], float], str], object_name: Optional[str] = None):
    if object_name is None:
        object_name = f"{storage_manager.config.custom_files_folder}/reward_function.py"
    try:
        if isinstance(reward_function, str):
            data_bytes = reward_function.encode('utf-8')
            storage_manager._upload_data(object_name, data_bytes, len(data_bytes), 'text/x-python')
        else:
            buffer = function_to_bytes_buffer(reward_function)
            storage_manager._upload_data(object_name, buffer, buffer.getbuffer().nbytes, 'text/x-python')
    except Exception as e:
        raise FileUploadException("reward_function.py", str(e)) from e
    

def verify_object_exists(minio_client: MinioClient, object_name: str) -> bool:
    try:
        minio_client.stat_object("tcc-experiments", object_name)
        return True
    except Exception:
        return False

@partial_transformer
def upload_training_params_file(_, model_name: str):
    local_yaml_path = None
    try:
        logger.info("Generating local training_params.yaml...")
        relevant_envs = EnvVars(DR_LOCAL_S3_MODEL_PREFIX=model_name, DR_LOCAL_S3_BUCKET=settings.minio.bucket_name)
        relevant_envs.load_to_environment()

        yaml_key, local_yaml_path = writing_on_temp_training_yml(model_name)
        logger.info(f"Generated {local_yaml_path}, uploading to {yaml_key}")

        storage_manager.upload_local_file(local_yaml_path, yaml_key)

        if not storage_manager.object_exists(yaml_key):
            raise StorageError(f"Verification failed: {yaml_key} not found after upload.")
        logger.info(f"Verified: Training params file exists at {yaml_key}")

    except Exception as e:
        raise BaseExceptionTransformers("Failed to upload training parameters file", e)
    finally:
        if local_yaml_path and os.path.exists(local_yaml_path):
            try:
                os.remove(local_yaml_path)
                logger.info(f"Cleaned up local file: {local_yaml_path}")
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {local_yaml_path}: {e}")

@transformer
def start_training(_):
    try:
        logger.info("Attempting to start DeepRacer Docker stack...")
        docker_manager.cleanup_previous_run(prune_system=True)
        docker_manager.start_deepracer_stack()
        logger.info("DeepRacer Docker stack started successfully.")
    except DockerError as e:
        logger.error(f"DockerError starting stack: {e}")
        raise BaseExceptionTransformers("Docker stack startup failed", e)
    except Exception as e:
        logger.error(f"Unexpected error starting stack: {type(e).__name__}: {e}")
        raise BaseExceptionTransformers("Unexpected error during stack startup", e)

@transformer
def stop_training_transformer(_):
    try:
        logger.info("Stopping DeepRacer Docker stack via transformer...")
        docker_manager.cleanup_previous_run(prune_system=False)
        logger.info("DeepRacer Docker stack stopped via transformer.")
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to stop the training via transformer", e)

@transformer
def check_training_logs_transformer(_):
    try:
        docker_manager.check_logs("redis")
        docker_manager.check_logs("rl_coach")
        docker_manager.check_logs("robomaker")
        logger.info("Log check complete.")
        return True
    except Exception as e:
        logger.error(f"Error checking logs: {e}")
        return False
    

@partial_transformer
def expose_config_envs_from_dataclass(_, model_name: str, bucket_name: str) -> None:
    """
    Loads key DR_* environment variables into the current Python process environment.
    Needed primarily for helpers like writing_on_temp_training_yml that read os.environ.
    Container environment variables are set separately by DockerManager.
    """
    try:
        env_loader = EnvVars(
             DR_LOCAL_S3_MODEL_PREFIX=model_name,
             DR_LOCAL_S3_BUCKET=bucket_name,
             DR_AWS_APP_REGION=os.getenv('DR_AWS_APP_REGION', 'us-east-1'),
        )
        env_loader.load_to_environment()
        logger.info(f"Loaded DR_* vars for model '{model_name}' into current process environment.")
    except Exception as e:
        logger.warning(f"Failed to load DR_* vars into process environment: {e}")

@partial_transformer
def upload_ip_config(_, model_name: str):
    """Upload Redis IP config (ip.json and done flag) to S3"""
    # Determine Redis host (Docker DNS name)
    redis_host = os.environ.get('REDIS_HOST', settings.redis.host)
    # Prepare IP config
    ip_config = {'IP': redis_host}
    object_name = f'{model_name}/ip/ip.json'
    data_bytes = json.dumps(ip_config).encode('utf-8')
    # Upload ip.json
    storage_manager._upload_data(object_name, data_bytes, len(data_bytes), 'application/json')
    # Upload done flag
    done_key = f'{model_name}/ip/done'
    storage_manager._upload_data(done_key, b'done', len(b'done'), 'application/octet-stream')
    logger.info(f'Uploaded Redis IP config to {object_name} and done flag to {done_key}')