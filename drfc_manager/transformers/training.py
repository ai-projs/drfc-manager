import os
import json

from gloe import transformer
from minio import Minio as MinioClient

from drfc_manager.helpers.files_manager import create_folder, delete_files_on_folder
from drfc_manager.transformers.exceptions.base import BaseExceptionTransformers
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.helpers.training_params import writing_on_temp_training_yml

from drfc_manager.utils.docker.docker_manager import DockerManager
from drfc_manager.utils.docker.exceptions import DockerError
from drfc_manager.utils.minio.storage_manager import MinioStorageManager, StorageError
from drfc_manager.utils.minio.utilities import function_to_bytes_buffer
from drfc_manager.utils.minio.exceptions.file_upload_exception import (
    FileUploadException,
)
from drfc_manager.utils.logging import logger

from drfc_manager.config_env import settings

sagemaker_temp_dir = os.path.expanduser("~/sagemaker_temp")
work_directory = os.path.expanduser("~/dr_work")

storage_manager = MinioStorageManager(settings)
docker_manager = DockerManager(settings)


@transformer
def create_training_data_directories(data):
    """Create the necessary directory structure for training data."""
    try:
        # Create the main custom_files directory in the current directory
        custom_files_dir = os.path.join(os.getcwd(), "custom_files")
        create_folder(custom_files_dir, 0o777)

        # Create the iteration_data directory
        iteration_data_dir = os.path.join(custom_files_dir, "iteration_data")
        create_folder(iteration_data_dir, 0o777)

        # Create the agent directory
        agent_dir = os.path.join(iteration_data_dir, "agent")
        create_folder(agent_dir, 0o777)

        # Create the training-simtrace directory
        training_simtrace_dir = os.path.join(agent_dir, "training-simtrace")
        create_folder(training_simtrace_dir, 0o777)

        # Create an empty iteration.csv file
        iteration_csv_path = os.path.join(training_simtrace_dir, "iteration.csv")
        if not os.path.exists(iteration_csv_path):
            with open(iteration_csv_path, "w"):
                pass
            os.chmod(iteration_csv_path, 0o666)

        logger.info("Created training data directory structure successfully")
    except PermissionError as e:
        raise BaseExceptionTransformers(
            f"Permission denied creating training data directories: {e}", e
        )
    except Exception as e:
        raise BaseExceptionTransformers(
            f"Failed to create training data directories: {e}", e
        )
    return data


@transformer
def create_sagemaker_temp_files(data):
    try:
        create_folder(sagemaker_temp_dir, 0o770)
    except PermissionError as e:
        raise BaseExceptionTransformers(
            f"Permission denied creating {sagemaker_temp_dir}", e
        )
    except Exception as e:
        raise BaseExceptionTransformers(f"Failed to create {sagemaker_temp_dir}", e)
    return data


@transformer
def check_if_metadata_is_available(data):
    try:
        create_folder(work_directory)
        delete_files_on_folder(work_directory)
    except PermissionError as e:
        raise BaseExceptionTransformers(
            f"Permission denied accessing {work_directory}", e
        )
    except Exception as e:
        raise BaseExceptionTransformers(f"Failed to setup {work_directory}", e)
    return data


@transformer
def upload_hyperparameters(data):
    try:
        storage_manager.upload_hyperparameters(data["hyperparameters"])
    except Exception as e:
        raise BaseExceptionTransformers("Failed to upload hyperparameters", e)
    return data


@transformer
def upload_metadata(data):
    try:
        storage_manager.upload_model_metadata(data["model_metadata"])
    except Exception as e:
        raise BaseExceptionTransformers("Failed to upload model metadata", e)
    return data


@transformer
def upload_reward_function(data):
    reward_function = data.get("reward_function_code") or data.get("reward_function")
    object_name = data.get("reward_function_obj_location_custom")
    if object_name is None:
        object_name = f"{storage_manager.config.custom_files_folder}/reward_function.py"
    try:
        if isinstance(reward_function, str):
            data_bytes = reward_function.encode("utf-8")
            storage_manager._upload_data(
                object_name, data_bytes, len(data_bytes), "text/x-python"
            )
        else:
            buffer = function_to_bytes_buffer(reward_function)
            storage_manager._upload_data(
                object_name, buffer, buffer.getbuffer().nbytes, "text/x-python"
            )
    except Exception as e:
        raise FileUploadException("reward_function.py", str(e)) from e
    return data


def verify_object_exists(minio_client: MinioClient, object_name: str) -> bool:
    try:
        minio_client.stat_object("tcc-experiments", object_name)
        return True
    except Exception:
        return False


@transformer
def upload_training_params_file(data):
    model_name = data["model_name"]
    local_yaml_path = None
    try:
        logger.info("Generating local training_params.yaml...")
        relevant_envs = EnvVars(
            DR_LOCAL_S3_MODEL_PREFIX=model_name,
            DR_LOCAL_S3_BUCKET=settings.minio.bucket_name,
        )
        relevant_envs.load_to_environment()

        yaml_key, local_yaml_path = writing_on_temp_training_yml(model_name)
        logger.info(f"Generated {local_yaml_path}, uploading to {yaml_key}")

        storage_manager.upload_local_file(local_yaml_path, yaml_key)

        if not storage_manager.object_exists(yaml_key):
            raise StorageError(
                f"Verification failed: {yaml_key} not found after upload."
            )
        logger.info(f"Verified: Training params file exists at {yaml_key}")

    except Exception as e:
        raise BaseExceptionTransformers("Failed to upload training parameters file", e)
    finally:
        if local_yaml_path and os.path.exists(local_yaml_path):
            try:
                os.remove(local_yaml_path)
                logger.info(f"Cleaned up local file: {local_yaml_path}")
            except OSError as e:
                logger.warning(
                    f"Failed to remove temporary file {local_yaml_path}: {e}"
                )
    return data


@transformer
def start_training(data):
    try:
        logger.info("Attempting to start DeepRacer Docker stack...")
        docker_manager.cleanup_previous_run(prune=True)
        docker_manager.start_deepracer_stack()
        logger.info("DeepRacer Docker stack started successfully.")
    except DockerError as e:
        logger.error(f"DockerError starting stack: {e}")
        raise BaseExceptionTransformers("Docker stack startup failed", e)
    except Exception as e:
        logger.error(f"Unexpected error starting stack: {type(e).__name__}: {e}")
        raise BaseExceptionTransformers("Unexpected error during stack startup", e)
    return data


@transformer
def stop_training_transformer(data):
    try:
        logger.info("Stopping DeepRacer Docker stack via transformer...")
        docker_manager.cleanup_previous_run(prune=False)
        logger.info("DeepRacer Docker stack stopped via transformer.")
    except Exception as e:
        raise BaseExceptionTransformers(
            "It was not possible to stop the training via transformer", e
        )
    return data


@transformer
def check_training_logs_transformer(data):
    try:
        docker_manager.check_logs("redis")
        docker_manager.check_logs("rl_coach")
        docker_manager.check_logs("robomaker")
        logger.info("Log check complete.")
        return data
    except Exception as e:
        logger.error(f"Error checking logs: {e}")
        return data


@transformer
def expose_config_envs_from_dataclass(data):
    model_name = data["model_name"]
    bucket_name = data["bucket_name"]
    try:
        env_loader = EnvVars(
            DR_LOCAL_S3_MODEL_PREFIX=model_name,
            DR_LOCAL_S3_BUCKET=bucket_name,
            DR_AWS_APP_REGION=os.getenv("DR_AWS_APP_REGION", "us-east-1"),
        )
        env_loader.load_to_environment()
        logger.info(
            f"Loaded DR_* vars for model '{model_name}' into current process environment."
        )
    except Exception as e:
        logger.warning(f"Failed to load DR_* vars into process environment: {e}")
    return data


@transformer
def upload_ip_config(data):
    model_name = data["model_name"]
    redis_host = os.environ.get("REDIS_HOST", settings.redis.host)
    ip_config = {"IP": redis_host}
    object_name = f"{model_name}/ip/ip.json"
    data_bytes = json.dumps(ip_config).encode("utf-8")
    storage_manager._upload_data(
        object_name, data_bytes, len(data_bytes), "application/json"
    )
    done_key = f"{model_name}/ip/done"
    storage_manager._upload_data(
        done_key, b"done", len(b"done"), "application/octet-stream"
    )
    logger.info(
        f"Uploaded Redis IP config to {object_name} and done flag to {done_key}"
    )
    return data
