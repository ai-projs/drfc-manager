import io
import os
from typing import Dict, Any

import yaml
from drfc_manager.utils.docker.docker_manager import DockerManager
from drfc_manager.utils.docker.exceptions.base import DockerError
from drfc_manager.utils.minio.storage_manager import MinioStorageManager
from drfc_manager.config_env import settings
from drfc_manager.evaluation.get_compose_files import get_compose_files
from gloe import transformer
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.utils.logging import logger

storage_manager = MinioStorageManager(settings)
docker_manager = DockerManager(settings)


@transformer
def start_evaluation_stack(data: Dict[str, Any]):
    """Start the evaluation Docker stack using DockerManager and generated config."""
    config = data.get('config')
    if not config:
        raise ValueError("start_evaluation_stack requires 'config' in the input data dictionary.")

    stack_name = data['stack_name']
    model_name = config.model_name
    original_prefix = config.model_name
    clone = config.clone

    logger.info(f"Starting evaluation for model {model_name} in stack {stack_name}")

    try:
        docker_style = os.environ.get('DR_DOCKER_STYLE', 'compose').lower()
        if docker_style == "swarm":
            services = docker_manager.list_services(stack_name)
            if services:
                raise DockerError(f"Stack {stack_name} already running (found services). Stop evaluation first.")

        if clone:
            cloned_prefix = f"{original_prefix}-E"
            logger.info(f"Cloning requested: {original_prefix} -> {cloned_prefix}")
            s3_bucket = os.environ.get('DR_LOCAL_S3_BUCKET')
            if model_name != cloned_prefix:
                try:
                    storage_manager.copy_directory(s3_bucket, f"{original_prefix}/model", f"{cloned_prefix}/model")
                    storage_manager.copy_directory(s3_bucket, f"{original_prefix}/ip", f"{cloned_prefix}/ip")

                    os.environ['DR_LOCAL_S3_MODEL_PREFIX'] = cloned_prefix
                    if hasattr(settings.deepracer, 'local_s3_model_prefix'):
                        settings.deepracer.local_s3_model_prefix = cloned_prefix
                    config.model_name = cloned_prefix
                    config.env_vars.DR_LOCAL_S3_MODEL_PREFIX = cloned_prefix
                    model_name = cloned_prefix
                except Exception as e:
                    logger.error(f"Error cloning model: {e}")
                    raise RuntimeError(f"Failed to clone model from {original_prefix} to {cloned_prefix}: {e}") from e

        eval_config_dict = EnvVars.generate_evaluation_config()
        yaml_content = yaml.dump(eval_config_dict, default_flow_style=False, default_style="'", explicit_start=True)
        yaml_bytes = io.BytesIO(yaml_content.encode('utf-8'))
        yaml_length = yaml_bytes.getbuffer().nbytes

        s3_yaml_name = os.environ.get('DR_CURRENT_PARAMS_FILE', 'eval_params.yaml')
        s3_prefix = os.environ.get('DR_LOCAL_S3_MODEL_PREFIX')
        yaml_key = os.path.normpath(os.path.join(s3_prefix, s3_yaml_name))

        storage_manager._upload_data(
            object_name=yaml_key,
            data=yaml_bytes,
            length=yaml_length,
            content_type='application/x-yaml'
        )

        compose_files_str = get_compose_files()

        if docker_style == "swarm":
            output = docker_manager.deploy_stack(stack_name=stack_name, compose_files=compose_files_str)
        else:
            output = docker_manager.compose_up(project_name=stack_name, compose_files=compose_files_str)

        data.update({
            "status": "success",
            "output": output,
            "model_name": model_name,
            "original_prefix": original_prefix
        })
        return data

    except Exception as e:
        logger.error(f"Error starting evaluation stack: {type(e).__name__}: {e}")
        if clone and 'cloned_prefix' in locals() and os.environ.get('DR_LOCAL_S3_MODEL_PREFIX') == cloned_prefix:
            logger.info(f"Attempting to revert environment prefix to {original_prefix} after failure.")
            os.environ['DR_LOCAL_S3_MODEL_PREFIX'] = original_prefix
            if hasattr(settings.deepracer, 'local_s3_model_prefix'):
                settings.deepracer.local_s3_model_prefix = original_prefix
            config.model_name = original_prefix
            config.env_vars.DR_LOCAL_S3_MODEL_PREFIX = original_prefix

        data.update({
            "status": "error",
            "error": str(e),
            "type": type(e).__name__,
            "model_name": model_name,
            "original_prefix": original_prefix
        })
        return data