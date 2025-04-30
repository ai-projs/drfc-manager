import io
import os
from typing import Dict, Any

import yaml
from src.utils.docker.docker_manager import DockerManager
from src.utils.docker.exceptions.base import DockerError
from src.utils.minio.storage_manager import MinioStorageManager
from src.config import settings
from src.evaluation.prepare_evaluation_configs import prepare_evaluation_config
from src.evaluation.get_compose_files import get_compose_files
from gloe import transformer


storage_manager = MinioStorageManager(settings)
docker_manager = DockerManager(settings)


@transformer
def start_evaluation_stack(data: Dict[str, Any]):
    """Start the evaluation Docker stack using DockerManager and generated config."""
    stack_name = data['stack_name']
    model_name = data['model_name']
    original_prefix = data['original_prefix']
    clone = data['clone']

    print(f"Starting evaluation for model {model_name} in stack {stack_name}")

    try:
        docker_style = os.environ.get('DR_DOCKER_STYLE', 'compose').lower()
        if docker_style == "swarm":
            services = docker_manager.list_services(stack_name)
            if services:
                raise DockerError(f"Stack {stack_name} already running (found services). Stop evaluation first.")

        if clone:
            cloned_prefix = f"{original_prefix}-E"
            print(f"Cloning requested: {original_prefix} -> {cloned_prefix}")
            s3_bucket = os.environ.get('DR_LOCAL_S3_BUCKET')
            if model_name != cloned_prefix:
                try:
                    storage_manager.copy_directory(s3_bucket, f"{original_prefix}/model", f"{cloned_prefix}/model")
                    storage_manager.copy_directory(s3_bucket, f"{original_prefix}/ip", f"{cloned_prefix}/ip")

                    os.environ['DR_LOCAL_S3_MODEL_PREFIX'] = cloned_prefix

                    if hasattr(settings.deepracer, 'local_s3_model_prefix'):
                         settings.deepracer.local_s3_model_prefix = cloned_prefix
                    data['model_name'] = cloned_prefix
                    model_name = cloned_prefix
                except Exception as e:
                     print(f"Error cloning model: {e}")
                     raise RuntimeError(f"Failed to clone model from {original_prefix} to {cloned_prefix}: {e}") from e


        eval_config_dict = prepare_evaluation_config()
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

        data["status"] = "success"
        data["output"] = output
        return data

    except Exception as e:
        print(f"Error starting evaluation stack: {type(e).__name__}: {e}")
        # Attempt to revert prefix in environment if cloning happened before failure
        if clone and 'cloned_prefix' in locals() and os.environ.get('DR_LOCAL_S3_MODEL_PREFIX') == cloned_prefix:
             print(f"Attempting to revert environment prefix to {original_prefix} after failure.")
             os.environ['DR_LOCAL_S3_MODEL_PREFIX'] = original_prefix
             if hasattr(settings.deepracer, 'local_s3_model_prefix'):
                  settings.deepracer.local_s3_model_prefix = original_prefix

        data["status"] = "error"
        data["error"] = str(e)
        data["type"] = type(e).__name__
        return data