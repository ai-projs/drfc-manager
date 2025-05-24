import os
from typing import Dict, Any, List
from src.utils.docker.docker_manager import DockerManager
from src.config import settings
from gloe import transformer
from src.utils.docker.utilities import _adjust_composes_file_names
from src.types.docker import ComposeFileType
from src.utils.logging import logger

docker_manager = DockerManager(settings)

@transformer
def stop_evaluation_stack(data: Dict[str, Any]):
    """Stop the evaluation Docker stack using DockerManager."""
    stack_name = data.get('stack_name')
    if not stack_name:
        run_id = os.environ.get('DR_RUN_ID', getattr(settings.deepracer, 'run_id', 0))
        stack_name = f"deepracer-eval-{run_id}"
        logger.info(f"Stack name not in data, reconstructing: {stack_name}")
        data['stack_name'] = stack_name

    logger.info(f"Stopping evaluation stack: {stack_name}")

    try:
        docker_style = os.environ.get('DR_DOCKER_STYLE', 'compose').lower()
        if docker_style == "swarm":
            output = docker_manager.remove_stack(stack_name=stack_name)
        else:
            eval_compose_paths: List[str] = _adjust_composes_file_names([ComposeFileType.EVAL.value])

            if not eval_compose_paths or not eval_compose_paths[0]:
                 logger.error(f"Could not resolve path for {ComposeFileType.EVAL.value}. Cannot perform compose down accurately.")
                 raise ValueError(f"Could not resolve path for {ComposeFileType.EVAL.value} using _adjust_composes_file_names")

            base_compose_file_path = eval_compose_paths[0]

            if not base_compose_file_path or not os.path.exists(base_compose_file_path):
                 logger.error(f"Evaluation compose file path not found or invalid: '{base_compose_file_path}'. Cannot perform compose down accurately.")
                 raise ValueError(f"Resolved evaluation compose file path does not exist: '{base_compose_file_path}'")
            else:
                 output = docker_manager.compose_down(project_name=stack_name, compose_files=base_compose_file_path, remove_volumes=True)

        data["status"] = "success"
        data["output"] = output
        return data
    except Exception as e:
        logger.error(f"Error stopping evaluation stack: {e}")
        data["status"] = "error"
        data["error"] = str(e)
        data["type"] = type(e).__name__
        return data