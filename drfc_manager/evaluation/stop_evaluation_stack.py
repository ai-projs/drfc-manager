import os
from typing import Dict, Any, List
from drfc_manager.utils.docker.docker_manager import DockerManager
from drfc_manager.config_env import settings
from gloe import transformer
from drfc_manager.utils.docker.utilities import adjust_composes_file_names
from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.logging import logger
from drfc_manager.utils.docker.exceptions.base import DockerError
from drfc_manager.types.env_vars import EnvVars

docker_manager = DockerManager(settings)
env_vars = EnvVars()


@transformer
def stop_evaluation_stack(data: Dict[str, Any]):
    """Stop the evaluation Docker stack using DockerManager."""
    stack_name = data.get("stack_name")
    if not stack_name:
        run_id = env_vars.DR_RUN_ID
        stack_name = f"deepracer-eval-{run_id}"
        logger.info(f"Stack name not in data, reconstructing: {stack_name}")
        data["stack_name"] = stack_name

    logger.info(f"Stopping evaluation stack: {stack_name}")

    try:
        docker_style = env_vars.DR_DOCKER_STYLE.lower()
        if docker_style == "swarm":
            output = docker_manager.remove_stack(stack_name=stack_name)
        else:
            eval_compose_paths: List[str] = adjust_composes_file_names(
                [ComposeFileType.EVAL.value]
            )

            if not eval_compose_paths or not eval_compose_paths[0]:
                logger.error(
                    f"Could not resolve path for {ComposeFileType.EVAL.value}. Cannot perform compose down accurately."
                )
                raise ValueError(
                    f"Could not resolve path for {ComposeFileType.EVAL.value} using _adjust_composes_file_names"
                )

            base_compose_file_path = eval_compose_paths[0]

            if not base_compose_file_path or not os.path.exists(base_compose_file_path):
                logger.error(
                    f"Evaluation compose file path not found or invalid: '{base_compose_file_path}'. Cannot perform compose down accurately."
                )
                raise ValueError(
                    f"Resolved evaluation compose file path does not exist: '{base_compose_file_path}'"
                )
            else:
                output = docker_manager.compose_down(
                    project_name=stack_name,
                    compose_files=base_compose_file_path,
                    remove_volumes=True,
                )

        if output and output.strip():
            logger.debug(output)

        data["status"] = "success"
        data["output"] = output
        return data
    except DockerError as e:
        logger.error(f"DockerError stopping stack: {e}")
        data["status"] = "error"
        data["error"] = str(e)
        data["type"] = "DockerError"
        return data
    except Exception as e:
        logger.error(f"Unexpected error stopping stack: {type(e).__name__}: {e}")
        data["status"] = "error"
        data["error"] = str(e)
        data["type"] = type(e).__name__
        return data
