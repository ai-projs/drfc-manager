from typing import Dict, Any, Optional
import os
import io
import datetime
import yaml
from drfc_manager.config_env import settings
from drfc_manager.evaluation.stop_evaluation_stack import stop_evaluation_stack
from drfc_manager.evaluation.get_compose_files import get_compose_files
from drfc_manager.utils.docker.docker_manager import DockerManager
from drfc_manager.utils.docker.exceptions.base import DockerError
from drfc_manager.utils.minio.storage_manager import MinioStorageManager
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.utils.logging import setup_logging
from drfc_manager.utils.logging_config import get_logger

storage_manager = MinioStorageManager(settings)
docker_manager = DockerManager(settings)
env_vars = EnvVars()
logger = get_logger("evaluation_pipeline")


def evaluate_pipeline(
    model_name: str,
    quiet: bool = True,
    clone: bool = False,
    run_id: Optional[int] = None,
    world_name: Optional[str] = None,
    number_of_trials: Optional[int] = None,
    is_continuous: Optional[bool] = None,
    save_mp4: Optional[bool] = None,
    eval_checkpoint: Optional[str] = None,
    reset_behind_dist: Optional[float] = None,
    off_track_penalty: Optional[float] = None,
    collision_penalty: Optional[float] = None,
    reverse_direction: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Starts model evaluation in DeepRacer using simplified approach with EnvVars directly.

    Args:
        model_name (str): Name of the model prefix to evaluate.
        quiet (bool): If True, suppress verbose output. Defaults to True.
        clone (bool): Copy model into new prefix (<model_name>-E) before evaluating.
        run_id (int, optional): Run ID for the stack name.
        world_name (str, optional): Override the evaluation world/track.
        number_of_trials (int, optional): Override number of evaluation trials.
        is_continuous (bool, optional): Override if evaluation should be continuous.
        save_mp4 (bool, optional): Override if MP4 videos should be saved.
        eval_checkpoint (str, optional): Which model checkpoint to evaluate ('last' or specific).
        reset_behind_dist (float, optional): Distance behind for resets.
        off_track_penalty (float, optional): Penalty for going off track.
        collision_penalty (float, optional): Penalty for collisions.
        reverse_direction (bool, optional): Whether to evaluate in reverse direction.

    Returns:
        Dict[str, Any]: Results of the evaluation pipeline execution.
    """
    effective_run_id = run_id if run_id is not None else env_vars.DR_RUN_ID
    log_path = setup_logging(
        run_id=effective_run_id, model_name=model_name, quiet=quiet
    )

    env_vars.update(
        DR_RUN_ID=effective_run_id,
        DR_LOCAL_S3_MODEL_PREFIX=model_name,
        DR_LOCAL_S3_BUCKET=settings.minio.bucket_name,
    )
    env_vars.load_to_environment()

    if world_name:
        env_vars.update(DR_WORLD_NAME=world_name)
    if number_of_trials is not None:
        env_vars.update(DR_EVAL_NUMBER_OF_TRIALS=number_of_trials)
    if is_continuous is not None:
        env_vars.update(DR_EVAL_IS_CONTINUOUS=is_continuous)
    if save_mp4 is not None:
        env_vars.update(DR_EVAL_SAVE_MP4=save_mp4)
    if eval_checkpoint:
        env_vars.update(DR_EVAL_CHECKPOINT=eval_checkpoint)
    if reset_behind_dist is not None:
        env_vars.update(DR_EVAL_RESET_BEHIND_DIST=reset_behind_dist)
    if off_track_penalty is not None:
        env_vars.update(DR_EVAL_OFF_TRACK_PENALTY=off_track_penalty)
    if collision_penalty is not None:
        env_vars.update(DR_EVAL_COLLISION_PENALTY=collision_penalty)
    if reverse_direction is not None:
        env_vars.update(DR_EVAL_REVERSE_DIRECTION=reverse_direction)
    env_vars.load_to_environment()

    logger.info(
        f"Starting evaluation pipeline for model: {model_name}, Run ID: {effective_run_id}"
    )
    
    # Stop all existing DeepRacer containers to avoid port conflicts
    logger.info("Stopping all existing DeepRacer containers to avoid port conflicts...")
    try:
        # Stop evaluation pipeline for this run ID
        stop_evaluation_pipeline(run_id=effective_run_id)
        
        # Stop training pipeline if running
        from drfc_manager.pipelines.training import stop_training_pipeline
        stop_training_pipeline()
        
        # Additional cleanup: stop any remaining DeepRacer containers
        import subprocess
        import time
        
        # Stop all deepracer stacks
        cleanup_cmds = [
            ["docker", "stack", "rm", "deepracer-0"],
            ["docker", "stack", "rm", "deepracer-eval-0"],
            ["docker", "stack", "rm", "deepracer-1"],
            ["docker", "stack", "rm", "deepracer-eval-1"],
        ]
        
        for cmd in cleanup_cmds:
            try:
                subprocess.run(cmd, check=False, capture_output=True, timeout=10)
            except Exception as e:
                logger.debug(f"Cleanup command failed (expected): {e}")
        
        # Stop all deepracer compose projects
        compose_cleanup_cmds = [
            ["docker", "compose", "-p", "deepracer-0", "down", "--volumes", "--remove-orphans"],
            ["docker", "compose", "-p", "deepracer-eval-0", "down", "--volumes", "--remove-orphans"],
            ["docker", "compose", "-p", "deepracer-1", "down", "--volumes", "--remove-orphans"],
            ["docker", "compose", "-p", "deepracer-eval-1", "down", "--volumes", "--remove-orphans"],
        ]
        
        for cmd in compose_cleanup_cmds:
            try:
                subprocess.run(cmd, check=False, capture_output=True, timeout=10)
            except Exception as e:
                logger.debug(f"Compose cleanup command failed (expected): {e}")
        
        # Wait a moment for containers to stop
        time.sleep(5)
        
        logger.info("DeepRacer container cleanup completed")
        
    except Exception as e:
        logger.warning(f"Cleanup warning (continuing): {e}")

    base_webviewer_port = env_vars.DR_WEBVIEWER_PORT
    base_robomaker_port = env_vars.DR_ROBOMAKER_EVAL_PORT
    base_gui_port = env_vars.DR_ROBOMAKER_GUI_PORT

    port_offset = effective_run_id * 10

    env_vars.update(
        DR_WEBVIEWER_PORT=str(base_webviewer_port + port_offset),
        DR_ROBOMAKER_EVAL_PORT=str(base_robomaker_port + port_offset),
        DR_ROBOMAKER_GUI_PORT=str(base_gui_port + port_offset),
    )
    env_vars.load_to_environment()

    env_vars.update(
        DR_CURRENT_PARAMS_FILE=env_vars.DR_LOCAL_S3_EVAL_PARAMS_FILE,
        STACK_NAME=f"deepracer-eval-{effective_run_id}",
        ROBOMAKER_COMMAND="./run.sh run evaluation.launch"
    )
    env_vars.load_to_environment()

    eval_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    env_vars.DR_SIMTRACE_S3_PREFIX = f"{model_name}/evaluation-{eval_time}"

    try:
        docker_style = env_vars.DR_DOCKER_STYLE.lower()

        if docker_style == "swarm" and docker_manager.list_services(f"deepracer-eval-{effective_run_id}"):
            raise DockerError(f"Stack deepracer-eval-{effective_run_id} already running")

        if clone:
            cloned_prefix = f"{model_name}-E"
            logger.info(f"Cloning model: {model_name} → {cloned_prefix}")
            if model_name != cloned_prefix:
                storage_manager.copy_model_files(
                    f"{model_name}/model", f"{cloned_prefix}/model"
                )
                storage_manager.copy_model_files(
                    f"{model_name}/ip", f"{cloned_prefix}/ip"
                )

                env_vars.update(DR_LOCAL_S3_MODEL_PREFIX=cloned_prefix)
                env_vars.load_to_environment()
                
                model_name = cloned_prefix

        eval_config_dict = env_vars.generate_evaluation_config()
        yaml_content = yaml.dump(
            eval_config_dict,
            default_flow_style=False,
            default_style="'",
            explicit_start=True,
        )
        yaml_bytes = io.BytesIO(yaml_content.encode("utf-8"))
        yaml_length = yaml_bytes.getbuffer().nbytes

        s3_yaml_name = env_vars.DR_CURRENT_PARAMS_FILE
        s3_prefix = env_vars.DR_LOCAL_S3_MODEL_PREFIX
        if not s3_prefix:
            raise ValueError("DR_LOCAL_S3_MODEL_PREFIX is not set")
        if not s3_yaml_name:
            raise ValueError("DR_CURRENT_PARAMS_FILE is not set")

        yaml_key = os.path.normpath(os.path.join(s3_prefix, s3_yaml_name))

        storage_manager._upload_data(
            object_name=yaml_key,
            data=yaml_bytes,
            length=yaml_length,
            content_type="application/x-yaml",
        )
        logger.info(f"Uploaded evaluation config for {model_name}")

        compose_files_str = get_compose_files()

        if quiet:
            try:
                if docker_style == "swarm":
                    output = docker_manager.deploy_stack(
                        stack_name=f"deepracer-eval-{effective_run_id}", compose_files=compose_files_str
                    )
                else:
                    output = docker_manager.compose_up(
                        project_name=f"deepracer-eval-{effective_run_id}", compose_files=compose_files_str
                    )
                logger.info(f"Evaluation started successfully for {model_name}")
            except Exception as e:
                logger.error(f"Failed to start evaluation: {str(e)}")
                raise
        else:
            if docker_style == "swarm":
                output = docker_manager.deploy_stack(
                    stack_name=f"deepracer-eval-{effective_run_id}", compose_files=compose_files_str
                )
            else:
                output = docker_manager.compose_up(
                    project_name=f"deepracer-eval-{effective_run_id}", compose_files=compose_files_str
                )

            if output and output.strip():
                logger.debug(output)
            logger.info(f"Evaluation started for {model_name}")

        result = {
            "status": "success",
            "output": "",
            "model_name": model_name,
            "original_prefix": model_name,
            "run_timestamp": eval_time,
            "log_file": log_path,
        }
        return result

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")

        if (
            clone
            and "cloned_prefix" in locals()
            and env_vars.DR_LOCAL_S3_MODEL_PREFIX == cloned_prefix
        ):
            logger.info(f"Reverting to {model_name} after failure")
            env_vars.update(DR_LOCAL_S3_MODEL_PREFIX=model_name)
            env_vars.load_to_environment()

        result = {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__,
            "model_name": model_name,
            "original_prefix": model_name,
            "log_file": log_path,
        }
        return result


def stop_evaluation_pipeline(run_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Stops the DeepRacer evaluation Docker stack for a given Run ID using Python logic.

    Args:
        run_id (int, optional): The Run ID of the stack to stop.
                                Defaults to env DR_RUN_ID or current setting or 0.

    Returns:
        Dict[str, Any]: Results of the stop operation.
    """
    effective_run_id = run_id if run_id is not None else env_vars.DR_RUN_ID

    stack_name = f"deepracer-eval-{effective_run_id}"
    logger.info(f"Stopping evaluation stack: {stack_name} (Run ID: {effective_run_id})")

    env_vars.update(DR_RUN_ID=str(effective_run_id))
    env_vars.load_to_environment()
    result = stop_evaluation_stack({"stack_name": stack_name})

    if result and "output" in result and result["output"]:
        if result["output"].strip():
            logger.debug(result["output"])

    logger.info("Stop evaluation completed")
    return result
