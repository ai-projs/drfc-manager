from typing import Dict, Any, Optional
import os
from src.config import settings
from src.evaluation.setup_evaluation_env import setup_evaluation_env
from src.evaluation.start_evaluation_stack import start_evaluation_stack
from src.evaluation.stop_evaluation_stack import stop_evaluation_stack
from gloe import If
from gloe.utils import forward
from src.transformers.general import passthrough

def evaluate_pipeline(
    model_name: str,
    quiet: bool = False,
    clone: bool = False,
    run_id: Optional[int] = None,
    # Add other EnvVars overrides here if needed, e.g., world_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Starts model evaluation in DeepRacer using Python logic and EnvVars.

    Args:
        model_name (str): Name of the model prefix to evaluate.
        quiet (bool): If True, suppress verbose output (currently minimal effect).
        clone (bool): Copy model into new prefix (<model_name>-E) before evaluating.
        run_id (int, optional): Run ID for the stack name. Defaults to env DR_RUN_ID or 0.
        # Add other args corresponding to EnvVars overrides

    Returns:
        Dict[str, Any]: Results of the evaluation pipeline execution.
    """
    # Determine Run ID
    effective_run_id = run_id if run_id is not None else int(os.getenv('DR_RUN_ID', getattr(settings.deepracer, 'run_id', 0)))
    settings.deepracer.run_id = effective_run_id # Update runtime setting

    print(f"Starting evaluation pipeline for model: {model_name}, Run ID: {effective_run_id}")
    original_env_prefix = os.environ.get('DR_LOCAL_S3_MODEL_PREFIX') # Store initial state

    # First stop any running evaluation with the same Run ID
    stop_result = stop_evaluation_pipeline(run_id=effective_run_id)

    # --- Add defensive check for None ---
    if stop_result is None:
        print(f"Warning: stop_evaluation_pipeline returned None for run_id {effective_run_id}. Cannot check status.")
    elif stop_result.get("status") == "error":
         print(f"Warning: Failed to stop existing evaluation stack for run_id {effective_run_id}: {stop_result.get('error')}")
         # Continue anyway, start might fail if resources are locked.

    # --- Pass parameters via initial_data ---
    initial_data = {
        "model_name": model_name,
        "run_id": effective_run_id,
        "clone": clone,
        "quiet": quiet,
        "original_prefix": model_name # Keep original prefix here too for consistency
    }

    evaluation_pipeline_flow = forward[Dict[str, Any]]() >> (
        setup_evaluation_env >>
        start_evaluation_stack >>
        If(lambda data: data.get("status") == "success")
            .Then(passthrough)
        .Else(passthrough)
    )

    result = evaluation_pipeline_flow(initial_data)
    print("Evaluation pipeline execution completed.")

    # --- Cleanup environment after run ---
    final_prefix = os.environ.get('DR_LOCAL_S3_MODEL_PREFIX')
    # --- Get original prefix from the result dictionary ---
    prefix_to_restore = result.get('original_prefix', original_env_prefix)

    # Restore original model prefix in environment and settings if it was changed
    if final_prefix != prefix_to_restore and prefix_to_restore is not None:
        # print(f"Restoring original model prefix '{prefix_to_restore}' in environment and settings.")
        os.environ['DR_LOCAL_S3_MODEL_PREFIX'] = prefix_to_restore
        # Check if attribute exists before setting
        if hasattr(settings.deepracer, 'local_s3_model_prefix'):
            settings.deepracer.local_s3_model_prefix = prefix_to_restore
    elif prefix_to_restore is None and final_prefix is not None:
         # print("Unsetting DR_LOCAL_S3_MODEL_PREFIX in environment.")
         os.environ.pop('DR_LOCAL_S3_MODEL_PREFIX', None)
         if hasattr(settings.deepracer, 'local_s3_model_prefix'):
              # Decide if settings.deepracer.local_s3_model_prefix should also be cleared
              # settings.deepracer.local_s3_model_prefix = None # Example
              pass

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
    # Determine Run ID
    effective_run_id = run_id if run_id is not None else int(os.getenv('DR_RUN_ID', getattr(settings.deepracer, 'run_id', 0)))

    stack_name = f"deepracer-eval-{effective_run_id}"
    print(f"Stopping evaluation stack: {stack_name} (Run ID: {effective_run_id})")

    # Set DR_RUN_ID and DR_DOCKER_STYLE in environment for stop_evaluation_stack context if needed
    os.environ['DR_RUN_ID'] = str(effective_run_id)
    if not os.environ.get('DR_DOCKER_STYLE'): # Ensure docker style is set for stop logic
         os.environ['DR_DOCKER_STYLE'] = getattr(settings.docker, 'dr_docker_style', 'compose')

    initial_data = {"stack_name": stack_name}
    stop_pipeline_flow = forward[Dict[str, Any]]() >> (
        stop_evaluation_stack >>
        If(lambda data: data.get("status") == "success")
            .Then(passthrough)
        .Else(passthrough)
    )

    result = stop_pipeline_flow(initial_data)
    print("Stop evaluation pipeline execution completed.")
    return result