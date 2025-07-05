import os
from typing import List
from drfc_manager.config_env import settings
from drfc_manager.types.env_vars import EnvVars

# Import the enum and the utility function
from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.docker.utilities import adjust_composes_file_names
from drfc_manager.utils.logging import logger
from drfc_manager.utils.paths import get_logs_dir

env_vars = EnvVars()

def get_compose_files() -> str:
    """
    Determines the Docker Compose file paths to use for evaluation,
    leveraging the ComposeFileType enum and utility functions.
    """
    compose_types: List[ComposeFileType] = [ComposeFileType.EVAL]

    if settings.minio.server_url:
        compose_types.append(ComposeFileType.ENDPOINT)
        s3_auth_mode = env_vars.DR_LOCAL_S3_AUTH_MODE
        if s3_auth_mode != "role":
            compose_types.append(ComposeFileType.KEYS)
    elif not settings.minio.server_url:
        compose_types.append(ComposeFileType.AWS)
        if env_vars.DR_CLOUD_WATCH_ENABLE:
            compose_types.append(ComposeFileType.CWLOG)

    if env_vars.DR_ROBOMAKER_MOUNT_LOGS:
        compose_types.append(ComposeFileType.MOUNT)
        model_prefix = env_vars.DR_LOCAL_S3_MODEL_PREFIX
        mount_dir = str(get_logs_dir(model_prefix))
        env_vars.update(DR_MOUNT_DIR=mount_dir)
        env_vars.load_to_environment()
        try:
            os.makedirs(mount_dir, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create log mount directory {mount_dir}: {e}")
    else:
        env_vars.update(DR_MOUNT_DIR=None)
        env_vars.load_to_environment()

    # Host X Display Overlays
    if env_vars.DR_HOST_X:
        display = env_vars.DR_DISPLAY
        if not display:
            logger.warning(
                "DR_HOST_X is true, but DISPLAY environment variable is not set."
            )
        else:
            is_wsl2 = (
                "microsoft" in os.uname().release.lower()
                and "wsl2" in os.uname().release.lower()
            )
            if is_wsl2:
                compose_types.append(ComposeFileType.XORG_WSL)
            else:
                xauthority = env_vars.DR_XAUTHORITY
                default_xauthority = os.path.expanduser("~/.Xauthority")
                if not xauthority and not os.path.exists(default_xauthority):
                    logger.warning(
                        f"XAUTHORITY not set and {default_xauthority} does not exist. GUI may fail."
                    )
                elif not xauthority:
                    env_vars.update(DR_XAUTHORITY=default_xauthority)
                    env_vars.load_to_environment()
                compose_types.append(ComposeFileType.XORG)

    docker_style = env_vars.DR_DOCKER_STYLE.lower()
    if docker_style == "swarm":
        compose_types.append(ComposeFileType.EVAL_SWARM)

    compose_file_names = [ct.value for ct in compose_types]
    compose_file_paths = adjust_composes_file_names(compose_file_names)

    
    separator = settings.docker.dr_docker_file_sep 
    return separator.join(f for f in compose_file_paths if f)
