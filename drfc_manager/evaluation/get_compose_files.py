import os
from typing import List
from drfc_manager.config_env import settings
from drfc_manager.utils.str_to_bool import str2bool
# Import the enum and the utility function
from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.docker.utilities import _adjust_composes_file_names
from drfc_manager.utils.logging import logger
from drfc_manager.utils.paths import get_logs_dir

def get_compose_files() -> str:
    """
    Get the list of Docker Compose files needed for the current configuration.
    
    Returns:
        str: Space-separated list of Docker Compose file paths
    """
    compose_types = [ComposeFileType.EVAL]

    # Mount logs if enabled
    mount_logs = str2bool(os.environ.get('DR_ROBOMAKER_MOUNT_LOGS', 'False'))
    if mount_logs:
        compose_types.append(ComposeFileType.MOUNT)
        model_prefix = os.environ.get('DR_LOCAL_S3_MODEL_PREFIX', 'unknown_model')
        mount_dir = get_logs_dir(model_prefix)
        os.environ['DR_MOUNT_DIR'] = str(mount_dir)
        try:
            mount_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create log mount directory {mount_dir}: {e}")
    else:
         os.environ.pop('DR_MOUNT_DIR', None)

    # Host X Display Overlays
    host_x_enabled = str2bool(os.environ.get('DR_HOST_X', 'False'))
    if host_x_enabled:
        display = os.environ.get('DISPLAY')
        if not display:
            logger.warning("DR_HOST_X is true, but DISPLAY environment variable is not set.")
        else:
            is_wsl2 = 'microsoft' in os.uname().release.lower() and 'wsl2' in os.uname().release.lower()
            if is_wsl2:
                compose_types.append(ComposeFileType.XORG_WSL)
            else:
                xauthority = os.environ.get('XAUTHORITY')
                default_xauthority = os.path.expanduser("~/.Xauthority")
                if not xauthority and not os.path.exists(default_xauthority):
                    logger.warning(f"XAUTHORITY not set and {default_xauthority} does not exist. GUI may fail.")
                elif not xauthority:
                    os.environ['XAUTHORITY'] = default_xauthority
                compose_types.append(ComposeFileType.XORG)

    # Docker Style Overlay (Swarm)
    docker_style = os.environ.get('DR_DOCKER_STYLE', 'compose').lower()
    if docker_style == "swarm":
        compose_types.append(ComposeFileType.EVAL_SWARM)

    compose_file_names = [ct.value for ct in compose_types]
    compose_file_paths = _adjust_composes_file_names(compose_file_names)

    separator = getattr(settings.docker, 'dr_docker_file_sep', ' -f ')
    return separator.join(compose_file_paths)