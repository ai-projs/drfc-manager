import os
from typing import List
from src.config import settings
from src.utils.str_to_bool import str2bool
# Import the enum and the utility function
from src.types.docker import ComposeFileType
from src.utils.docker.utilities import _adjust_composes_file_names
from src.utils.logging import logger

def get_compose_files() -> str:
    """
    Determines the Docker Compose file paths to use for evaluation,
    leveraging the ComposeFileType enum and utility functions.
    """
    compose_types: List[ComposeFileType] = [ComposeFileType.EVAL] # Start with base evaluation file

    # Conditionally add overlays based on config/env vars

    # S3/AWS Configuration Overlays
    if settings.minio.server_url: # Local S3 endpoint configured
        compose_types.append(ComposeFileType.ENDPOINT)
        s3_auth_mode = os.environ.get('DR_LOCAL_S3_AUTH_MODE', 'profile')
        if s3_auth_mode != 'role': # Use keys unless using role auth
            compose_types.append(ComposeFileType.KEYS)
    elif not settings.minio.server_url: # Not local, assume AWS cloud context
        # Add AWS metadata overlay if needed (e.g., for IAM role auth)
        # You might need a specific setting like settings.aws.use_iam_role to decide this
        # For now, let's assume if not local minio, we need the AWS overlay
        compose_types.append(ComposeFileType.AWS)
        # Add CloudWatch logging overlay if enabled
        if str2bool(os.environ.get('DR_CLOUD_WATCH_ENABLE', 'False')):
             compose_types.append(ComposeFileType.CWLOG)


    # Log Mounting Overlay
    mount_logs = str2bool(os.environ.get('DR_ROBOMAKER_MOUNT_LOGS', 'False'))
    if mount_logs:
        compose_types.append(ComposeFileType.MOUNT)
        model_prefix = os.environ.get('DR_LOCAL_S3_MODEL_PREFIX', 'unknown_model')
        dr_dir = os.environ.get('DRFC_REPO_ABS_PATH', '/opt/DRfC') # Use env var set by setup
        mount_dir = os.path.join(dr_dir, "data/logs/robomaker", model_prefix)
        os.environ['DR_MOUNT_DIR'] = mount_dir
        try:
            os.makedirs(mount_dir, exist_ok=True)
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
    return separator.join(f for f in compose_file_paths if f)