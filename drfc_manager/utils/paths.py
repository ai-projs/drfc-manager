import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent

INTERNAL_DIRS = {
    'config': PACKAGE_ROOT / 'config',
    'logs': PACKAGE_ROOT / 'logs',
    'tmp': PACKAGE_ROOT / 'tmp',
    'data': PACKAGE_ROOT / 'data',
    'comms': PACKAGE_ROOT / 'tmp' / 'comms',
    'docker_composes': PACKAGE_ROOT / 'config' / 'drfc-images'
}

def ensure_dir_exists(dir_path: Path) -> None:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        os.chmod(dir_path, 0o777)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directory {dir_path}: {e}")
    except Exception as e:
        raise Exception(f"Failed to create directory {dir_path}: {e}")

def get_internal_path(dir_name: str, *subpaths: str) -> Path:
    if dir_name not in INTERNAL_DIRS:
        raise ValueError(f"Unknown internal directory: {dir_name}")
    
    path = INTERNAL_DIRS[dir_name]
    if subpaths:
        path = path.joinpath(*subpaths)
    
    parent = path.parent
    while parent != PACKAGE_ROOT and parent != PACKAGE_ROOT.parent:
        try:
            parent.mkdir(parents=True, exist_ok=True)
            os.chmod(parent, 0o777)
        except PermissionError as e:
            raise PermissionError(f"Permission denied creating parent directory {parent}: {e}")
        except Exception as e:
            raise Exception(f"Failed to create parent directory {parent}: {e}")
        parent = parent.parent
    
    ensure_dir_exists(path)
    return path

def get_comms_dir(run_id: int) -> Path:
    """Get the communications directory for a specific run."""
    return get_internal_path('comms', str(run_id))

def get_logs_dir(model_prefix: str) -> Path:
    """Get the logs directory for a specific model."""
    return get_internal_path('logs', 'robomaker', model_prefix)

def get_docker_compose_path(compose_name: str) -> Path:
    """Get the path to a docker compose file. Do not create the directory, just check existence."""
    path = INTERNAL_DIRS['docker_composes'] / f'docker-compose-{compose_name}.yml'
    if not path.exists():
        raise FileNotFoundError(f"Docker compose file not found: {path}")
    return path 