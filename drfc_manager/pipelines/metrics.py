import os
import time
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

from drfc_manager.config_env import settings
from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.docker.utilities import _adjust_composes_file_names
from drfc_manager.utils.logging import logger, setup_logging


try:
    from drfc_manager.utils.paths import PACKAGE_ROOT
except ImportError:
    PACKAGE_ROOT = Path(__file__).parent.parent.parent

GRAFANA_DEFAULT_URL = "http://localhost:3000"
GRAFANA_DEFAULT_TIMEOUT = 90
GRAFANA_DEFAULT_USERNAME = "admin"
GRAFANA_DEFAULT_PASSWORD = "admin"
METRICS_STACK_NAME = "deepracer-metrics"


CONFIG_PATHS = [
    PACKAGE_ROOT / "config" / "drfc-images" / "metrics" / "configuration.env",
    Path(__file__).parent.parent.parent / "config" / "drfc-images" / "metrics" / "configuration.env",
]


@dataclass
class MetricsResult:
    """Class to represent the result of metrics operations."""
    status: str
    error: Optional[str] = None
    error_type: Optional[str] = None
    grafana_url: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    log_file: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def success(cls, **kwargs) -> 'MetricsResult':
        """Create a success result."""
        return cls(status="success", **kwargs)

    @classmethod
    def from_exception(cls, exception: Exception, log_file: Optional[str] = None) -> 'MetricsResult':
        """Create an error result from an exception."""
        return cls(
            status="error",
            error=str(exception),
            error_type=type(exception).__name__,
            log_file=log_file
        )


def get_metrics_compose_files() -> str:
    """
    Determines the Docker Compose file paths to use for metrics stack.
    
    Returns:
        str: Space-separated list of compose file paths
    """
    compose_types: List[ComposeFileType] = [ComposeFileType.METRICS]
    
    if not settings.minio.server_url:
        compose_types.append(ComposeFileType.AWS)
    
    compose_file_names = [ct.value for ct in compose_types]
    compose_file_paths = _adjust_composes_file_names(compose_file_names)
    
    separator = os.environ.get('DR_DOCKER_FILE_SEP', ' -f ')
    return separator.join(f for f in compose_file_paths if f)


def _get_grafana_config() -> Dict[str, str]:
    """
    Get Grafana configuration from environment file.
    
    Returns:
        Dict[str, str]: Grafana configuration including credentials
    """
    config_paths = [Path(p) for p in CONFIG_PATHS]
    for config_path in config_paths:
        if config_path.exists():
            config = {}
            with open(config_path) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        config[key] = value
            return config
    raise FileNotFoundError(
        "Grafana configuration file not found. Tried paths:\n" + 
        "\n".join(f"  - {p}" for p in config_paths)
    )


def _execute_docker_compose(cmd: str, quiet: bool = True) -> None:
    """
    Execute a docker compose command.
    
    Args:
        cmd (str): The docker compose command to execute
        quiet (bool): If True, suppress verbose output
    """
    if not quiet:
        logger.info(f"Executing command: {cmd}")
    os.system(cmd)


def _log_grafana_info(result: MetricsResult, quiet: bool) -> None:
    """
    Log Grafana information if not in quiet mode.
    
    Args:
        result (MetricsResult): The metrics result containing Grafana info
        quiet (bool): If True, suppress verbose output
    """
    if not quiet and result.grafana_url and result.credentials:
        logger.info("Metrics stack started successfully")
        logger.info(f"Grafana available at: {result.grafana_url}")
        logger.info("Credentials:")
        logger.info(f"  Username: {result.credentials['username']}")
        logger.info(f"  Password: {result.credentials['password']}")
        logger.info("\nThe following dashboards are available:")
        logger.info("  - DeepRacer Training template (default)")
        logger.info("    Shows training rewards, progress, lap times, and entropy")


def start_metrics_pipeline(quiet: bool = True) -> MetricsResult:
    """
    Start the metrics stack (Telegraf, InfluxDB, Grafana) and return the access URL.
    
    Args:
        quiet (bool): If True, suppress verbose output. Defaults to True.
        
    Returns:
        MetricsResult: Results of the metrics pipeline execution.
    """
    log_path = setup_logging(quiet=quiet)
    
    try:
        compose_files = get_metrics_compose_files()
        if not compose_files:
            raise ValueError("No compose files found for metrics stack")
            
        grafana_config = _get_grafana_config()
        
        cmd = f"docker compose -f {compose_files} -p {METRICS_STACK_NAME} up -d"
        _execute_docker_compose(cmd, quiet)
        
        if not _wait_for_grafana(GRAFANA_DEFAULT_URL, GRAFANA_DEFAULT_TIMEOUT):
            raise TimeoutError("Grafana failed to start within timeout period")
            
        result = MetricsResult.success(
            grafana_url=GRAFANA_DEFAULT_URL,
            credentials={
                "username": grafana_config.get("GF_SECURITY_ADMIN_USER", GRAFANA_DEFAULT_USERNAME),
                "password": grafana_config.get("GF_SECURITY_ADMIN_PASSWORD", GRAFANA_DEFAULT_PASSWORD)
            },
            log_file=log_path
        )
        
        _log_grafana_info(result, quiet)
        return result
        
    except Exception as e:
        logger.error(f"Error starting metrics stack: {type(e).__name__}: {e}")
        return MetricsResult.from_exception(e, log_path)


def stop_metrics_pipeline() -> MetricsResult:
    """
    Stop the metrics stack.
    
    Returns:
        MetricsResult: Results of the stop operation.
    """
    try:
        compose_files = get_metrics_compose_files()
        if not compose_files:
            raise ValueError("No compose files found for metrics stack")
            
        cmd = f"docker compose -f {compose_files} -p {METRICS_STACK_NAME} down"
        _execute_docker_compose(cmd)
        
        result = MetricsResult.success(message="Metrics stack stopped successfully")
        logger.info(result.message)
        return result
        
    except Exception as e:
        logger.error(f"Error stopping metrics stack: {type(e).__name__}: {e}")
        return MetricsResult.from_exception(e)


def _wait_for_grafana(url: str = GRAFANA_DEFAULT_URL, timeout: int = GRAFANA_DEFAULT_TIMEOUT) -> bool:
    """
    Wait for Grafana to be ready by checking its health endpoint.
    
    Args:
        url (str): Base URL for Grafana
        timeout (int): Maximum time to wait in seconds
        
    Returns:
        bool: True if Grafana is ready, False if timeout
    """
    start_time = time.time()
    health_url = f"{url}/api/health"
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    
    return False 