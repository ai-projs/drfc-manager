from datetime import datetime
import subprocess
import time
import json
import socket
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from gloe import transformer
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.utils.logging_config import get_logger, configure_logging
from drfc_manager.utils.env_utils import get_subprocess_env

env_vars = EnvVars()
logger = get_logger(__name__)

# Default configuration
DEFAULT_TOPIC = "/racecar/main_camera/camera_link/camera_sensor/image"
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360
DEFAULT_QUALITY = 75
DEFAULT_VIEWER_PORT = 8100
DEFAULT_PROXY_PORT = 8090
DEFAULT_DELAY = 5
MAX_PORT_ATTEMPTS = 10

# Process patterns for finding and killing processes
STREAMLIT_PROCESS_PATTERN = "streamlit run drfc_manager.viewers.streamlit_viewer:app"
UVICORN_PROCESS_PATTERN = "uvicorn drfc_manager.viewers.stream_proxy:app"
PROXY_LOG_BASENAME = "stream_proxy"
STREAMLIT_LOG_BASENAME = "streamlit_viewer"

# Use environment variable for log directory or fall back to user's home directory
log_dir = os.environ.get('DRFC_LOG_DIR', os.path.expanduser('~/drfc_logs'))
log_file_name = f"{log_dir}/viewer_{env_vars.DR_RUN_ID}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
configure_logging(log_file=log_file_name)


@dataclass
class ViewerConfig:
    run_id: int
    topic: str = DEFAULT_TOPIC
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    quality: int = DEFAULT_QUALITY
    port: int = DEFAULT_VIEWER_PORT
    proxy_port: int = DEFAULT_PROXY_PORT

    def update_environment(self, containers: List[str]) -> None:
        """Update environment variables with viewer configuration."""
        env_vars.update(
            DR_RUN_ID=self.run_id,
            DR_LOCAL_S3_MODEL_PREFIX=env_vars.DR_LOCAL_S3_MODEL_PREFIX,
            DR_VIEWER_CONTAINERS=json.dumps(containers),
            DR_VIEWER_QUALITY=self.quality,
            DR_VIEWER_WIDTH=self.width,
            DR_VIEWER_HEIGHT=self.height,
            DR_VIEWER_TOPIC=self.topic,
            DR_DYNAMIC_PROXY_PORT=self.proxy_port
        )
        env_vars.load_to_environment()


def _find_available_port(
    start_port: int, host: str = "0.0.0.0", max_attempts: int = MAX_PORT_ATTEMPTS
) -> Optional[int]:
    for attempt in range(max_attempts):
        port = start_port + attempt
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                logger.debug(f"Port {port} is available.")
                return port
        except socket.error as e:
            logger.debug(
                f"Port {port} is in use (Attempt {attempt + 1}/{max_attempts}): {e}"
            )
            if attempt == max_attempts - 1:
                logger.error(
                    f"Could not find an available port after {max_attempts} attempts starting from {start_port}."
                )
                return None
    return None


def _check_pid_exists(pid: int) -> bool:
    try:
        env_vars.load_to_environment()
        env = get_subprocess_env(env_vars)
        subprocess.run(
            ["kill", "-0", str(pid)],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env
        )
        return True
    except (subprocess.CalledProcessError, ValueError):
        return False
    except FileNotFoundError:
        logger.warning("Could not find 'kill' command to check process existence.")
        return False


def _create_wait_for_containers(delay: int):
    @transformer
    def wait_for_containers(data: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for the containers to initialize before proceeding."""
        if delay > 0:
            logger.info(f"Waiting {delay} seconds for containers to initialize...")
            time.sleep(delay)
        return data

    return wait_for_containers


def wait_for_containers(delay: int):
    """Factory function to create a transformer that waits for containers to initialize."""
    return _create_wait_for_containers(delay)


def _kill_processes_by_pattern(pattern: str) -> Tuple[bool, List[str]]:
    killed_pids = []
    errors = []
    success = True
    try:
        pgrep_cmd = ["pgrep", "-f", pattern]
        env_vars.load_to_environment()
        env = get_subprocess_env(env_vars)
        result = subprocess.run(pgrep_cmd, capture_output=True, text=True, check=False, env=env)

        if result.returncode == 0 and result.stdout:
            pids = result.stdout.strip().split("\n")
            logger.info(
                f"Found {len(pids)} process(es) matching pattern '{pattern}': {', '.join(pids)}"
            )
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    logger.info(f"Sending SIGTERM to PID {pid}...")
                    kill_cmd_term = ["kill", str(pid)]
                    subprocess.run(
                        kill_cmd_term,
                        check=False,
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        env=env
                    )

                    time.sleep(1.0)

                    if _check_pid_exists(pid):
                        logger.warning(
                            f"PID {pid} still exists after SIGTERM. Sending SIGKILL..."
                        )
                        kill_cmd_kill = ["kill", "-9", str(pid)]
                        subprocess.run(
                            kill_cmd_kill, check=True, capture_output=True, text=True, env=env
                        )
                        logger.info(f"Successfully sent SIGKILL to PID {pid}.")
                        time.sleep(0.2)
                        if _check_pid_exists(pid):
                            logger.error(f"PID {pid} STILL exists even after SIGKILL!")
                            errors.append(f"PID {pid} could not be terminated.")
                            success = False
                        else:
                            killed_pids.append(pid_str)
                    else:
                        logger.info(f"PID {pid} terminated successfully after SIGTERM.")
                        killed_pids.append(pid_str)

                except ValueError:
                    logger.warning(
                        f"Invalid PID '{pid_str}' found for pattern '{pattern}'."
                    )
                except subprocess.CalledProcessError as kill_err:
                    err_msg = f"Failed to send SIGKILL to PID {pid_str}: {kill_err.stderr.strip()}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                    success = False
                except Exception as e:
                    err_msg = f"Unexpected error killing PID {pid_str}: {e}"
                    logger.error(err_msg, exc_info=True)
                    errors.append(err_msg)
                    success = False
        elif result.returncode == 1:
            logger.info(f"No processes found matching pattern '{pattern}'.")
        else:
            err_msg = (
                f"Error running pgrep for pattern '{pattern}': {result.stderr.strip()}"
            )
            logger.error(err_msg)
            errors.append(err_msg)
            success = False

    except FileNotFoundError:
        err_msg = "'pgrep' or 'kill' command not found. Cannot reliably kill processes."
        logger.error(err_msg)
        errors.append(err_msg)
        success = False
    except Exception as e:
        err_msg = f"Error finding/killing processes with pattern '{pattern}': {e}"
        logger.error(err_msg, exc_info=True)
        errors.append(err_msg)
        success = False

    return success, errors


@transformer
def get_robomaker_containers(config: ViewerConfig) -> Dict[str, Any]:
    logger.info(
        f"Attempting to find Robomaker containers for run {config.run_id} (Docker style: {env_vars.DR_DOCKER_STYLE})"
    )
    containers = []
    try:
        env_vars.load_to_environment()
        env = get_subprocess_env(env_vars)
        if env_vars.DR_DOCKER_STYLE.lower() != "swarm":
            cmd = [
                "docker",
                "ps",
                "--format",
                "{{.ID}}",
                "--filter",
                f"name=deepracer-{config.run_id}",
                "--filter",
                "name=robomaker",
            ]
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=15, env=env
            )
            containers = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            logger.info(
                f"Found {len(containers)} Robomaker container IDs: {containers}"
            )
        else:
            service_name = f"deepracer-{config.run_id}_robomaker"
            cmd = [
                "docker",
                "service",
                "ps",
                service_name,
                "--format",
                "{{.ID}}",
                "--filter",
                "desired-state=running",
            ]
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=15, env=env
            )
            task_ids = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            logger.info(
                f"Found {len(task_ids)} running Robomaker task IDs for service {service_name}: {task_ids}"
            )

            for task_id in task_ids:
                ip_cmd = [
                    "docker",
                    "inspect",
                    task_id,
                    "--format",
                    '{{range .NetworksAttachments}}{{if eq .Network.Spec.Name "sagemaker-local"}}{{range .Addresses}}{{index (split . "/") 0}}{{end}}{{end}}{{end}}',
                ]
                logger.debug(f"Running command: {' '.join(ip_cmd)}")
                ip_result = subprocess.run(
                    ip_cmd, check=True, capture_output=True, text=True, timeout=10, env=env
                )
                ip_address = ip_result.stdout.strip()
                if ip_address:
                    logger.debug(f"Found IP {ip_address} for task {task_id}")
                    containers.append(ip_address)
                else:
                    logger.warning(
                        f"Could not find IP address on 'sagemaker-local' network for task {task_id}"
                    )

            logger.info(
                f"Found {len(containers)} Robomaker container IPs on Swarm: {containers}"
            )

        if not containers:
            logger.warning(
                f"No running Robomaker containers found for run {config.run_id}."
            )

        return {"status": "success", "containers": containers, "config": config}
    except subprocess.TimeoutExpired as e:
        logger.error(f"TimeoutExpired while finding containers: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Timeout while running command: {e.cmd}",
            "type": "TimeoutExpired",
        }
    except subprocess.CalledProcessError as e:
        logger.error(
            f"CalledProcessError finding containers: {e.stderr}", exc_info=True
        )
        return {
            "status": "error",
            "error": f"Command '{e.cmd}' failed: {e.stderr}",
            "type": "CalledProcessError",
        }
    except Exception as e:
        logger.error(f"Unexpected error finding containers: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "type": type(e).__name__}


@transformer
def start_stream_proxy(data: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Attempting to start stream proxy...")
    if data.get("status") != "success":
        logger.warning("Skipping proxy start due to previous step failure.")
        return data

    config: Optional[ViewerConfig] = data.get("config")
    containers: List[str] = data.get("containers", [])

    if not config:
        return {"status": "error", "error": "ViewerConfig missing in input data"}

    logger.info(
        f"Checking for and stopping existing processes matching: '{UVICORN_PROCESS_PATTERN}'"
    )
    kill_success, kill_errors = _kill_processes_by_pattern(UVICORN_PROCESS_PATTERN)
    if not kill_success:
        logger.warning(
            f"Issues encountered while trying to kill existing proxy processes: {kill_errors}"
        )

    logger.info(f"Finding available port starting from {config.proxy_port}...")
    available_port = _find_available_port(config.proxy_port)
    if available_port is None:
        return {
            "status": "error",
            "error": f"Could not find available port for proxy near {config.proxy_port}",
        }
    if available_port != config.proxy_port:
        logger.info(
            f"Using port {available_port} for proxy (original {config.proxy_port} was busy)."
        )
        config.proxy_port = available_port

    env_vars.update(DR_DYNAMIC_PROXY_PORT=config.proxy_port)
    env_vars.update(DR_VIEWER_CONTAINERS=json.dumps(containers))
    env_vars.load_to_environment()
    logger.info(f"Environment variables updated: {env_vars}")
    
    proxy_script = Path(__file__).parent.parent / "viewers" / "stream_proxy.py"
    if not proxy_script.exists():
        logger.error(f"Stream proxy script not found at {proxy_script}")
        return {
            "status": "error",
            "error": f"Stream proxy script not found: {proxy_script}",
        }

    cmd = [
        "uvicorn",
        "drfc_manager.viewers.stream_proxy:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(config.proxy_port),
        "--workers",
        "1",
    ]

    process = None
    try:
        logger.info(f"Starting proxy server process: {' '.join(cmd)}")
        env = get_subprocess_env(env_vars)
        env["DR_VIEWER_CONTAINERS"] = json.dumps(containers)
        logger.info(f"Environment variables for proxy process: {env}") 
        log_file = open(log_file_name, "w")
 
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            text=True,
            env=env
        )

        time.sleep(2)
        if process.poll() is None:
            proxy_url = f"http://localhost:{config.proxy_port}"
            logger.info(
                f"Stream proxy server started successfully (PID: {process.pid}) at {proxy_url}"
            )
            data["proxy_url"] = proxy_url
            data["proxy_pid"] = process.pid
            data["config"] = config
            return data
        else:
            stdout, stderr = process.communicate()
            logger.error(
                f"Stream proxy server failed to start. Process exited with code {process.poll()}."
            )
            logger.error(f"Proxy STDOUT: {stdout[:500]}")
            logger.error(f"Proxy STDERR: {stderr[:500]}")
            return {
                "status": "error",
                "error": "Proxy server failed to start",
                "exit_code": process.poll(),
            }

    except Exception as e:
        logger.error(f"Failed to start stream proxy process: {e}", exc_info=True)
        if process and process.poll() is None:
            process.terminate()
        return {
            "status": "error",
            "error": f"Exception starting proxy: {str(e)}",
            "type": type(e).__name__,
        }


@transformer
def start_streamlit_viewer(data: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Attempting to start Streamlit viewer...")
    if data.get("status") != "success":
        logger.warning("Skipping Streamlit viewer start due to previous step failure.")
        return data

    config: Optional[ViewerConfig] = data.get("config")
    containers: List[str] = data.get("containers", [])
    logger.info("Containers at final step: %s", containers)
    proxy_url: str = data.get("proxy_url", "")

    if not config:
        return {"status": "error", "error": "ViewerConfig missing in input data"}
    if not proxy_url:
        logger.warning(
            "Proxy URL not found in input data, Streamlit might not function correctly."
        )

    logger.info(
        f"Checking for and stopping existing processes matching: '{STREAMLIT_PROCESS_PATTERN}'"
    )
    kill_success, kill_errors = _kill_processes_by_pattern(STREAMLIT_PROCESS_PATTERN)
    if not kill_success:
        logger.warning(
            f"Issues encountered while trying to kill existing Streamlit processes: {kill_errors}"
        )

    logger.info(f"Finding available port starting from {config.port}...")
    available_port = _find_available_port(config.port)
    if available_port is None:
        return {
            "status": "error",
            "error": f"Could not find available port for Streamlit near {config.port}",
        }
    if available_port != config.port:
        logger.info(
            f"Using port {available_port} for Streamlit (original {config.port} was busy)."
        )
        config.port = available_port

    viewer_script = Path(__file__).parent.parent / "viewers" / "streamlit_viewer.py"
    if not viewer_script.exists():
        logger.error(f"Streamlit viewer script not found at {viewer_script}")
        return {
            "status": "error",
            "error": f"Streamlit viewer script not found: {viewer_script}",
        }
    
    config.update_environment(containers)

    cmd = [
        "streamlit",
        "run",
        str(viewer_script),
        "--server.port",
        str(config.port),
        "--server.headless",
        "true",
    ]

    process = None
    try:
        logger.info(f"Starting Streamlit viewer process: {' '.join(cmd)}")
        env = get_subprocess_env(env_vars)
        env["DR_VIEWER_CONTAINERS"] = json.dumps(containers)
        log_file = open(log_file_name, "w")
        process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env, text=True)

        time.sleep(4)
        if process.poll() is None:
            streamlit_url = f"http://localhost:{config.port}"
            logger.info(
                f"Streamlit viewer started successfully (PID: {process.pid}) at {streamlit_url}"
            )
            return {
                "status": "success",
                "message": "Streamlit viewer started successfully.",
                "viewer_url": streamlit_url,
                "viewer_pid": process.pid,
                "proxy_url": proxy_url,
                "proxy_pid": data.get("proxy_pid"),
            }
        else:
            stdout, stderr = process.communicate()
            logger.error(
                f"Streamlit viewer failed to start. Process exited with code {process.poll()}."
            )
            logger.error(f"Streamlit STDOUT: {stdout[:500]}")
            logger.error(f"Streamlit STDERR: {stderr[:500]}")
            return {
                "status": "error",
                "error": "Streamlit viewer failed to start",
                "exit_code": process.poll(),
            }

    except Exception as exc:
        err_msg = str(exc)
        err_type = type(exc).__name__
        logger.error(
            f"Failed to start Streamlit viewer process: {err_msg}", exc_info=True
        )
        if process and process.poll() is None:
            process.terminate()
        return {
            "status": "error",
            "error": f"Exception starting Streamlit: {err_msg}",
            "type": err_type,
        }


@transformer
def stop_viewer_process(_) -> Dict[str, Any]:
    logger.info("Attempting to stop viewer processes...")
    all_success = True
    all_errors = []

    logger.info(f"Stopping processes matching: '{STREAMLIT_PROCESS_PATTERN}'")
    streamlit_success, streamlit_errors = _kill_processes_by_pattern(
        STREAMLIT_PROCESS_PATTERN
    )
    if not streamlit_success:
        all_success = False
        all_errors.extend(streamlit_errors)
        logger.warning("Issues encountered stopping Streamlit processes.")
    else:
        logger.info("Streamlit processes stopped (or none were running).")

    logger.info(f"Stopping processes matching: '{UVICORN_PROCESS_PATTERN}'")
    proxy_success, proxy_errors = _kill_processes_by_pattern(UVICORN_PROCESS_PATTERN)
    if not proxy_success:
        all_success = False
        all_errors.extend(proxy_errors)
        logger.warning("Issues encountered stopping proxy processes.")
    else:
        logger.info("Proxy processes stopped (or none were running).")

    if all_success:
        logger.info("All targeted viewer processes stopped successfully.")
        return {"status": "success", "message": "Viewer and proxy processes stopped."}
    else:
        logger.error(
            f"Failed to cleanly stop all viewer processes. Errors: {all_errors}"
        )
        return {
            "status": "error",
            "error": "Failed to stop one or more viewer processes",
            "details": all_errors,
        }


def start_viewer_pipeline(
    update: bool = False,
    port: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: Optional[int] = None,
    topic: Optional[str] = None,
    proxy_port: Optional[int] = None,
    delay: int = DEFAULT_DELAY,
) -> Dict[str, Any]:
    """Start the viewer pipeline.

    Args:
        update: Whether to update an existing viewer.
        port: The Streamlit viewer port (default: 8100).
        width: The stream width in pixels (default: 480).
        height: The stream height in pixels (default: 360).
        quality: The stream quality (1-100) (default: 75).
        topic: The ROS topic to stream (default: /racecar/deepracer/kvs_stream).
        proxy_port: The Stream Proxy port (default: 8090).
        delay: Seconds to wait for RoboMaker to start before starting viewer (default: 5).

    Returns:
        Dict with pipeline outcome.
    """
    run_id = env_vars.DR_RUN_ID
    env_vars.load_to_environment()

    # Use provided values or defaults
    config = ViewerConfig(
        run_id=run_id,
        topic=topic or DEFAULT_TOPIC,
        width=width or DEFAULT_WIDTH,
        height=height or DEFAULT_HEIGHT,
        quality=quality or DEFAULT_QUALITY,
        port=port or DEFAULT_VIEWER_PORT,
        proxy_port=proxy_port or DEFAULT_PROXY_PORT,
    )

    stop_viewer_process(None)

    # Build pipeline
    robomaker_containers = get_robomaker_containers
    pipeline = (
        robomaker_containers
        >> wait_for_containers(delay)
        >> start_stream_proxy
        >> start_streamlit_viewer
    )

    logger.info(f"Starting viewer pipeline for Run ID: {run_id}")
    result = pipeline(config)
    logger.info("Viewer pipeline complete.")

    return result


def stop_viewer_pipeline() -> Dict[str, Any]:
    """
    Stop the viewer pipeline and kill associated processes.

    Returns:
        Dict with pipeline outcome
    """
    env_vars.load_to_environment()

    logger.info("Stopping DeepRacer Viewer Pipeline")
    try:
        result = stop_viewer_process(None)
        logger.info("Viewer pipeline stopped.")
        return result
    except Exception as e:
        logger.error("Error stopping viewer pipeline", exc_info=True)
        return {"status": "error", "error": str(e), "type": type(e).__name__}
