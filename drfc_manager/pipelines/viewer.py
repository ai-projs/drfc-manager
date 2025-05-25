import os
import subprocess
import time
import json
import logging
import socket
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from gloe import transformer

from drfc_manager.config_env import settings

DEFAULT_VIEWER_PORT = 8100
DEFAULT_PROXY_PORT = 8090
DEFAULT_TOPIC = "/racecar/deepracer/kvs_stream"
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360
DEFAULT_QUALITY = 75
DEFAULT_DELAY = 5
MAX_PORT_ATTEMPTS = 20
STREAMLIT_PROCESS_PATTERN = "streamlit run drfc_manager.viewers.streamlit_viewer:app"
UVICORN_PROCESS_PATTERN = "uvicorn drfc_manager.viewers.stream_proxy:app"
TEMP_DIR = Path(tempfile.gettempdir())
LOG_FILE = TEMP_DIR / "viewer_pipeline.log"
PROXY_LOG_BASENAME = "stream_proxy"
STREAMLIT_LOG_BASENAME = "streamlit_viewer"

log_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

logger = logging.getLogger('viewer_pipeline')
logger.setLevel(logging.INFO)
if os.environ.get('DRFC_DEBUG', 'false').lower() == 'true':
    logger.setLevel(logging.DEBUG)

handlers = []

console_logging = os.environ.get('DRFC_CONSOLE_LOGGING', 'false').lower() == 'true'

def get_user_tmp_dir():
    user_tmp = Path(tempfile.gettempdir()) / os.environ.get('USER', 'unknown_user')
    try:
        user_tmp.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Could not create user temp directory {user_tmp}: {e}")
    return user_tmp

try:
    USER_TMP = get_user_tmp_dir()
    VIEWER_LOG = USER_TMP / "viewer_pipeline.log"
    log_file_path = Path(VIEWER_LOG)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(log_formatter)
    handlers.append(file_handler)
    
    if console_logging:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        handlers.append(stream_handler)
        
    if handlers:
        logger.info(f"Viewer pipeline logging configured to file: {log_file_path}")
except OSError as e:
    logger.warning(f"Could not open log file {LOG_FILE}. Logging might be incomplete. Error: {e}")
    # Only add fallback console logging if explicitly requested or no handlers
    if console_logging or not handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        handlers.append(stream_handler)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

for handler in handlers:
    logger.addHandler(handler)
logger.propagate = False


@dataclass
class ViewerConfig:
    run_id: int
    topic: str = DEFAULT_TOPIC
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    quality: int = DEFAULT_QUALITY
    port: int = DEFAULT_VIEWER_PORT
    proxy_port: int = DEFAULT_PROXY_PORT


def _find_available_port(start_port: int, host: str = '0.0.0.0', max_attempts: int = MAX_PORT_ATTEMPTS) -> Optional[int]:
    for attempt in range(max_attempts):
        port = start_port + attempt
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                logger.debug(f"Port {port} is available.")
                return port
        except socket.error as e:
            logger.debug(f"Port {port} is in use (Attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                logger.error(f"Could not find an available port after {max_attempts} attempts starting from {start_port}.")
                return None
    return None

def _check_pid_exists(pid: int) -> bool:
    try:
        subprocess.run(["kill", "-0", str(pid)], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
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
        result = subprocess.run(pgrep_cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0 and result.stdout:
            pids = result.stdout.strip().split('\n')
            logger.info(f"Found {len(pids)} process(es) matching pattern '{pattern}': {', '.join(pids)}")
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    logger.info(f"Sending SIGTERM to PID {pid}...")
                    kill_cmd_term = ["kill", str(pid)]
                    subprocess.run(kill_cmd_term, check=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

                    time.sleep(1.0)

                    if _check_pid_exists(pid):
                        logger.warning(f"PID {pid} still exists after SIGTERM. Sending SIGKILL...")
                        kill_cmd_kill = ["kill", "-9", str(pid)]
                        kill_result = subprocess.run(kill_cmd_kill, check=True, capture_output=True, text=True)
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
                    logger.warning(f"Invalid PID '{pid_str}' found for pattern '{pattern}'.")
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
            err_msg = f"Error running pgrep for pattern '{pattern}': {result.stderr.strip()}"
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
    logger.info(f"Attempting to find Robomaker containers for run {config.run_id} (Docker style: {settings.deepracer.docker_style})")
    containers = []
    try:
        if settings.deepracer.docker_style.lower() != "swarm":
            cmd = [
                "docker", "ps", "--format", "{{.ID}}",
                "--filter", f"name=deepracer-{config.run_id}",
                "--filter", "name=robomaker"
            ]
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
            containers = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            logger.info(f"Found {len(containers)} Robomaker container IDs: {containers}")
        else:
            service_name = f"deepracer-{config.run_id}_robomaker"
            cmd = ["docker", "service", "ps", service_name, "--format", "{{.ID}}", "--filter", "desired-state=running"]
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
            task_ids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            logger.info(f"Found {len(task_ids)} running Robomaker task IDs for service {service_name}: {task_ids}")

            for task_id in task_ids:
                ip_cmd = [
                    "docker", "inspect", task_id,
                    "--format", '{{range .NetworksAttachments}}{{if eq .Network.Spec.Name "sagemaker-local"}}{{range .Addresses}}{{split . "/" 0}}{{end}}{{end}}{{end}}'
                ]
                logger.debug(f"Running command: {' '.join(ip_cmd)}")
                ip_result = subprocess.run(ip_cmd, check=True, capture_output=True, text=True, timeout=10)
                ip_address = ip_result.stdout.strip()
                if ip_address:
                    logger.debug(f"Found IP {ip_address} for task {task_id}")
                    containers.append(ip_address)
                else:
                    logger.warning(f"Could not find IP address on 'sagemaker-local' network for task {task_id}")

            logger.info(f"Found {len(containers)} Robomaker container IPs on Swarm: {containers}")

        if not containers:
            logger.warning(f"No running Robomaker containers found for run {config.run_id}.")

        return {
            "status": "success",
            "containers": containers,
            "config": config
        }
    except subprocess.TimeoutExpired as e:
        logger.error(f"TimeoutExpired while finding containers: {e}", exc_info=True)
        return {"status": "error", "error": f"Timeout while running command: {e.cmd}", "type": "TimeoutExpired"}
    except subprocess.CalledProcessError as e:
        logger.error(f"CalledProcessError finding containers: {e.stderr}", exc_info=True)
        return {"status": "error", "error": f"Command '{e.cmd}' failed: {e.stderr}", "type": "CalledProcessError"}
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

    logger.info(f"Checking for and stopping existing processes matching: '{UVICORN_PROCESS_PATTERN}'")
    kill_success, kill_errors = _kill_processes_by_pattern(UVICORN_PROCESS_PATTERN)
    if not kill_success:
        logger.warning(f"Issues encountered while trying to kill existing proxy processes: {kill_errors}")

    logger.info(f"Finding available port starting from {config.proxy_port}...")
    available_port = _find_available_port(config.proxy_port)
    if available_port is None:
        return {"status": "error", "error": f"Could not find available port for proxy near {config.proxy_port}"}
    if available_port != config.proxy_port:
         logger.info(f"Using port {available_port} for proxy (original {config.proxy_port} was busy).")
         config.proxy_port = available_port

    proxy_script = Path(__file__).parent.parent / "viewers" / "stream_proxy.py"
    if not proxy_script.exists():
        logger.error(f"Stream proxy script not found at {proxy_script}")
        return {"status": "error", "error": f"Stream proxy script not found: {proxy_script}"}

    env = os.environ.copy()
    env['DR_VIEWER_CONTAINERS'] = json.dumps(containers)
    env['DR_PROXY_PORT'] = str(config.proxy_port)

    cmd = ["uvicorn", "drfc_manager.viewers.stream_proxy:app", "--host", "0.0.0.0", "--port", str(config.proxy_port), "--workers", "1"]

    USER_TMP = Path(tempfile.gettempdir()) / os.environ.get('USER', 'unknown_user')
    USER_TMP.mkdir(parents=True, exist_ok=True)
    proxy_stdout_log = USER_TMP / f"{PROXY_LOG_BASENAME}_{config.proxy_port}_stdout.log"
    proxy_stderr_log = USER_TMP / f"{PROXY_LOG_BASENAME}_{config.proxy_port}_stderr.log"
    logger.info(f"Proxy logs will be written to: {proxy_stdout_log} and {proxy_stderr_log}")

    process = None
    stdout_file = None
    stderr_file = None
    try:
        stdout_file = open(proxy_stdout_log, 'w')
        stderr_file = open(proxy_stderr_log, 'w')
        logger.info(f"Starting proxy server process: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
        )

        time.sleep(2)
        if process.poll() is None:
            proxy_url = f"http://localhost:{config.proxy_port}"
            logger.info(f"Stream proxy server started successfully (PID: {process.pid}) at {proxy_url}")
            data["proxy_url"] = proxy_url
            data["proxy_pid"] = process.pid
            data["config"] = config
            data["proxy_log_handles"] = (stdout_file, stderr_file)
            return data
        else:
            logger.error(f"Stream proxy server failed to start. Process exited with code {process.poll()}.")
            stdout_file.close()
            stderr_file.close()
            stdout_content = proxy_stdout_log.read_text(errors='ignore')
            stderr_content = proxy_stderr_log.read_text(errors='ignore')
            logger.error(f"Proxy STDOUT: {stdout_content[:500]}")
            logger.error(f"Proxy STDERR: {stderr_content[:500]}")
            return {"status": "error", "error": "Proxy server failed to start", "exit_code": process.poll()}

    except Exception as e:
        logger.error(f"Failed to start stream proxy process: {e}", exc_info=True)
        if process and process.poll() is None:
            process.terminate()
        if stdout_file: stdout_file.close()
        if stderr_file: stderr_file.close()
        return {"status": "error", "error": f"Exception starting proxy: {str(e)}", "type": type(e).__name__}

@transformer
def start_streamlit_viewer(data: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Attempting to start Streamlit viewer...")
    if data.get("status") != "success":
        logger.warning("Skipping Streamlit viewer start due to previous step failure.")
        return data

    config: Optional[ViewerConfig] = data.get("config")
    containers: List[str] = data.get("containers", [])
    proxy_url: str = data.get("proxy_url", "")

    if not config:
        return {"status": "error", "error": "ViewerConfig missing in input data"}
    if not proxy_url:
         logger.warning("Proxy URL not found in input data, Streamlit might not function correctly.")

    logger.info(f"Checking for and stopping existing processes matching: '{STREAMLIT_PROCESS_PATTERN}'")
    kill_success, kill_errors = _kill_processes_by_pattern(STREAMLIT_PROCESS_PATTERN)
    if not kill_success:
        logger.warning(f"Issues encountered while trying to kill existing Streamlit processes: {kill_errors}")

    logger.info(f"Finding available port starting from {config.port}...")
    available_port = _find_available_port(config.port)
    if available_port is None:
        return {"status": "error", "error": f"Could not find available port for Streamlit near {config.port}"}
    if available_port != config.port:
        logger.info(f"Using port {available_port} for Streamlit (original {config.port} was busy).")
        config.port = available_port

    viewer_script = Path(__file__).parent.parent / "viewers" / "streamlit_viewer.py"
    if not viewer_script.exists():
         logger.error(f"Streamlit viewer script not found at {viewer_script}")
         return {"status": "error", "error": f"Streamlit viewer script not found: {viewer_script}"}

    env = os.environ.copy()
    env['DR_RUN_ID'] = str(config.run_id)
    env['DR_LOCAL_S3_MODEL_PREFIX'] = settings.deepracer.local_s3_model_prefix
    env['DR_VIEWER_CONTAINERS'] = json.dumps(containers)
    env['DR_VIEWER_QUALITY'] = str(config.quality)
    env['DR_VIEWER_WIDTH'] = str(config.width)
    env['DR_VIEWER_HEIGHT'] = str(config.height)
    env['DR_VIEWER_TOPIC'] = config.topic
    env['DR_PROXY_PORT'] = str(config.proxy_port)

    cmd = ["streamlit", "run", str(viewer_script), "--server.port", str(config.port), "--server.headless", "true"]

    USER_TMP = Path(tempfile.gettempdir()) / os.environ.get('USER', 'unknown_user')
    USER_TMP.mkdir(parents=True, exist_ok=True)
    streamlit_stdout_log = USER_TMP / f"{STREAMLIT_LOG_BASENAME}_{config.port}_stdout.log"
    streamlit_stderr_log = USER_TMP / f"{STREAMLIT_LOG_BASENAME}_{config.port}_stderr.log"
    logger.info(f"Streamlit logs will be written to: {streamlit_stdout_log} and {streamlit_stderr_log}")

    process = None
    stdout_file = None
    stderr_file = None
    try:
        stdout_file = open(streamlit_stdout_log, 'w')
        stderr_file = open(streamlit_stderr_log, 'w')
        logger.info(f"Starting Streamlit viewer process: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env
        )

        time.sleep(4)
        if process.poll() is None:
            streamlit_url = f"http://localhost:{config.port}"
            logger.info(f"Streamlit viewer started successfully (PID: {process.pid}) at {streamlit_url}")
            if "proxy_log_handles" in data:
                try:
                     data["proxy_log_handles"][0].close()
                     data["proxy_log_handles"][1].close()
                except Exception as e:
                     logger.warning(f"Could not close proxy log handles: {e}")

            return {
                "status": "success",
                "message": f"Streamlit viewer started successfully.",
                "viewer_url": streamlit_url,
                "viewer_pid": process.pid,
                "proxy_url": proxy_url,
                "proxy_pid": data.get("proxy_pid")
            }
        else:
            logger.error(f"Streamlit viewer failed to start. Process exited with code {process.poll()}.")
            stdout_file.close()
            stderr_file.close()
            stdout_content = streamlit_stdout_log.read_text(errors='ignore')
            stderr_content = streamlit_stderr_log.read_text(errors='ignore')
            logger.error(f"Streamlit STDOUT: {stdout_content[:500]}")
            logger.error(f"Streamlit STDERR: {stderr_content[:500]}")
            if "proxy_log_handles" in data:
                 try:
                     data["proxy_log_handles"][0].close()
                     data["proxy_log_handles"][1].close()
                 except Exception as e: logger.warning(f"Could not close proxy log handles on streamlit failure: {e}")
            return {"status": "error", "error": "Streamlit viewer failed to start", "exit_code": process.poll()}

    except Exception as e:
        logger.error(f"Failed to start Streamlit viewer process: {e}", exc_info=True)
        if process and process.poll() is None:
            process.terminate()
        if stdout_file: stdout_file.close()
        if stderr_file: stderr_file.close()
        if "proxy_log_handles" in data:
            try:
                data["proxy_log_handles"][0].close()
                data["proxy_log_handles"][1].close()
            except Exception as e: logger.warning(f"Could not close proxy log handles on streamlit exception: {e}")
        return {"status": "error", "error": f"Exception starting Streamlit: {str(e)}", "type": type(e).__name__}


@transformer
def stop_viewer_process(_) -> Dict[str, Any]:
    logger.info("Attempting to stop viewer processes...")
    all_success = True
    all_errors = []

    logger.info(f"Stopping processes matching: '{STREAMLIT_PROCESS_PATTERN}'")
    streamlit_success, streamlit_errors = _kill_processes_by_pattern(STREAMLIT_PROCESS_PATTERN)
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
        logger.error(f"Failed to cleanly stop all viewer processes. Errors: {all_errors}")
        return {"status": "error", "error": "Failed to stop one or more viewer processes", "details": all_errors}


def start_viewer_pipeline(
    update: bool = False,
    port: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: Optional[int] = None,
    topic: Optional[str] = None,
    proxy_port: Optional[int] = None,
    delay: int = DEFAULT_DELAY,
    quiet: bool = True
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
        quiet: If True, suppress console logging (default: True).
        
    Returns:
        Dict with pipeline outcome.
    """
    if quiet:
        os.environ['DRFC_CONSOLE_LOGGING'] = 'false'
    else:
        os.environ['DRFC_CONSOLE_LOGGING'] = 'true'
    
    # Set default run_id from environment or settings
    run_id = int(os.environ.get('DR_RUN_ID', getattr(settings.deepracer, 'run_id', 0)))
    
    # Ensure Docker style is set on the DeepRacer settings (DR_DOCKER_STYLE env var)
    settings.deepracer.docker_style = os.environ.get('DR_DOCKER_STYLE', settings.deepracer.docker_style)
    
    # Use provided values or defaults
    config = ViewerConfig(
        run_id=run_id,
        topic=topic or DEFAULT_TOPIC,
        width=width or DEFAULT_WIDTH,
        height=height or DEFAULT_HEIGHT,
        quality=quality or DEFAULT_QUALITY,
        port=port or DEFAULT_VIEWER_PORT,
        proxy_port=proxy_port or DEFAULT_PROXY_PORT
    )
    
    # Set up pipeline steps
    if update:
        pipeline = stop_viewer_process >> get_robomaker_containers
    else:
        pipeline = get_robomaker_containers
        
    pipeline = pipeline >> wait_for_containers(delay) >> start_stream_proxy >> start_streamlit_viewer
    
    logger.info(f"Starting viewer pipeline for Run ID: {run_id}")
    result = pipeline(config)
    logger.info("Viewer pipeline complete.")
    
    return result

def stop_viewer_pipeline(quiet: bool = True) -> Dict[str, Any]:
    """
    Stop the viewer pipeline and kill associated processes.
    
    Args:
        quiet: If True, suppress console logging (default: True)
        
    Returns:
        Dict with pipeline outcome
    """
    if quiet:
        os.environ['DRFC_CONSOLE_LOGGING'] = 'false'
    else:
        os.environ['DRFC_CONSOLE_LOGGING'] = 'true'
        
    logger.info("Stopping DeepRacer Viewer Pipeline")
    try:
        result = stop_viewer_process(None)
        logger.info("Viewer pipeline stopped.")
        return result
    except Exception as e:
        logger.error("Error stopping viewer pipeline", exc_info=True)
        return {"status": "error", "error": str(e), "type": type(e).__name__}