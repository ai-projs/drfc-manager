from enum import Enum
import os
import subprocess
import time
from typing import List, Tuple, Optional, Dict

from src.config import settings
from src.utils.docker.exceptions.base import DockerError
from src.utils.redis.manager import RedisManager
from src.types.docker import ComposeFileType
from src.utils.logging import logger


class DockerManager:
    """Handles Docker setup, execution, and cleanup for DeepRacer training using python-on-whales."""

    def __init__(self, config=settings):
        self.config = config
        self.project_name = f"deepracer-{self.config.deepracer.run_id}"
        self.redis_manager = RedisManager(config)
        
    def _run_command(self, command: List[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
        logger.debug(f"Executing: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=capture,
                text=True,
                env=os.environ.copy()
            )
            if capture and result.stdout:
                logger.debug(f"Stdout:\n{result.stdout}")
            if capture and result.stderr:
                logger.debug(f"Stderr:\n{result.stderr}")
            return result
        except subprocess.CalledProcessError as e:
            raise DockerError(
                message=f"Docker command failed with exit code {e.returncode}",
                command=command,
                stderr=e.stderr
            ) from e
        except Exception as e:
            raise DockerError(
                message=f"Failed to execute command: {e}",
                command=command
            ) from e
        
    def cleanup_previous_run(self, prune_system: bool = True):
        """Stop existing containers and optionally prune Docker resources."""
        logger.info(f"Cleaning up previous run for project {self.project_name}...")
        
        self._run_command(
            ["docker", "compose", "-p", self.project_name, "down", "--remove-orphans", "--volumes"],
            check=False
        )
        time.sleep(2)

        if prune_system:
            logger.info("Pruning unused Docker resources...")
            self._run_command(["docker", "network", "prune", "-f"], check=False)
            self._run_command(["docker", "system", "prune", "-f"], check=False)
            time.sleep(2)

    def _get_compose_file_paths(self, file_types: List[ComposeFileType]) -> List[str]:
        """Get full paths for compose files."""
        from src.utils.docker.utilities import _adjust_composes_file_names
        return _adjust_composes_file_names([file_type.value for file_type in file_types])

    def _prepare_compose_files(self, workers: int) -> Tuple[List[str], bool]:
        """Prepare all necessary compose files and determine if multi-worker is configured."""
        training_compose_path = self._get_compose_file_paths([ComposeFileType.TRAINING])[0]
        temp_compose_path = self.redis_manager.create_modified_compose_file(training_compose_path)
        
        compose_file_types = [ComposeFileType.KEYS, ComposeFileType.ENDPOINT]
        
        if self.config.deepracer.robomaker_mount_logs:
            compose_file_types.append(ComposeFileType.MOUNT)

        multi_added = False
        if workers > 1 and self.config.docker.docker_style != "swarm":
            if self._setup_multiworker_env():
                compose_file_types.append(ComposeFileType.ROBOMAKER_MULTI)
                multi_added = True
        
        additional_compose_files = self._get_compose_file_paths(compose_file_types)
        final_compose_files = [temp_compose_path] + additional_compose_files
        
        return final_compose_files, multi_added

    def _setup_multiworker_env(self) -> bool:
        """Set up environment for multiple workers."""
        if not self.config.docker.drfc_base_path or not os.path.isdir(self.config.docker.drfc_base_path):
            logger.warning(f"DRFC_REPO_ABS_PATH is not set or invalid. Skipping robomaker-multi.")
            return False
            
        os.environ["DR_DIR"] = str(self.config.docker.drfc_base_path)
        
        comms_dir = os.path.join(
            self.config.docker.drfc_base_path, 
            "tmp", 
            f"comms.{self.config.deepracer.run_id}"
        )
        
        try:
            os.makedirs(os.path.dirname(comms_dir), exist_ok=True)
            os.makedirs(comms_dir, exist_ok=True)
            logger.info(f"Created comms dir: {comms_dir}")
            return True
        except OSError as e:
            logger.warning(f"Failed to create comms directory '{comms_dir}': {e}. Multi-worker may fail.")
            return False

    def _set_runtime_env_vars(self, workers: int):
        """Set environment variables for Docker Compose."""
        os.environ["DR_ROBOMAKER_GUI_PORT"] = str(
            self.config.deepracer.robomaker_gui_port_base + self.config.deepracer.run_id
        )
        os.environ["DR_ROBOMAKER_TRAIN_PORT"] = str(
            self.config.deepracer.robomaker_train_port_base + self.config.deepracer.run_id
        )
        os.environ["DR_CURRENT_PARAMS_FILE"] = self.config.deepracer.local_s3_training_params_file
        os.environ["DR_RUN_ID"] = str(self.config.deepracer.run_id)
        os.environ['REDIS_HOST'] = self.config.redis.host
        os.environ['REDIS_PORT'] = str(self.config.redis.port)

        if workers > 1:
            os.environ["ROBOMAKER_COMMAND"] = "/opt/simapp/run.sh multi distributed_training.launch"
        else:
            os.environ["ROBOMAKER_COMMAND"] = "/opt/simapp/run.sh run distributed_training.launch"
        logger.info(f"ROBOMAKER_COMMAND set to: {os.environ.get('ROBOMAKER_COMMAND')}")

    def start_deepracer_stack(self):
        """Start the DeepRacer Docker stack with all required services."""
        workers = self.config.deepracer.workers
        logger.info(f"Starting DeepRacer stack for project {self.project_name} with {workers} workers...")

        compose_files, multi_added = self._prepare_compose_files(workers)
        temp_compose_path = compose_files[0]
        logger.info(f"Using compose files: {compose_files}")

        # 3. Set runtime environment variables
        self._set_runtime_env_vars(workers)

        try:
            cmd = ["docker", "compose"]
            for file in compose_files:
                cmd.extend(["-f", file])
            cmd.extend(["-p", self.project_name, "up", "-d", "--remove-orphans", "--force-recreate"])

            if workers > 1 and multi_added:
                cmd.extend(["--scale", f"robomaker={workers}"])
            elif workers > 1 and not multi_added:
                logger.warning("Not scaling RoboMaker because robomaker-multi config was not included.")
            
            self._run_command(cmd)
            
        except Exception as e:
            raise DockerError(f"Failed to start DeepRacer stack: {e}")
        finally:
            self._cleanup_temp_file(temp_compose_path)

        self.check_container_status(workers)
        
    def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary file if it exists."""
        if file_path and os.path.exists(file_path):
            try:
                # os.remove(file_path)
                logger.info(f"Cleaned up temporary file {file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")

    def check_container_status(self, expected_workers: int):
        """Check if the expected containers are running."""
        logger.info("Checking container status...")
        time.sleep(5)
        
        self._run_command(["docker", "compose", "-p", self.project_name, "ps"], check=False)

        robomaker_running_cmd = [
            "docker", "ps",
            "--filter", f"label=com.docker.compose.project={self.project_name}",
            "--filter", "label=com.docker.compose.service=robomaker",
            "--filter", "status=running", "-q"
        ]
        result = self._run_command(robomaker_running_cmd, check=False)
        running_ids = result.stdout.strip().splitlines() if result.stdout else []

        if running_ids:
            logger.info(f"Found running RoboMaker containers: {len(running_ids)}")
            if len(running_ids) == expected_workers:
                logger.info(f"Successfully started {expected_workers} RoboMaker workers.")
            else:
                logger.warning(f"Expected {expected_workers} workers, but found {len(running_ids)} running.")
        else:
            logger.warning("No RoboMaker containers are running.")

    def check_logs(self, service_name: str, tail: int = 30):
        """Get logs for a specific service."""
        logger.info(f"\n--- Logs for {service_name} (tail {tail}) ---")
        cmd = ["docker", "compose", "-p", self.project_name, "logs", service_name, "--tail", str(tail)]
        self._run_command(cmd, check=False)

    def compose_up(self, project_name: str, compose_files: str, scale_options: Optional[Dict[str, int]] = None):
        """Runs docker compose up command."""
        cmd = ["docker", "compose"]
        # Split the compose_files string by the separator used to join them
        separator = getattr(settings.docker, 'dr_docker_file_sep', ' -f ')
        files_list = compose_files.split(separator)
        for file in files_list:
             if file.strip(): # Avoid empty strings if splitting results in them
                  # Assume the first part doesn't have the separator prefix
                  if not cmd[-1] == "-f":
                       cmd.extend(["-f", file.strip()])
                  else:
                       cmd.append(file.strip())


        cmd.extend(["-p", project_name, "up", "-d", "--remove-orphans"]) # Consider --force-recreate if needed

        if scale_options:
            for service, replicas in scale_options.items():
                cmd.extend(["--scale", f"{service}={replicas}"])

        result = self._run_command(cmd)
        return result.stdout # Or return the whole result object

    def compose_down(self, project_name: str, compose_files: str, remove_volumes: bool = True):
        """Runs docker compose down command."""
        cmd = ["docker", "compose"]
        # Split files like in compose_up
        separator = getattr(settings.docker, 'dr_docker_file_sep', ' -f ')
        files_list = compose_files.split(separator)
        for file in files_list:
            if file.strip():
                 if not cmd[-1] == "-f":
                      cmd.extend(["-f", file.strip()])
                 else:
                      cmd.append(file.strip())

        cmd.extend(["-p", project_name, "down", "--remove-orphans"])
        if remove_volumes:
            cmd.append("--volumes")

        result = self._run_command(cmd, check=False) # Allow failure if stack doesn't exist
        return result.stdout

    def deploy_stack(self, stack_name: str, compose_files: str):
        """Deploys a stack in Docker Swarm."""
        cmd = ["docker", "stack", "deploy"]
        # Split files like in compose_up, but use -c for swarm
        separator = getattr(settings.docker, 'dr_docker_file_sep', ' -f ') # Swarm might use different separator? Use same for now.
        files_list = compose_files.split(separator)
        for file in files_list:
            if file.strip():
                 # Swarm uses -c for compose files
                 if not cmd[-1] == "-c":
                      cmd.extend(["-c", file.strip()])
                 else:
                      cmd.append(file.strip())

        # Add detach flag based on docker version if needed (logic from start.sh)
        # docker_major_version = ... # Need a way to get docker version
        # if docker_major_version > 24:
        #     cmd.append("--detach=true")

        cmd.append(stack_name)
        result = self._run_command(cmd)
        return result.stdout

    def remove_stack(self, stack_name: str):
        """Removes a stack from Docker Swarm."""
        cmd = ["docker", "stack", "rm", stack_name]
        result = self._run_command(cmd, check=False) # Allow failure if stack doesn't exist
        return result.stdout

    def list_services(self, stack_name: str) -> List[str]:
         """Lists services for a given swarm stack."""
         cmd = ["docker", "stack", "ps", stack_name, "--format", "{{.Name}}", "--filter", "desired-state=running"]
         result = self._run_command(cmd, check=False)
         if result.returncode == 0 and result.stdout:
              return result.stdout.strip().splitlines()
         return [] 