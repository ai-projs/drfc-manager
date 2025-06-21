import os
import subprocess
import time
from typing import List, Tuple, Optional, Dict

from drfc_manager.config_env import settings
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.utils.docker.exceptions.base import DockerError
from drfc_manager.utils.redis.manager import RedisManager
from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.logging import logger
from drfc_manager.utils.paths import get_comms_dir
from drfc_manager.utils.env_utils import get_subprocess_env


class DockerManager:
    """Handles Docker setup, execution, and cleanup for DeepRacer training using python-on-whales."""

    def __init__(self, config=settings, env_vars: Optional[EnvVars] = None):
        self.config = config
        self.env_vars = EnvVars()
        if env_vars:
            self.env_vars.update(**{k: v for k, v in env_vars.__dict__.items() if not k.startswith('_')})
            self.env_vars.load_to_environment()
        run_id = getattr(self.env_vars, 'DR_RUN_ID', 0)
        self.project_name = f"deepracer-{run_id}"
        self.redis_manager = RedisManager(config)

    def _run_command(
        self, command: List[str], check: bool = True, capture: bool = True, 
        env: Optional[Dict[str, str]] = None
    ) -> subprocess.CompletedProcess:
        logger.debug(f"Executing: {' '.join(command)}")
        try:
            env = get_subprocess_env(self.env_vars)
            result = subprocess.run(
                command,
                check=check,
                capture_output=capture,
                text=True,
                env=env,
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
                stderr=e.stderr,
            ) from e
        except Exception as e:
            raise DockerError(
                message=f"Failed to execute command: {e}", command=command
            ) from e

    def cleanup_previous_run(self, prune_system: bool = True):
        """Stop existing containers and optionally prune Docker resources."""
        logger.info(f"Cleaning up previous run for project {self.project_name}...")

        self._run_command(
            [
                "docker",
                "compose",
                "-p",
                self.project_name,
                "down",
                "--remove-orphans",
                "--volumes",
            ],
            check=False,
        )
        time.sleep(2)

        if prune_system:
            logger.info("Pruning unused Docker resources...")
            self._run_command(["docker", "network", "prune", "-f"], check=False)
            self._run_command(["docker", "system", "prune", "-f"], check=False)
            time.sleep(2)

    def _get_compose_file_paths(self, file_types: List[ComposeFileType]) -> List[str]:
        """Get full paths for compose files."""
        from drfc_manager.utils.docker.utilities import adjust_composes_file_names

        return adjust_composes_file_names(
            [file_type.value for file_type in file_types]
        )

    def _prepare_compose_files(self, workers: int) -> Tuple[List[str], bool]:
        """Prepare all necessary compose files and determine if multi-worker is configured."""
        training_compose_path = self._get_compose_file_paths(
            [ComposeFileType.TRAINING]
        )[0]
        temp_compose_path = self.redis_manager.create_modified_compose_file(
            training_compose_path
        )

        compose_file_types = [ComposeFileType.KEYS, ComposeFileType.ENDPOINT]

        if getattr(self.env_vars, 'DR_ROBOMAKER_MOUNT_LOGS', False):
            compose_file_types.append(ComposeFileType.MOUNT)

        multi_added = False
        if workers > 1 and getattr(self.env_vars, 'DR_DOCKER_STYLE', 'compose') != "swarm":
            if self._setup_multiworker_env():
                compose_file_types.append(ComposeFileType.ROBOMAKER_MULTI)
                multi_added = True

        additional_compose_files = self._get_compose_file_paths(compose_file_types)
        final_compose_files = [temp_compose_path] + additional_compose_files

        return final_compose_files, multi_added

    def _setup_multiworker_env(self) -> bool:
        """Set up environment for multiple workers."""
        try:
            run_id = getattr(self.env_vars, 'DR_RUN_ID', 0) if self.env_vars else 0
            comms_dir = get_comms_dir(run_id)
            self.env_vars.update(DR_DIR=str(comms_dir.parent.parent))
            self.env_vars.load_to_environment()
            logger.info(f"Created comms dir: {comms_dir}")
            return True
        except OSError as e:
            logger.warning(
                f"Failed to create comms directory: {e}. Multi-worker may fail."
            )
            return False

    def _set_runtime_env_vars(self, workers: int):
        """Set environment variables for Docker Compose."""
        logger.info("Setting up runtime environment variables...")
        
        logger.info(f"Initial EnvVars state: {self.env_vars}")
        
        params_file = getattr(self.env_vars, 'DR_LOCAL_S3_TRAINING_PARAMS_FILE', 'training_params.yaml')
        
        # Update with required values
        required_vars = {
            'DR_CURRENT_PARAMS_FILE': params_file,
            'DR_CAMERA_KVS_ENABLE': False,
            'DR_SIMAPP_SOURCE': self.env_vars.DR_SIMAPP_SOURCE,
            'DR_SIMAPP_VERSION': self.env_vars.DR_SIMAPP_VERSION,
            'DR_WORLD_NAME': self.env_vars.DR_WORLD_NAME,
            'DR_KINESIS_STREAM_NAME': self.env_vars.DR_KINESIS_STREAM_NAME,
            'DR_GUI_ENABLE': self.env_vars.DR_GUI_ENABLE,
            'DR_ROBOMAKER_TRAIN_PORT': self.env_vars.DR_ROBOMAKER_TRAIN_PORT,
            'DR_ROBOMAKER_GUI_PORT': self.env_vars.DR_ROBOMAKER_GUI_PORT,
            'DR_LOCAL_ACCESS_KEY_ID': self.env_vars.DR_LOCAL_ACCESS_KEY_ID,
            'DR_LOCAL_SECRET_ACCESS_KEY': self.env_vars.DR_LOCAL_SECRET_ACCESS_KEY,
            'DR_LOCAL_S3_PRETRAINED': self.env_vars.DR_LOCAL_S3_PRETRAINED,
            'DR_LOCAL_S3_PRETRAINED_PREFIX': self.env_vars.DR_LOCAL_S3_PRETRAINED_PREFIX,
            'DR_LOCAL_S3_PRETRAINED_CHECKPOINT': self.env_vars.DR_LOCAL_S3_PRETRAINED_CHECKPOINT,
            'DR_LOCAL_S3_HYPERPARAMETERS_KEY': self.env_vars.DR_LOCAL_S3_HYPERPARAMETERS_KEY,
            'DR_LOCAL_S3_MODEL_METADATA_KEY': self.env_vars.DR_LOCAL_S3_MODEL_METADATA_KEY,
            'REDIS_HOST': self.env_vars.REDIS_HOST,
            'REDIS_PORT': self.env_vars.REDIS_PORT,
            'DR_MINIO_URL': self.env_vars.DR_MINIO_URL
        }
        
        self.env_vars.update(**required_vars)
        self.env_vars.load_to_environment()
        logger.info("Updated environment variables with required values")
        
        if workers > 1:
            self.env_vars.update(ROBOMAKER_COMMAND="/opt/simapp/run.sh run distributed_training.launch")
            self.env_vars.load_to_environment()
            logger.info("Set RoboMaker command for distributed training")
        else:
            self.env_vars.update(ROBOMAKER_COMMAND="/opt/simapp/run.sh run distributed_training.launch")
            self.env_vars.load_to_environment()
            logger.info("Set RoboMaker command for single worker")
        
        self.env_vars.load_to_environment()
        logger.info("Loaded all environment variables")
        
        critical_vars = ['DR_SIMAPP_SOURCE', 'DR_SIMAPP_VERSION', 'REDIS_HOST', 'REDIS_PORT', 'DR_MINIO_URL']
        missing_vars = [var for var in critical_vars if not os.environ.get(var)]
        if missing_vars:
            logger.error(f"Missing critical environment variables in os.environ: {', '.join(missing_vars)}")
            logger.error(f"Current os.environ state: {dict(os.environ)}")
            raise DockerError(f"Missing critical environment variables: {', '.join(missing_vars)}")
        logger.info("Verified all critical environment variables are set in os.environ")

    def start_deepracer_stack(self):
        """Start the DeepRacer Docker stack."""
        try:
            logger.info("Starting DeepRacer Docker stack...")
            
            # Prepare Docker Compose file
            compose_files, multi_added = self._prepare_compose_files(self.env_vars.DR_WORKERS)
            temp_compose_path = compose_files[0]
            logger.info(f"Using Docker Compose files: {compose_files}")
            
            # Set environment variables
            self._set_runtime_env_vars(self.env_vars.DR_WORKERS)
            logger.info("Environment variables set successfully")
            
            # Start the stack
            logger.info("Starting Docker Compose stack...")
            cmd = ["docker", "compose"]
            for file in compose_files:
                cmd.extend(["-f", file])
            cmd.extend(
                [
                    "-p",
                    self.project_name,
                    "up",
                    "-d",
                    "--remove-orphans",
                    "--force-recreate",
                ]
            )

            if self.env_vars.DR_WORKERS > 1 and multi_added:
                cmd.extend(["--scale", f"robomaker={self.env_vars.DR_WORKERS}"])
            elif self.env_vars.DR_WORKERS > 1 and not multi_added:
                logger.warning(
                    "Not scaling RoboMaker because robomaker-multi config was not included."
                )

            # Log the full command and environment before executing
            logger.debug(f"Executing Docker Compose command: {' '.join(cmd)}")
            logger.debug("Current environment variables:")
            for var in sorted(os.environ.keys()):
                if var.startswith('DR_'):
                    logger.debug(f"{var}={os.environ[var]}")

            env = get_subprocess_env(self.env_vars)
            self._run_command(cmd, env=env)

            # Wait for services to be ready
            logger.info("Waiting for services to be ready...")
            time.sleep(5)  # Give services time to start
            
            # Check Redis container
            logger.info("Checking Redis container status...")
            redis_status = self._run_command(["docker", "ps", "--filter", "name=redis", "--format", "{{.Status}}"], check=False)
            logger.info(f"Redis container status: {redis_status.stdout.strip()}")
            
            # Check Redis logs
            logger.info("Checking Redis logs...")
            redis_logs = self._run_command(["docker", "logs", f"{self.project_name}-redis-1"], check=False)
            logger.info(f"Redis logs:\n{redis_logs.stdout}")
            
            # Check RoboMaker container
            logger.info("Checking RoboMaker container status...")
            robomaker_status = self._run_command(["docker", "ps", "--filter", "name=robomaker", "--format", "{{.Status}}"], check=False)
            logger.info(f"RoboMaker container status: {robomaker_status.stdout.strip()}")
            
            # Check RoboMaker logs
            logger.info("Checking RoboMaker logs...")
            robomaker_logs = self._run_command(["docker", "logs", f"{self.project_name}-robomaker-1"], check=False)
            logger.info(f"RoboMaker logs:\n{robomaker_logs.stdout}")
            
            logger.info("DeepRacer Docker stack started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start DeepRacer Docker stack: {str(e)}")
            raise DockerError(f"Failed to start DeepRacer Docker stack: {str(e)}") from e
        finally:
            self._cleanup_temp_file(temp_compose_path)

    def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary file if it exists."""
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file {file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")

    def check_container_status(self, expected_workers: int):
        """Check if the expected containers are running."""
        logger.info("Checking container status...")
        time.sleep(5)

        self._run_command(
            ["docker", "compose", "-p", self.project_name, "ps"], check=False
        )

        robomaker_running_cmd = [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={self.project_name}",
            "--filter",
            "label=com.docker.compose.service=robomaker",
            "--filter",
            "status=running",
            "-q",
        ]
        result = self._run_command(robomaker_running_cmd, check=False)
        running_ids = result.stdout.strip().splitlines() if result.stdout else []

        if running_ids:
            logger.info(f"Found running RoboMaker containers: {len(running_ids)}")
            if len(running_ids) == expected_workers:
                logger.info(
                    f"Successfully started {expected_workers} RoboMaker workers."
                )
            else:
                logger.warning(
                    f"Expected {expected_workers} workers, but found {len(running_ids)} running."
                )
        else:
            logger.warning("No RoboMaker containers are running.")

    def check_logs(self, service_name: str, tail: int = 30):
        """Get logs for a specific service."""
        logger.info(f"\n--- Logs for {service_name} (tail {tail}) ---")
        cmd = [
            "docker",
            "compose",
            "-p",
            self.project_name,
            "logs",
            service_name,
            "--tail",
            str(tail),
        ]
        self._run_command(cmd, check=False)

    def compose_up(
        self,
        project_name: str,
        compose_files: str,
        scale_options: Optional[Dict[str, int]] = None,
    ):
        """Runs docker compose up command."""
        cmd = ["docker", "compose"]
        # Split the compose_files string by the separator used to join them
        separator = getattr(settings.docker, "dr_docker_file_sep", " -f ")
        files_list = compose_files.split(separator)
        for file in files_list:
            if file.strip():  # Avoid empty strings if splitting results in them
                # Assume the first part doesn't have the separator prefix
                if not cmd[-1] == "-f":
                    cmd.extend(["-f", file.strip()])
                else:
                    cmd.append(file.strip())

        cmd.extend(
            ["-p", project_name, "up", "-d", "--remove-orphans"]
        )  # Consider --force-recreate if needed

        if scale_options:
            for service, replicas in scale_options.items():
                cmd.extend(["--scale", f"{service}={replicas}"])

        result = self._run_command(cmd)
        return result.stdout  # Or return the whole result object

    def compose_down(
        self, project_name: str, compose_files: str, remove_volumes: bool = True
    ):
        """Runs docker compose down command."""
        cmd = ["docker", "compose"]
        # Split files like in compose_up
        separator = getattr(settings.docker, "dr_docker_file_sep", " -f ")
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

        result = self._run_command(
            cmd, check=False
        )  # Allow failure if stack doesn't exist
        return result.stdout

    def deploy_stack(self, stack_name: str, compose_files: str):
        """Deploys a stack in Docker Swarm."""
        cmd = ["docker", "stack", "deploy"]
        # Split files like in compose_up, but use -c for swarm
        separator = getattr(
            settings.docker, "dr_docker_file_sep", " -f "
        )  # Swarm might use different separator? Use same for now.
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
        result = self._run_command(
            cmd, check=False
        )  # Allow failure if stack doesn't exist
        return result.stdout

    def list_services(self, stack_name: str) -> List[str]:
        """Lists services for a given swarm stack."""
        cmd = [
            "docker",
            "stack",
            "ps",
            stack_name,
            "--format",
            "{{.Name}}",
            "--filter",
            "desired-state=running",
        ]
        result = self._run_command(cmd, check=False)
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip().splitlines()
        return []
