import os
import subprocess
import time
from typing import List, Tuple, Optional, Dict

from drfc_manager.config_env import settings
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.utils.docker.exceptions.base import DockerError
from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.logging import logger
from drfc_manager.utils.minio.storage_manager import MinioStorageManager
from drfc_manager.utils.paths import get_comms_dir
from drfc_manager.utils.env_utils import get_subprocess_env


storage_manager = MinioStorageManager(settings)


class DockerManager:
    """Handles Docker setup, execution, and cleanup for DeepRacer training using python-on-whales."""

    def __init__(self, config=settings, env_vars: Optional[EnvVars] = None, project_name: str = "deepracer"):
        self.config = config
        self.env_vars = EnvVars()
        if env_vars:
            self.env_vars.update(**{k: v for k, v in env_vars.__dict__.items() if not k.startswith('_')})
            self.env_vars.load_to_environment()
        self.project_name = project_name
        self.model_name = env_vars.DR_LOCAL_S3_MODEL_PREFIX if env_vars else None
        run_id = getattr(self.env_vars, 'DR_RUN_ID', 0)
        self.project_name = f"deepracer-{run_id}"

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
        """Clean up previous DeepRacer runs."""
        logger.info("Cleaning up previous DeepRacer runs...")
        
        try:
            # Stop and remove Docker Compose stack
            compose_cmd = ["docker", "compose", "-p", self.project_name, "down", "--volumes", "--remove-orphans"]
            self._run_command(compose_cmd, check=False)
            logger.info("Cleaned up Docker Compose stack")
        except Exception as e:
            logger.debug(f"Docker Compose cleanup failed (might not exist): {e}")
        
        try:
            # Stop and remove Docker Swarm stack
            swarm_cmd = ["docker", "stack", "rm", self.project_name]
            self._run_command(swarm_cmd, check=False)
            logger.info("Cleaned up Docker Swarm stack")
        except Exception as e:
            logger.debug(f"Docker Swarm cleanup failed (might not exist): {e}")
        
        if prune_system:
            try:
                # Prune unused Docker resources
                prune_cmd = ["docker", "system", "prune", "-f"]
                self._run_command(prune_cmd, check=False)
                logger.info("Pruned unused Docker resources")
            except Exception as e:
                logger.warning(f"Failed to prune Docker resources: {e}")
        
        logger.info("Cleanup completed")

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
        # Use the original compose file directly
        temp_compose_path = training_compose_path

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
        
        # Update with required values - only those defined in system.env and run.env
        required_vars = {
            'DR_CURRENT_PARAMS_FILE': params_file,
            'DR_CAMERA_KVS_ENABLE': self.env_vars.DR_CAMERA_KVS_ENABLE,
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
            'DR_MINIO_URL': self.env_vars.DR_MINIO_URL,
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
        
        critical_vars = ['DR_SIMAPP_SOURCE', 'DR_SIMAPP_VERSION', 'DR_MINIO_URL']
        missing_vars = [var for var in critical_vars if not os.environ.get(var)]
        if missing_vars:
            logger.error(f"Missing critical environment variables in os.environ: {', '.join(missing_vars)}")
            logger.error(f"Current os.environ state: {dict(os.environ)}")
            raise DockerError(f"Missing critical environment variables: {', '.join(missing_vars)}")
        logger.info("Verified all critical environment variables are set in os.environ")

    def _create_network_if_not_exists(self, network_name: str = "sagemaker-local"):
        """Create the sagemaker-local network if it doesn't exist."""
        logger.info(f"Checking if network {network_name} exists...")
        
        # Check if network exists
        check_cmd = ["docker", "network", "ls", "--filter", f"name={network_name}", "--format", "{{.Name}}"]
        result = self._run_command(check_cmd, check=False)
        
        if network_name not in result.stdout:
            logger.info(f"Creating network {network_name}...")
            create_cmd = ["docker", "network", "create", network_name]
            self._run_command(create_cmd)
            logger.info(f"Network {network_name} created successfully")
        else:
            logger.info(f"Network {network_name} already exists")

    def _connect_containers_to_network(self):
        """Explicitly connect containers to the sagemaker-local network if needed."""
        logger.info("Ensuring containers are connected to sagemaker-local network...")
        
        for service in ["rl_coach", "robomaker"]:
            container_name = f"{self.project_name}-{service}-1"
            
            # Check if container is already on the network
            cmd = ["docker", "inspect", container_name, "--format", "{{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}"]
            result = self._run_command(cmd, check=False)
            
            if result.returncode == 0 and "sagemaker-local" in result.stdout:
                logger.info(f"Container {container_name} is already connected to sagemaker-local network")
            else:
                logger.info(f"Connecting container {container_name} to sagemaker-local network...")
                connect_cmd = ["docker", "network", "connect", "sagemaker-local", container_name]
                self._run_command(connect_cmd, check=False)
                logger.info(f"Connected {container_name} to sagemaker-local network")

    def _verify_network_connectivity(self):
        """Verify that containers are connected to the sagemaker-local network."""
        logger.info("Verifying network connectivity...")
        
        # Check if containers are on the sagemaker-local network
        cmd = ["docker", "network", "inspect", "sagemaker-local", "--format", "{{.Containers}}"]
        result = self._run_command(cmd, check=False)
        
        if result.returncode == 0:
            logger.info("sagemaker-local network exists and contains containers")
            logger.debug(f"Network containers: {result.stdout}")
        else:
            logger.warning("sagemaker-local network not found or no containers connected")
        
        # Check specific containers
        for service in ["rl_coach", "robomaker"]:
            container_name = f"{self.project_name}-{service}-1"
            cmd = ["docker", "inspect", container_name, "--format", "{{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}"]
            result = self._run_command(cmd, check=False)
            
            if result.returncode == 0 and "sagemaker-local" in result.stdout:
                logger.info(f"Container {container_name} is connected to sagemaker-local network")
            else:
                logger.warning(f"Container {container_name} is NOT connected to sagemaker-local network")
                logger.debug(f"Container networks: {result.stdout}")

    def start_deepracer_stack(self):
        """Start the DeepRacer Docker stack."""
        logger.info("Starting DeepRacer Docker stack...")
        
        # Initialize temp_compose_path to avoid UnboundLocalError
        temp_compose_path = None
        
        # Create the required network
        self._create_network_if_not_exists()
        
        try:
            # Prepare training configuration before starting containers
            self._prepare_training_config()
            
            # Prepare Docker Compose file
            compose_files, multi_added = self._prepare_compose_files(self.env_vars.DR_WORKERS)
            temp_compose_path = compose_files[0]
            logger.info(f"Using Docker Compose files: {compose_files}")
            
            # Set environment variables
            self._set_runtime_env_vars(self.env_vars.DR_WORKERS)
            logger.info("Environment variables set successfully")
            
            # Start the stack
            logger.info("Starting Docker Compose stack...")
            
            if getattr(self.env_vars, 'DR_DOCKER_STYLE', 'compose').lower() == "swarm":
                # Use Docker Swarm mode
                logger.info("Using Docker Swarm mode")
                cmd = ["docker", "stack", "deploy"]
                for file in compose_files:
                    cmd.extend(["-c", file])
                cmd.extend(["--detach=true", self.project_name])
                
                if self.env_vars.DR_WORKERS > 1 and multi_added:
                    logger.warning("Scaling not supported in Swarm mode - using compose file configuration")
            else:
                # Use Docker Compose mode
                logger.info("Using Docker Compose mode")
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
                    ]
                )
                
                if self.env_vars.DR_WORKERS > 1 and multi_added:
                    cmd.extend(["--scale", f"robomaker={self.env_vars.DR_WORKERS}"])
            
            # Execute the command
            env = get_subprocess_env(self.env_vars)
            result = self._run_command(cmd, env=env)  # noqa: F841
            
            # Wait for containers to be ready
            if not self._wait_for_containers_ready():
                logger.warning("Containers may not be fully ready, but continuing...")
            
            # Additional delay to ensure IP addresses are stable
            logger.info("Waiting for IP addresses to stabilize...")
            time.sleep(10)
            
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
        elif file_path is None:
            logger.debug("No temporary file to clean up")

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

    def _prepare_training_config(self):
        """Prepare training configuration and upload to S3 before starting containers."""
        logger.info("Preparing training configuration...")
        
        try:
            # Import here to avoid circular imports
            from drfc_manager.helpers.training_params import writing_on_temp_training_yml
            
            # Generate training configuration using existing helper
            yaml_key, local_yaml_path = writing_on_temp_training_yml(self.env_vars.DR_LOCAL_S3_MODEL_PREFIX)
            
            # Upload the generated YAML file to S3
            storage_manager.upload_local_file(local_yaml_path, yaml_key)
            
            # Clean up local file
            if os.path.exists(local_yaml_path):
                os.remove(local_yaml_path)
                logger.info(f"Cleaned up local file: {local_yaml_path}")
            
            logger.info(f"Training configuration uploaded to S3: {yaml_key}")
            
        except Exception as e:
            logger.error(f"Failed to prepare training configuration: {e}")
            raise

    def _wait_for_containers_ready(self, timeout: int = 60):
        """Wait for containers to be ready and have stable IP addresses."""
        logger.info("Waiting for containers to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if containers are running
                containers = self._run_command(
                    ["docker", "ps", "--filter", f"name={self.project_name}", "--format", "{{.Names}}"],
                    check=False
                )
                
                if containers.returncode == 0 and containers.stdout.strip():
                    container_names = containers.stdout.strip().split('\n')
                    logger.info(f"Found containers: {container_names}")
                    
                    # Wait a bit more for IP addresses to stabilize
                    time.sleep(5)
                    return True
                    
            except Exception as e:
                logger.debug(f"Error checking containers: {e}")
            
            time.sleep(2)
        
        logger.warning(f"Timeout waiting for containers to be ready after {timeout} seconds")
        return False
