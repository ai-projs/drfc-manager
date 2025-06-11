import os
import time
import shutil
import subprocess
from typing import List, Optional, Tuple, cast
from contextlib import contextmanager

from drfc_manager.config_env import AppConfig
from drfc_manager.utils.docker.command_executor import CommandExecutor
from drfc_manager.utils.docker.compose_manager import ComposeManager
from drfc_manager.utils.docker.docker_constants import (
    DockerStyle,
    ERROR_MESSAGES,
)
from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.docker.exceptions import DockerError
from drfc_manager.utils.docker.swarm_manager import SwarmManager
from drfc_manager.utils.logging_config import get_logger
from drfc_manager.utils.paths import get_comms_dir
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.utils.redis.manager import RedisManager


class DockerManager:
    """Handles Docker setup, execution, and cleanup for DeepRacer training."""

    def __init__(
        self,
        config: AppConfig,
        project_name: Optional[str] = None,
        command_executor: Optional[CommandExecutor] = None,
    ):
        """Initialize the Docker manager.

        Args:
            config: Application configuration
            project_name: Optional project name
            command_executor: Optional command executor instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.project_name = project_name or f"deepracer-{self.config.deepracer.run_id}"
        self.docker_style = DockerStyle(self.config.docker.docker_style)

        self.command_executor = command_executor or CommandExecutor()

        # Pass docker_style to RedisManager
        self.redis_manager = RedisManager(
            config=self.config, docker_style=self.docker_style
        )

        if self.docker_style == DockerStyle.SWARM:
            self.manager = SwarmManager(self.project_name, self.command_executor)
        else:
            self.manager = ComposeManager(self.project_name, self.command_executor)

    def _run_command(
        self, command: List[str], check: bool = True, capture: bool = True
    ) -> subprocess.CompletedProcess:
        """Execute a command and return the result.

        Args:
            command: List of command arguments to execute
            check: Whether to check the return code
            capture: Whether to capture stdout and stderr

        Returns:
            CompletedProcess object containing the command result

        Raises:
            DockerError: If the command fails and check is True
        """
        self.logger.debug(f"Executing: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=capture,
                text=True,
                env=os.environ.copy(),
            )
            if capture and result.stdout:
                self.logger.debug(f"Stdout:\n{result.stdout}")
            if capture and result.stderr:
                self.logger.debug(f"Stderr:\n{result.stderr}")
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

    @contextmanager
    def _managed_compose_file(self, base_compose_path: str):
        """Context manager for handling temporary compose file lifecycle.

        Args:
            base_compose_path: Path to the base compose file

        Yields:
            Path to the temporary compose file
        """
        temp_compose_path = None
        try:
            temp_compose_path = self.redis_manager.create_modified_compose_file(
                base_compose_path
            )
            yield temp_compose_path
        finally:
            if temp_compose_path and os.path.exists(temp_compose_path):
                try:
                    os.remove(temp_compose_path)
                    self.logger.debug("cleaned_up_temp_file", path=temp_compose_path)
                except Exception as e:
                    self.logger.warning(
                        "temp_file_cleanup_failed", path=temp_compose_path, error=str(e)
                    )

    @contextmanager
    def _managed_comms_dir(self, run_id: int):
        """Context manager for handling comms directory lifecycle.

        Args:
            run_id: Run ID for the comms directory

        Yields:
            Path to the comms directory
        """
        comms_dir = get_comms_dir(run_id)
        try:
            if comms_dir.exists():
                shutil.rmtree(comms_dir)
                self.logger.info("cleaned_up_existing_comms_dir", path=str(comms_dir))

            comms_dir.mkdir(parents=True, exist_ok=True)
            yield comms_dir
        except Exception as e:
            self.logger.error(
                "comms_dir_setup_failed", path=str(comms_dir), error=str(e)
            )
            raise

    @contextmanager
    def _managed_docker_stack(self):
        """Context manager for handling Docker stack lifecycle.

        Yields:
            None
        """
        try:
            yield
        finally:
            try:
                self.cleanup_previous_run(prune=False)
            except Exception as e:
                self.logger.warning("stack_cleanup_failed", error=str(e))

    def _set_runtime_env_vars(self, workers: int):
        """Set environment variables for Docker Compose."""
        # Create a new EnvVars instance
        env_vars = EnvVars()

        # Set image-related environment variables
        env_vars.DR_SIMAPP_SOURCE = self.config.docker.simapp_image.split(":")[0]
        env_vars.DR_SIMAPP_VERSION = self.config.docker.simapp_image.split(":")[1]
        env_vars.DR_MINIO_IMAGE = self.config.docker.minio_image

        # Set worker and style configuration
        env_vars.DR_WORKERS = workers
        env_vars.DR_DOCKER_STYLE = self.config.docker.docker_style

        # Set Redis configuration - Redis is started internally by SimApp
        env_vars.REDIS_HOST = "rl_coach"  # Use the service name from docker-compose
        env_vars.REDIS_PORT = 6379  # Default Redis port

        # Set port-related environment variables
        env_vars.DR_ROBOMAKER_GUI_PORT = str(
            self.config.deepracer.robomaker_gui_port_base + self.config.deepracer.run_id
        )
        env_vars.DR_ROBOMAKER_TRAIN_PORT = str(
            self.config.deepracer.robomaker_train_port_base
            + self.config.deepracer.run_id
        )

        # Set S3-related environment variables
        env_vars.DR_LOCAL_S3_BUCKET = self.config.deepracer.local_s3_bucket
        env_vars.DR_LOCAL_S3_MODEL_PREFIX = self.config.deepracer.local_s3_model_prefix
        env_vars.DR_LOCAL_S3_PRETRAINED = self.config.deepracer.pretrained_model
        env_vars.DR_LOCAL_S3_PRETRAINED_PREFIX = (
            self.config.deepracer.pretrained_s3_prefix
        )
        env_vars.DR_LOCAL_S3_PRETRAINED_CHECKPOINT = ""
        env_vars.DR_LOCAL_S3_HYPERPARAMETERS_KEY = "custom_files/hyperparameters.json"
        env_vars.DR_LOCAL_S3_MODEL_METADATA_KEY = "custom_files/model_metadata.json"
        env_vars.DR_LOCAL_S3_REWARD_KEY = "custom_files/reward_function.py"

        # Set AWS-related environment variables
        env_vars.DR_AWS_APP_REGION = self.config.aws.region
        env_vars.DR_LOCAL_ACCESS_KEY_ID = self.config.minio.access_key
        env_vars.DR_LOCAL_SECRET_ACCESS_KEY = self.config.minio.secret_key

        # Set DeepRacer-specific environment variables
        env_vars.DR_RUN_ID = self.config.deepracer.run_id  # This is already an int
        # Use default values from EnvVars class for these settings
        env_vars.DR_WORLD_NAME = env_vars.DR_WORLD_NAME  # Use default from EnvVars
        env_vars.DR_CAMERA_KVS_ENABLE = (
            env_vars.DR_CAMERA_KVS_ENABLE
        )  # Use default from EnvVars
        env_vars.DR_GUI_ENABLE = env_vars.DR_GUI_ENABLE  # Use default from EnvVars
        env_vars.DR_KINESIS_STREAM_NAME = (
            env_vars.DR_KINESIS_STREAM_NAME
        )  # Use default from EnvVars
        env_vars.DR_TRAIN_RTF = env_vars.DR_TRAIN_RTF  # Use default from EnvVars
        env_vars.DR_GAZEBO_ARGS = env_vars.DR_GAZEBO_ARGS  # Use default from EnvVars

        # Set RoboMaker command based on number of workers
        if workers > 1:
            env_vars.ROBOMAKER_COMMAND = (
                "/opt/simapp/run.sh multi distributed_training.launch"
            )
        else:
            env_vars.ROBOMAKER_COMMAND = (
                "/opt/simapp/run.sh run distributed_training.launch"
            )
        self.logger.info(f"ROBOMAKER_COMMAND set to: {env_vars.ROBOMAKER_COMMAND}")

        # Set current params file
        env_vars.DR_CURRENT_PARAMS_FILE = (
            self.config.deepracer.local_s3_training_params_file
        )

        # Set mount directories
        env_vars.DR_ROBOMAKER_MOUNT_SIMAPP_DIR = (
            "/opt/simapp"  # This is the default path in the container
        )
        env_vars.DR_ROBOMAKER_MOUNT_SCRIPTS_DIR = (
            "/scripts"  # This is the default path in the container
        )

        # Set environment variables
        for key, value in env_vars.__dict__.items():
            if value is not None:
                os.environ[key] = str(value)

        # Check for required environment variables
        required_vars = [
            "DR_SIMAPP_SOURCE",
            "DR_SIMAPP_VERSION",
            "DR_WORKERS",
            "DR_DOCKER_STYLE",
            "DR_ROBOMAKER_GUI_PORT",
            "DR_ROBOMAKER_TRAIN_PORT",
            "DR_LOCAL_S3_BUCKET",
            "DR_LOCAL_S3_MODEL_PREFIX",
            "DR_AWS_APP_REGION",
            "DR_RUN_ID",
            "DR_WORLD_NAME",
            "DR_ROBOMAKER_MOUNT_SIMAPP_DIR",  # Added this to required vars
            "DR_ROBOMAKER_MOUNT_SCRIPTS_DIR",  # Added this to required vars
        ]

        missing_vars = [var for var in required_vars if not os.environ.get(var)]

        if missing_vars:
            raise DockerError(
                f"Critical environment variables not set: {', '.join(missing_vars)}"
            )

        # Update CommandExecutor with current environment
        self.command_executor = CommandExecutor(env=os.environ.copy())
        if self.docker_style == DockerStyle.COMPOSE:
            self.manager = ComposeManager(self.project_name, self.command_executor)
        elif self.docker_style == DockerStyle.SWARM:
            self.manager = SwarmManager(self.project_name, self.command_executor)
        else:
            # Should not happen, but good to handle
            raise DockerError(f"Unknown docker style: {self.docker_style}")

    def _setup_multiworker_env(self) -> bool:
        """Set up environment for multiple workers.

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            num_workers = self.config.deepracer.num_workers
            if num_workers > 1:
                self._set_runtime_env_vars(num_workers)

                with self._managed_comms_dir(self.config.deepracer.run_id) as comms_dir:
                    os.environ["DR_DIR"] = str(comms_dir.parent.parent)
                    self.logger.info(
                        "setup_multiworker_env",
                        num_workers=num_workers,
                        comms_dir=str(comms_dir),
                    )
                return True
            return False
        except Exception as e:
            self.logger.error("multiworker_setup_failed", error=str(e))
            return False

    def _prepare_compose_files(self) -> Tuple[List[str], bool]:
        """Prepare necessary compose files.

        Returns:
            Tuple of (compose file paths, needs_multiworker)
        """
        # Use internal compose files from config/drfc-images
        training_compose_path = self.manager._get_compose_file_paths(
            [ComposeFileType.TRAINING]
        )[0]

        # Include AWS keys and endpoint overrides, then SimApp and Robomaker scripts
        file_types = [
            ComposeFileType.KEYS,
            ComposeFileType.ENDPOINT,
            ComposeFileType.SIMAPP,
            ComposeFileType.ROBOMAKER_SCRIPTS,
        ]

        if self.config.deepracer.robomaker_mount_logs:
            file_types.append(ComposeFileType.MOUNT)

        needs_multiworker = self.config.deepracer.num_workers > 1
        if needs_multiworker and self.docker_style != DockerStyle.SWARM:
            if self._setup_multiworker_env():
                file_types.append(ComposeFileType.ROBOMAKER_MULTI)
            else:
                needs_multiworker = False

        additional_files = self.manager._get_compose_file_paths(file_types)

        # Use base training compose file without injecting Redis
        final_files = [training_compose_path] + additional_files

        self.logger.info(
            "preparing_compose_files",
            file_types=[ft.value for ft in file_types],
            needs_multiworker=needs_multiworker,
            temp_file=training_compose_path,
        )

        return final_files, needs_multiworker

    def cleanup_previous_run(self, prune: bool = True) -> None:
        """Stop existing containers and optionally prune Docker resources.

        Args:
            prune: Whether to prune Docker resources
        """
        try:
            self.logger.info(
                "cleaning_up_previous_run", project=self.project_name, prune=prune
            )

            if self.docker_style == DockerStyle.SWARM:
                manager = cast(SwarmManager, self.manager)
                manager.remove()
            else:
                manager = cast(ComposeManager, self.manager)
                compose_files = manager._get_compose_file_paths(
                    [ComposeFileType.TRAINING]
                )
                manager.down(compose_files)

            if prune:
                self.command_executor.run_command(
                    ["docker", "network", "prune", "-f"], check=False
                )
                self.command_executor.run_command(
                    ["docker", "system", "prune", "-f"], check=False
                )

            time.sleep(2)

        except Exception as e:
            self.logger.error("cleanup_failed", error=str(e))
            raise DockerError(ERROR_MESSAGES["cleanup_failed"].format(str(e)))

    def start_deepracer_stack(self) -> None:
        """Start the DeepRacer Docker stack with all required services."""
        workers = self.config.deepracer.num_workers
        self.logger.info(
            f"Starting DeepRacer stack for project {self.project_name} with {workers} workers..."
        )

        # Set environment variables before any Docker operations
        self._set_runtime_env_vars(workers)

        compose_files, multi_added = self._prepare_compose_files()

        try:
            # (Removed Redis bootstrap; SimApp image handles Redis internally)

            # Start RL-Coach and RoboMaker services
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
            if workers > 1 and multi_added:
                cmd.extend(["--scale", f"robomaker={workers}"])
            elif workers > 1 and not multi_added:
                self.logger.warning(
                    "Not scaling RoboMaker because robomaker-multi config was not included."
                )
            self._run_command(cmd)

            self.logger.info(
                "started_deepracer_stack",
                project=self.project_name,
                needs_multiworker=multi_added,
                num_workers=workers,
            )

            self.check_container_status()

        except Exception as e:
            self.logger.error("start_stack_failed", error=str(e))
            raise DockerError(ERROR_MESSAGES["start_stack_failed"].format(str(e)))
        finally:
            # No temporary compose file cleanup needed; using internal static files
            pass

    def check_container_status(self) -> bool:
        """Check if the expected number of containers are running and healthy.

        Returns:
            True if all expected containers are running and healthy, False otherwise
        """
        try:
            time.sleep(5)

            if self.docker_style == DockerStyle.SWARM:
                manager = cast(SwarmManager, self.manager)
                services = manager.list_services()
                expected_count = (
                    self.config.deepracer.num_workers + 1
                )  # +1 for sagemaker
                running = len(services) == expected_count

                if running:
                    for service in services:
                        cmd = [
                            "docker",
                            "service",
                            "ps",
                            service,
                            "--format",
                            "{{.CurrentState}}",
                            "--filter",
                            "desired-state=running",
                        ]
                        result = self.command_executor.run_command(cmd, check=False)
                        if "Running" not in result.stdout:
                            running = False
                            break

                self.logger.info(
                    "checking_swarm_status",
                    expected=expected_count,
                    running=len(services),
                    healthy=running,
                )
            else:
                cmd = [
                    "docker",
                    "ps",
                    "--filter",
                    f"label=com.docker.compose.project={self.project_name}",
                    "--filter",
                    "label=com.docker.compose.service=robomaker",
                    "--filter",
                    "status=running",
                    "--format",
                    "{{.Names}} {{.Status}}",
                ]
                result = self.command_executor.run_command(cmd, check=False)
                containers = result.stdout.strip().splitlines() if result.stdout else []

                expected_workers = self.config.deepracer.num_workers
                running = len(containers) == expected_workers

                if running:
                    for container in containers:
                        if "healthy" not in container.lower():
                            running = False
                            break

                self.logger.info(
                    "checking_compose_status",
                    expected=expected_workers,
                    running=len(containers),
                    healthy=running,
                )

                if containers:
                    if running:
                        self.logger.info(
                            f"Successfully started {expected_workers} healthy RoboMaker workers"
                        )
                    else:
                        self.logger.warning(
                            f"Expected {expected_workers} healthy workers, but found {len(containers)} running"
                        )
                else:
                    self.logger.warning("No RoboMaker containers are running")

            return running

        except Exception as e:
            self.logger.error("status_check_failed", error=str(e))
            return False

    def check_logs(self, service_name: str, tail: int = 30) -> str:
        """Get logs for a specific service.

        Args:
            service_name: Name of the service
            tail: Number of lines to show

        Returns:
            Service logs
        """
        try:
            log_output: str
            if self.docker_style == DockerStyle.SWARM:
                # SwarmManager doesn't have a specific 'logs' method, so we build the command directly
                cmd = [
                    "docker",
                    "service",
                    "logs",
                    f"{self.project_name}_{service_name}",
                    "--tail",
                    str(tail),
                ]
                result = self.command_executor.run_command(cmd, check=False)
                log_output = (
                    result.stdout if result and result.stdout is not None else ""
                )
            else:  # DockerStyle.COMPOSE
                manager = cast(ComposeManager, self.manager)
                log_output = manager.logs(
                    service_name, tail
                )  # ComposeManager.logs returns str

            self.logger.debug("retrieved_service_logs", service=service_name, tail=tail)
            return log_output

        except Exception as e:
            self.logger.error(
                "log_retrieval_failed", service=service_name, error=str(e)
            )
            return f"Error retrieving logs: {str(e)}"
