import os
import subprocess
import time
import yaml
import tempfile
from typing import List

from src.config import settings
from src.utils.docker.utilities import _adjust_composes_file_names  

class DockerError(Exception):
    """Custom exception for Docker-related errors."""
    def __init__(self, message: str, command: List[str] = None, stderr: str = None):
        super().__init__(message)
        self.command = command
        self.stderr = stderr

    def __str__(self):
        msg = super().__str__()
        if self.command:
            msg += f"\nCommand: {' '.join(self.command)}"
        if self.stderr:
            msg += f"\nStderr:\n{self.stderr}"
        return msg

class DockerManager:
    """Handles Docker setup, execution, and cleanup for DeepRacer training."""

    def __init__(self, config: settings = settings):
        self.config = config
        self.project_name = f"deepracer-{self.config.deepracer.run_id}"

    def _run_command(self, command: List[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
        """Helper to run subprocess commands and handle errors."""
        print(f"Executing: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=capture,
                text=True,
                env=os.environ.copy() # Pass current environment
            )
            if capture:
                 if result.stdout: print(f"Stdout:\n{result.stdout}")
                 if result.stderr: print(f"Stderr:\n{result.stderr}") # Print warnings too
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
        """Stops existing project containers and optionally prunes Docker resources."""
        print(f"Cleaning up previous run for project {self.project_name}...")
        # Stop project
        self._run_command(["docker", "compose", "-p", self.project_name, "down", "--remove-orphans", "--volumes"], check=False)
        time.sleep(2)

        if prune_system:
            print("Pruning unused Docker resources...")
            self._run_command(["docker", "network", "prune", "-f"], check=False)
            self._run_command(["docker", "system", "prune", "-f"], check=False)
            time.sleep(2)

    def _create_redis_network(self):
        """Creates the sagemaker-local network with the required subnet if it doesn't exist."""
        network_name = "sagemaker-local"
        subnet = "10.0.1.0/24"
        # Check if network exists
        check_cmd = ["docker", "network", "inspect", network_name]
        result = subprocess.run(check_cmd, check=False, capture_output=True)
        if result.returncode == 0:
             print(f"Network '{network_name}' already exists.")
             # TODO: Potentially check if the subnet matches
             return

        print(f"Creating Docker network '{network_name}' with subnet {subnet}...")
        self._run_command(["docker", "network", "create", f"--subnet={subnet}", network_name])

    def _prepare_compose_files(self, workers: int) -> List[str]:
        """Determines the list of compose files and creates a modified training file with Redis."""
        # --- Modify training compose file to include Redis and correct network ---
        training_compose_path = _adjust_composes_file_names(['training'])[0] # Get path
        temp_compose_path = None
        try:
            with open(training_compose_path, 'r') as file:
                compose_data = yaml.safe_load(file)
        except Exception as e:
             raise DockerError(f"Failed to load base training compose file '{training_compose_path}': {e}")

        # Create a temp file
        temp_fd, temp_compose_path = tempfile.mkstemp(suffix='.yml', prefix='docker-compose-training-redis-')
        os.close(temp_fd)

        # Add Redis service
        if 'services' not in compose_data:
            compose_data['services'] = {}
        
        compose_data['services']['redis'] = {
            'image': 'redis:alpine',
            'networks': {
                'default': {
                    'ipv4_address': '10.0.1.15'  # Changed to 15 to match expected IP
                }
            }
        }
        
        # Handle robomaker service if it exists
        if 'robomaker' in compose_data['services']:
            service = compose_data['services']['robomaker']
            
            # Add environment if not exists
            if 'environment' not in service:
                service['environment'] = {}
            
            # Add Redis environment variables - handle both dict and list formats
            if isinstance(service['environment'], dict):
                service['environment']['REDIS_IP'] = '10.0.1.15'
                service['environment']['REDIS_PORT'] = '6379'
            elif isinstance(service['environment'], list):
                service['environment'].append('REDIS_IP=10.0.1.15')
                service['environment'].append('REDIS_PORT=6379')
                
            # Handle depends_on - it might be a list or not exist
            if 'depends_on' not in service:
                service['depends_on'] = ['redis']
            elif isinstance(service['depends_on'], list):
                if 'redis' not in service['depends_on']:
                    service['depends_on'].append('redis')
        
        # Ensure network is correctly defined
        compose_data['networks'] = {
            'default': {
                'external': True, # Use the pre-created network
                'name': 'sagemaker-local'
            }
        }
        
        # Remove obsolete version tag
        if 'version' in compose_data:
            del compose_data['version']

        try:
            with open(temp_compose_path, 'w') as file:
                yaml.dump(compose_data, file)
            print(f"Created modified compose file with Redis at {temp_compose_path}")
        except Exception as e:
             os.remove(temp_compose_path) # Clean up temp file on error
             raise DockerError(f"Failed to write modified compose file '{temp_compose_path}': {e}")

        # --- Determine other compose files ---
        compose_names_to_resolve = ["keys", "endpoint"] # Base files besides training
        multi_added = False
        if self.config.deepracer.robomaker_mount_logs:
            compose_names_to_resolve.append("mount")

        if workers > 1 and self.config.docker.docker_style != "swarm":
            if self.config.docker.drfc_base_path and os.path.isdir(self.config.docker.drfc_base_path):
                # Set DR_DIR env var needed by the compose file
                os.environ["DR_DIR"] = str(self.config.docker.drfc_base_path)
                comms_dir = os.path.join(self.config.docker.drfc_base_path, "tmp", f"comms.{self.config.deepracer.run_id}")
                try:
                    os.makedirs(os.path.dirname(comms_dir), exist_ok=True)
                    os.makedirs(comms_dir, exist_ok=True)
                    print(f"Created comms dir: {comms_dir}")
                    compose_names_to_resolve.append("robomaker-multi")
                    multi_added = True
                except OSError as e:
                    print(f"WARNING: Failed to create comms directory '{comms_dir}': {e}. Multi-worker may fail.")
            else:
                 print(f"WARNING: DRFC_REPO_ABS_PATH ('{self.config.docker.drfc_base_path}') is not set or invalid. Skipping robomaker-multi.")

        # Get full paths for the other files + the temp training file
        final_compose_files = [temp_compose_path] + _adjust_composes_file_names(compose_names_to_resolve)
        return final_compose_files, multi_added # Return flag if multi was added for scaling logic

    def start_deepracer_stack(self):
        """Starts the full DeepRacer Docker stack (including Redis)."""
        workers = self.config.deepracer.workers
        print(f"Starting DeepRacer stack for project {self.project_name} with {workers} workers...")

        # 1. Ensure Network Exists
        self._create_redis_network()

        # 2. Prepare Compose Files (creates temp file with Redis)
        compose_files, multi_added = self._prepare_compose_files(workers)
        temp_compose_path = compose_files[0] # The first one is the temp file
        print(f"Using compose files: {compose_files}")

        # 3. Set Runtime Environment Variables for Containers
        # These are needed *inside* the containers defined in the compose files
        os.environ["DR_ROBOMAKER_GUI_PORT"] = str(self.config.deepracer.robomaker_gui_port_base + self.config.deepracer.run_id)
        os.environ["DR_ROBOMAKER_TRAIN_PORT"] = str(self.config.deepracer.robomaker_train_port_base + self.config.deepracer.run_id)
        os.environ["DR_CURRENT_PARAMS_FILE"] = self.config.deepracer.local_s3_training_params_file
        os.environ["DR_RUN_ID"] = str(self.config.deepracer.run_id) # Ensure it's in env for compose $ substitution

        # Set Robomaker command based on workers
        if workers > 1:
            os.environ["ROBOMAKER_COMMAND"] = "/opt/simapp/run.sh multi distributed_training.launch"
        else:
            os.environ["ROBOMAKER_COMMAND"] = "/opt/simapp/run.sh run distributed_training.launch"
        print(f"ROBOMAKER_COMMAND set to: {os.environ.get('ROBOMAKER_COMMAND')}")

        # 4. Build Docker Compose Command
        cmd = ["docker", "compose"]
        for file in compose_files:
            cmd.extend(["-f", file])
        cmd.extend(["-p", self.project_name, "up", "-d", "--remove-orphans", "--force-recreate"])

        # Add scaling only if multiple workers AND multi config was added
        if workers > 1 and multi_added:
             cmd.extend(["--scale", f"robomaker={workers}"])
        elif workers > 1 and not multi_added:
             print(f"WARNING: Not scaling RoboMaker because robomaker-multi config was not included.")

        # 5. Execute Command
        try:
            self._run_command(cmd)
        finally:
            # 6. Clean up temporary file
            if temp_compose_path and os.path.exists(temp_compose_path):
                try:
                    os.remove(temp_compose_path)
                    print(f"Cleaned up temporary compose file {temp_compose_path}")
                except OSError as e:
                    print(f"Warning: Failed to remove temporary compose file {temp_compose_path}: {e}")

        # 7. Verify
        self.check_container_status(workers)

    def check_container_status(self, expected_workers: int):
        """Checks the status of containers in the project."""
        print("Checking container status...")
        time.sleep(5) # Wait for containers to potentially stabilize
        check_result = self._run_command(["docker", "compose", "-p", self.project_name, "ps"], check=False)

        robomaker_running_cmd = [
            "docker", "ps",
            "--filter", f"label=com.docker.compose.project={self.project_name}",
            "--filter", "label=com.docker.compose.service=robomaker",
            "--filter", "status=running", "-q"
        ]
        result = self._run_command(robomaker_running_cmd, check=False)
        running_ids = result.stdout.strip().splitlines() if result.stdout else []

        if running_ids:
            print(f"Found running RoboMaker containers:\n" + "\n".join(running_ids))
            if len(running_ids) == expected_workers:
                 print(f"Successfully started {expected_workers} RoboMaker workers.")
            else:
                 print(f"WARNING: Expected {expected_workers} RoboMaker workers, but found {len(running_ids)} running.")
        else:
            print("WARNING: No RoboMaker containers seem to be running.")
            # TODO: Check logs of failed containers?

    def check_logs(self, service_name: str, tail: int = 30):
        """Gets logs for a specific service."""
        print(f"\n--- Logs for {service_name} (tail {tail}) ---")
        cmd = ["docker", "compose", "-p", self.project_name, "logs", service_name, "--tail", str(tail)]
        self._run_command(cmd, check=False)
