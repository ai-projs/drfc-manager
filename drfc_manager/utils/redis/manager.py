import os
import yaml
import tempfile
import subprocess
from typing import Dict, Any

from drfc_manager.config_env import settings
from drfc_manager.utils.docker.exceptions.base import DockerError
from drfc_manager.utils.logging import logger

class RedisManager:
    def __init__(self, config=settings):
        self.config = config
    
    def _ensure_redis_network(self):
        """Create the Redis network with fixed IP pool if it doesn't exist."""
        network_name = self.config.redis.network
        subnet = self.config.redis.subnet
        
        check_cmd = ["docker", "network", "inspect", network_name]
        result = subprocess.run(check_cmd, check=False, capture_output=True)
        
        if result.returncode == 0:
            logger.info(f"Network '{network_name}' already exists.")
            return
            
        logger.info(f"Creating Docker network '{network_name}' with subnet {subnet}...")
        self._run_command(["docker", "network", "create", f"--subnet={subnet}", network_name])
    
    def _run_command(self, command: list, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        try:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            raise DockerError(f"Command failed: {e.cmd}, Error: {e.stderr}") from e
    
    def add_redis_to_compose(self, compose_data: Dict[str, Any]) -> Dict[str, Any]:
        if 'services' not in compose_data:
            compose_data['services'] = {}
        
        # Add Redis service on the existing network (use DNS and port, no fixed IP)
        compose_data['services']['redis'] = {
            'image': 'redis:alpine',
            'restart': 'always',
            'networks': ['default']
        }
        
        # Add Redis environment variables to both services
        for service_name in ['rl_coach', 'robomaker']:
            if service_name in compose_data['services']:
                service = compose_data['services'][service_name]
                
                if 'environment' not in service:
                    service['environment'] = {}
                
                if isinstance(service['environment'], dict):
                    service['environment']['REDIS_HOST'] = 'redis'
                    service['environment']['REDIS_PORT'] = self.config.redis.port
                elif isinstance(service['environment'], list):
                    service['environment'].append('REDIS_HOST=redis')
                    service['environment'].append(f'REDIS_PORT={self.config.redis.port}')
                
                if 'depends_on' not in service:
                    service['depends_on'] = ['redis']
                elif isinstance(service['depends_on'], list):
                    if 'redis' not in service['depends_on']:
                        service['depends_on'].append('redis')
        
        if 'version' in compose_data:
            del compose_data['version']
            
        return compose_data
    
    def create_modified_compose_file(self, training_compose_path: str) -> str:
        try:
            with open(training_compose_path, 'r') as file:
                compose_data = yaml.safe_load(file)
        except Exception as e:
            raise DockerError(f"Failed to load base training compose file '{training_compose_path}': {e}")

        temp_fd, temp_compose_path = tempfile.mkstemp(suffix='.yml', prefix='docker-compose-training-redis-')
        os.close(temp_fd)

        modified_compose_data = self.add_redis_to_compose(compose_data)
        
        try:
            with open(temp_compose_path, 'w') as file:
                yaml.dump(modified_compose_data, file)
            logger.info(f"Created modified compose file with Redis at {temp_compose_path}")
            return temp_compose_path
        except Exception as e:
            os.remove(temp_compose_path)
            raise DockerError(f"Failed to write modified compose file '{temp_compose_path}': {e}")