import os
import yaml
import tempfile
from typing import Dict, Any

from src.config import settings
from src.utils.docker.exceptions.base import DockerError

class RedisManager:
    def __init__(self, config=settings):
        self.config = config
    
    def add_redis_to_compose(self, compose_data: Dict[str, Any]) -> Dict[str, Any]:
        if 'services' not in compose_data:
            compose_data['services'] = {}
        
        compose_data['services']['redis'] = {
            'image': 'redis:alpine',
            'networks': {
                'default': {
                    'ipv4_address': self.config.redis.ip
                }
            }
        }
        
        if 'robomaker' in compose_data['services']:
            service = compose_data['services']['robomaker']
            
            if 'environment' not in service:
                service['environment'] = {}
            
            if isinstance(service['environment'], dict):
                service['environment']['REDIS_IP'] = self.config.redis.ip
                service['environment']['REDIS_PORT'] = self.config.redis.port
            elif isinstance(service['environment'], list):
                service['environment'].append(f'REDIS_IP={self.config.redis.ip}')
                service['environment'].append(f'REDIS_PORT={self.config.redis.port}')
                
            if 'depends_on' not in service:
                service['depends_on'] = ['redis']
            elif isinstance(service['depends_on'], list):
                if 'redis' not in service['depends_on']:
                    service['depends_on'].append('redis')
        
        compose_data['networks'] = {
            'default': {
                'external': True,
                'name': self.config.redis.network
            }
        }
        
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
            print(f"Created modified compose file with Redis at {temp_compose_path}")
            return temp_compose_path
        except Exception as e:
            os.remove(temp_compose_path)
            raise DockerError(f"Failed to write modified compose file '{temp_compose_path}': {e}")