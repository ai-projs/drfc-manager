import os
import time
from typing import Optional
from src.utils.compose.client import DockerComposeClient

def wait_for_container(service_filter: str, wait_time: int = 30) -> str:
    """
    Wait up to 'wait_time' seconds for a container whose name contains 'service_filter'
    using the DockerComposeClient from python_on_whales.
    
    Returns:
        Container ID (str) if found; otherwise, an empty string.
    """
    # Instantiate with an empty list of compose files to get a generic DockerClient instance.
    docker_client = DockerComposeClient(compose_files=[]).client
    container = None
    start_time = time.time()
    while time.time() - start_time < wait_time:
        # Use the correct attribute: containers.list() (note the 's')
        for c in docker_client.container.list():
            if service_filter.lower() in c.name.lower():
                container = c
                break
        if container:
            break
        time.sleep(1)
    return container.id if container else ""

def get_container_logs(container_id: str, tail: Optional[int] = None) -> str:
    """
    Retrieves the logs for a container given its container_id.
    
    Args:
        container_id (str): The Docker container ID.
        tail (int, optional): Number of last lines to retrieve; if None, retrieves full logs.
        
    Returns:
        The container's logs as a string.
    """
    docker_client = DockerComposeClient(compose_files=[]).client
    try:
        # Retrieve container logs (without following)
        logs = docker_client.container.logs(container_id, follow=False, tail=tail)
        # Decode if the returned logs are bytes
        if isinstance(logs, bytes):
            return logs.decode("utf-8")
        return logs
    except Exception as e:
        return f"Error retrieving logs: {e}"