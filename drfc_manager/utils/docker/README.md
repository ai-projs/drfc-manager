# Docker Manager

A modular and testable implementation for managing Docker operations in the DeepRacer training environment.

## Overview

The Docker Manager is responsible for handling Docker setup, execution, and cleanup for DeepRacer training. It supports both Docker Compose and Docker Swarm deployment styles, and provides a unified interface for managing Docker operations.

## Components

### Command Executor

The `CommandExecutor` class provides a base implementation for executing Docker commands. It handles command execution, output capture, and error handling.

### Compose Manager

The `ComposeManager` class handles Docker Compose operations, including:
- Starting and stopping services
- Managing compose files
- Scaling services
- Checking service status
- Retrieving service logs

Features:
- Context managers for safe resource handling
- Specific exception types for different error scenarios
- Structured logging for better debugging
- Command building with validation

### Swarm Manager

The `SwarmManager` class handles Docker Swarm operations, including:
- Deploying and removing stacks
- Managing services
- Checking stack status
- Retrieving service logs

Features:
- Context managers for safe resource handling
- Specific exception types for different error scenarios
- Structured logging for better debugging
- Command building with validation

### Docker Manager

The main `DockerManager` class integrates both Compose and Swarm managers, providing a unified interface for:
- Setting up the Docker environment
- Managing environment variables
- Handling multi-worker configurations
- Starting and stopping the DeepRacer stack
- Checking container status
- Retrieving logs

### Exception Handling

The implementation includes a comprehensive exception hierarchy:

- `DockerError`: Base exception for all Docker-related errors
  - `ComposeError`: For Docker Compose operation errors
  - `SwarmError`: For Docker Swarm operation errors
  - `ContainerError`: For container-related errors
  - `ImageError`: For image-related errors
  - `NetworkError`: For network-related errors
  - `VolumeError`: For volume-related errors
  - `ConfigError`: For configuration-related errors
  - `ResourceError`: For resource management errors

Each exception type includes:
- Detailed error messages
- Command context when available
- Resource identifiers (container ID, image name, etc.)
- Standard error output when available

## Usage

```python
from drfc_manager.utils.docker.docker_manager import DockerManager
from drfc_manager.utils.docker.exceptions import DockerError, ComposeError, SwarmError

# Initialize the Docker manager
config = {
    "docker_style": "compose",  # or "swarm"
    "num_workers": 1,
    "env_vars": {
        "TEST_VAR": "test_value"
    }
}
docker_manager = DockerManager(
    config=config,
    project_name="test-project"
)

try:
    # Start the DeepRacer stack
    docker_manager.start_deepracer_stack()

    # Check container status
    status = docker_manager.check_container_status()

    # Get logs for a service
    logs = docker_manager.check_logs("robomaker")

except ComposeError as e:
    print(f"Compose operation failed: {e}")
    if e.command:
        print(f"Failed command: {' '.join(e.command)}")
    if e.stderr:
        print(f"Error output: {e.stderr}")
except SwarmError as e:
    print(f"Swarm operation failed: {e}")
    if e.command:
        print(f"Failed command: {' '.join(e.command)}")
    if e.stderr:
        print(f"Error output: {e.stderr}")
except DockerError as e:
    print(f"Docker operation failed: {e}")
finally:
    # Clean up
    docker_manager.cleanup_previous_run()
```

## Directory Structure

```
drfc_manager/utils/docker/
├── __init__.py
├── command_executor.py
├── compose_manager.py
├── docker_constants.py
├── docker_manager.py
├── exceptions.py
├── swarm_manager.py
├── utilities.py
└── README.md
```

## Best Practices

1. **Resource Management**: Context managers are used throughout the codebase to ensure proper resource cleanup, even in error cases.

2. **Error Handling**: A comprehensive exception hierarchy provides detailed error information and context.

3. **Logging**: Structured logging is used throughout the implementation for better debugging and monitoring.

4. **Type Safety**: Type hints are used consistently to improve code reliability and IDE support.

5. **Documentation**: All components are well-documented with docstrings and type hints.

6. **Modularity**: Each component has a single responsibility and is designed to be used independently.

7. **Testability**: All components are designed with testability in mind, using dependency injection and mocking.