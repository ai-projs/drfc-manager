from typing import List

from docker import APIClient as DockerClient
from gloe import partial_transformer, condition

from src.transformers.exceptions.base import BaseExceptionTransformers
from src.types.docker import DockerImages
from src.utils.commands.docker_compose import DockerComposeCommands
from src.utils.docker.utilities import check_if_image_has_container_running
from src.utils.minio.utilities import check_if_object_exists as _check_if_object_exists
from src.utils.minio.utilities import copy_object as _copy_object, remove_objects_folder as _remove_objects_folder

from minio import Minio as MinioClient
from minio.error import MinioException

ImageTag = str

docker_compose = DockerComposeCommands()


@partial_transformer
def image_tag_has_running_container(_, docker_client: DockerClient, image_tag: str) -> bool:
    is_running = check_if_image_has_container_running(docker_client, image_tag)
    return is_running


@partial_transformer
def images_tags_has_some_running_container(_, docker_client: DockerClient, image_tags: List[str | None]):
    is_running_containers = [check_if_image_has_container_running(docker_client, image_tag) for image_tag in image_tags]
    return any(is_running_containers)


@partial_transformer
def images_tags_has_running_container(_, docker_client: DockerClient, image_tags: List[str]):
    is_running_containers = [check_if_image_has_container_running(docker_client, image_tag) for image_tag in image_tags]
    return all(is_running_containers)


@partial_transformer
def echo(_, message: str):
    print(message)


@condition
def forward_condition(_condition: bool):
    return _condition


@partial_transformer
def check_if_model_exists(_, minio_client: MinioClient, model_name: str, overwrite: bool = False):
    try:
        file_to_check = '/reward_function.py'
        object_exists = _check_if_object_exists(minio_client, model_name + file_to_check)
        return object_exists if overwrite is False else not object_exists
    except MinioException as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers("It was not possible to check if the model exists", e)


@partial_transformer
def copy_object(_, minio_client: MinioClient, source_object_name: str, dest_object_name: str) -> bool:
    try:
        return _copy_object(minio_client, source_object_name, dest_object_name)
    except MinioException as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers(f"It was not possible to copy the object {source_object_name} to {dest_object_name}", e)


@partial_transformer
def remove_objects_folder(_, minio_client: MinioClient, object_name: str) -> bool:
    try:
        return _remove_objects_folder(minio_client, object_name)
    except MinioException as e:
        raise BaseExceptionTransformers(exception=e)
    except Exception as e:
        raise BaseExceptionTransformers(f"It was not possible remove the object {object_name}", e)


@partial_transformer
def up_composes(_, files_path: List[DockerImages]):
    docker_compose.up(files_path)
