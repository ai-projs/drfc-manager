from docker import APIClient as DockerClient

ImageId = str
ImageName = str


def check_if_image_has_container_running(docker_client: DockerClient, image_tag: str) -> bool:
    containers = docker_client.containers(filters={"status": "running"})
    image_id = _find_image_id_by_name(docker_client, image_tag)
    container_id = [container["ImageID"] for container in containers if container["ImageID"] == image_id]
    
    if len(container_id) == 0:
        return False
    if container_id[0] == image_id:
        return True
    
    return False


def _find_image_id_by_name(docker_client: DockerClient, image_tag: str) -> str:
    try:
        images = docker_client.images()
        image_repo_tags = [image for image in images if image_tag in image["RepoTags"]]
        image_id = image_repo_tags[0]["Id"]
        
        return image_id
    except IndexError:
        raise IndexError(f"No image was found for tag {image_tag}")
