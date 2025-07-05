from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class MinioConfig(BaseSettings):
    """MinIO S3 Storage Configuration"""

    model_config = SettingsConfigDict(env_prefix="MINIO_")

    server_url: str = Field(
        default="http://minio:9000", description="URL for the MinIO server"
    )
    access_key: str = Field(default="minioadmin", description="MinIO Access Key")
    secret_key: str = Field(default="minioadmin123", description="MinIO Secret Key")
    bucket_name: str = Field(
        default="tcc-experiments",
        description="Default bucket for DeepRacer models/files",
    )
    custom_files_folder: str = Field(
        default="custom_files",
        description="S3 prefix for custom files (reward fn, hyperparams)",
    )

    @validator("server_url", pre=True)
    def ensure_http_scheme(cls, v):
        if isinstance(v, str) and not v.startswith(("http://", "https://")):
            return f"http://{v}"
        return v


class DockerConfig(BaseSettings):
    """Docker and Container Configuration"""

    model_config = SettingsConfigDict(env_prefix="DOCKER_")

    # Docker daemon connection (optional)
    local_daemon_url: Optional[str] = Field(
        default=None,
        alias="LOCAL_SERVER_DOCKER_DAEMON",
        description="URL for local Docker daemon",
    )
    remote_daemon_url: Optional[str] = Field(
        default=None,
        alias="REMOTE_SERVER_DOCKER_DAEMON",
        description="URL for remote Docker daemon",
    )

    # Default Image Tags
    simapp_image: str = Field(
        default="awsdeepracercommunity/deepracer-simapp:5.3.3-gpu",
        alias="SIMAPP_IMAGE_REPOTAG",
        description="Default DeepRacer simulation image",
    )
    minio_image: str = Field(
        default="minio/minio:latest",
        alias="MINIO_IMAGE_REPOTAG",
        description="Default MinIO image",
    )

    # Docker style configuration
    docker_style: str = Field(
        default="compose",
        alias="DR_DOCKER_STYLE",
        description="Docker style to use (compose or swarm)",
    )
    dr_docker_file_sep: str = Field(
        default=" -f ", description="Separator used between docker compose files"
    )


class AWSConfig(BaseSettings):
    """AWS Configuration for DeepRacer Training"""

    region: str = "us-east-1"  # Default AWS region


class AppConfig(BaseSettings):
    """Main Application Configuration"""

    minio: MinioConfig = MinioConfig()
    docker: DockerConfig = DockerConfig()
    aws: AWSConfig = AWSConfig()

settings = AppConfig()
