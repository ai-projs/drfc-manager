from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, DirectoryPath, HttpUrl, validator

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()



class MinioConfig(BaseSettings):
    """MinIO S3 Storage Configuration"""
    model_config = SettingsConfigDict(env_prefix='MINIO_')

    server_url: HttpUrl = Field(default="http://minio:9000", description="URL for the MinIO server")
    access_key: str = Field(default="minioadmin", description="MinIO Access Key")
    secret_key: str = Field(default="minioadmin123", description="MinIO Secret Key")
    bucket_name: str = Field(default="tcc-experiments", description="Default bucket for DeepRacer models/files")
    custom_files_folder: str = Field(default="custom_files", description="S3 prefix for custom files (reward fn, hyperparams)")

    @validator('server_url', pre=True)
    def ensure_http_scheme(cls, v):
        if isinstance(v, str) and not v.startswith(('http://', 'https://')):
            return f"http://{v}"
        return v

class DockerConfig(BaseSettings):
    """Docker and Container Configuration"""
    model_config = SettingsConfigDict(env_prefix='DOCKER_')

    # Path to deepracer-for-cloud checkout (critical for multi-worker mounts)
    # Renamed from DRFC_REPO_ABS_PATH for clarity
    drfc_base_path: Optional[DirectoryPath] = Field(None, alias='DRFC_REPO_ABS_PATH', description="Absolute path to the deepracer-for-cloud repository")

    # Docker daemon connection (optional)
    local_daemon_url: Optional[str] = Field(None, alias='LOCAL_SERVER_DOCKER_DAEMON', description="URL for local Docker daemon")
    remote_daemon_url: Optional[str] = Field(None, alias='REMOTE_SERVER_DOCKER_DAEMON', description="URL for remote Docker daemon")

    # Default Image Tags
    simapp_image: str = Field("awsdeepracercommunity/deepracer-simapp:5.3.3-gpu", alias='SIMAPP_IMAGE_REPOTAG', description="Default DeepRacer simulation image")
    minio_image: str = Field("minio/minio:latest", alias='MINIO_IMAGE_REPOTAG', description="Default MinIO image")

class DeepRacerConfig(BaseSettings):
    """Runtime configuration for DeepRacer training, aligns with DR_* env vars"""
    model_config = SettingsConfigDict(env_prefix='DR_')

    run_id: int = Field(0, description="Identifier for the training/evaluation run")
    workers: int = Field(1, description="Number of RoboMaker workers")
    docker_style: str = Field("compose", description="Docker orchestration style ('compose' or 'swarm')")
    robomaker_mount_logs: bool = Field(False, description="Mount RoboMaker logs locally")

    # Default S3 paths (can be overridden by EnvVars dataclass per run)
    local_s3_model_prefix: str = "rl-deepracer-sagemaker"
    local_s3_bucket: str = "tcc-experiments"
    local_s3_training_params_file: str = "training_params.yaml"

    # Ports (defaults, will be adjusted by run_id)
    robomaker_gui_port_base: int = 6900
    robomaker_train_port_base: int = 9080

    pretrained_model: bool = False
    pretrained_s3_prefix: str = ''
    local_s3_model_prefix: str = ''

class RedisConfig(BaseSettings):
    """Redis configuration for DeepRacer Training."""
    model_config = SettingsConfigDict(env_prefix='REDIS_')

    host: str = Field(
        default="redis",
        description="Redis hostname (DNS service name in Docker network)"
    )
    port: int = Field(
        default=6379,
        description="Redis port"
    )

class AWSConfig(BaseSettings):
    """AWS Configuration for DeepRacer Training"""
    region: str = "us-east-1" # Default AWS region

class AppConfig(BaseSettings):
    """Main Application Configuration"""
    redis: RedisConfig = RedisConfig()
    minio: MinioConfig = MinioConfig()
    docker: DockerConfig = DockerConfig()
    deepracer: DeepRacerConfig = DeepRacerConfig()
    aws: AWSConfig = AWSConfig()

settings = AppConfig()
