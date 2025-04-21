import os
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class EnvVars:
    # ------------------ run.env ------------------
    DR_RUN_ID: int = 0
    DR_WORLD_NAME: str = "reInvent2019_wide_ccw"
    DR_RACE_TYPE: str = "TIME_TRIAL"
    DR_CAR_NAME: str = "FastCar"
    DR_CAR_BODY_SHELL_TYPE: str = "deepracer"
    DR_CAR_COLOR: str = "Red"
    
    DR_DISPLAY_NAME: str = "FastCar"  # Copied from DR_CAR_NAME
    DR_RACER_NAME: str = "FastCar"      # Copied from DR_CAR_NAME

    DR_ENABLE_DOMAIN_RANDOMIZATION: bool = False

    # Evaluation parameters
    DR_EVAL_NUMBER_OF_TRIALS: int = 3
    DR_EVAL_IS_CONTINUOUS: bool = True
    DR_EVAL_MAX_RESETS: int = 100
    DR_EVAL_OFF_TRACK_PENALTY: float = 3.0
    DR_EVAL_COLLISION_PENALTY: float = 5.0
    DR_EVAL_SAVE_MP4: bool = True
    DR_EVAL_CHECKPOINT: str = "last"

    # Opponent configuration
    DR_EVAL_OPP_S3_MODEL_PREFIX: str = "rl-deepracer-sagemaker"
    DR_EVAL_OPP_CAR_BODY_SHELL_TYPE: str = "deepracer"
    DR_EVAL_OPP_CAR_NAME: str = "FasterCar"
    DR_EVAL_OPP_DISPLAY_NAME: str = "FasterCar"
    DR_EVAL_OPP_RACER_NAME: str = "FasterCar"

    DR_EVAL_DEBUG_REWARD: bool = True
    DR_EVAL_RESET_BEHIND_DIST: float = 1.0
    DR_EVAL_REVERSE_DIRECTION: bool = False

    # Training configuration
    DR_TRAIN_CHANGE_START_POSITION: bool = True
    DR_TRAIN_REVERSE_DIRECTION: bool = False
    DR_TRAIN_ALTERNATE_DRIVING_DIRECTION: bool = False
    DR_TRAIN_START_POSITION_OFFSET: float = 0.0
    DR_TRAIN_ROUND_ROBIN_ADVANCE_DIST: float = 0.05
    DR_TRAIN_MULTI_CONFIG: bool = False
    DR_TRAIN_MIN_EVAL_TRIALS: int = 5
    DR_TRAIN_BEST_MODEL_METRIC: str = "progress"
    DR_TRAIN_MAX_STEPS_PER_ITERATION: Optional[int] = None
    DR_TRAIN_RTF: Optional[int] = None

    # Model paths and S3 keys
    DR_LOCAL_S3_MODEL_PREFIX: str = "rl-deepracer-sagemaker"
    DR_LOCAL_S3_PRETRAINED: bool = False
    DR_LOCAL_S3_PRETRAINED_PREFIX: str = "explr-zg-offt-speed-borders-1"
    DR_LOCAL_S3_PRETRAINED_CHECKPOINT: str = "best"
    DR_LOCAL_S3_CUSTOM_FILES_PREFIX: str = "custom_files"
    DR_LOCAL_S3_TRAINING_PARAMS_FILE: str = "training_params.yaml"
    DR_LOCAL_S3_EVAL_PARAMS_FILE: str = "evaluation_params.yaml"
    DR_LOCAL_S3_MODEL_METADATA_KEY: str = "custom_files/model_metadata.json"
    DR_LOCAL_S3_HYPERPARAMETERS_KEY: str = "custom_files/hyperparameters.json"
    DR_LOCAL_S3_REWARD_KEY: str = "custom_files/reward_function.py"
    DR_LOCAL_S3_METRICS_PREFIX: str = "rl-deepracer-sagemaker/metrics"
    DR_UPLOAD_S3_PREFIX: str = "rl-deepracer-sagemaker"
    DR_MINIO_URL: str = "http://minio:9000"

    # Obstacle avoidance
    DR_OA_NUMBER_OF_OBSTACLES: int = 6
    DR_OA_MIN_DISTANCE_BETWEEN_OBSTACLES: float = 2.0
    DR_OA_RANDOMIZE_OBSTACLE_LOCATIONS: bool = False
    DR_OA_IS_OBSTACLE_BOT_CAR: bool = False
    DR_OA_OBSTACLE_TYPE: str = "box_obstacle"
    DR_OA_OBJECT_POSITIONS: str = ""

    # Head-to-bot
    DR_H2B_IS_LANE_CHANGE: bool = False
    DR_H2B_LOWER_LANE_CHANGE_TIME: float = 3.0
    DR_H2B_UPPER_LANE_CHANGE_TIME: float = 5.0
    DR_H2B_LANE_CHANGE_DISTANCE: float = 1.0
    DR_H2B_NUMBER_OF_BOT_CARS: int = 3
    DR_H2B_MIN_DISTANCE_BETWEEN_BOT_CARS: float = 2.0
    DR_H2B_RANDOMIZE_BOT_CAR_LOCATIONS: bool = False
    DR_H2B_BOT_CAR_SPEED: float = 0.2
    DR_H2B_BOT_CAR_PENALTY: float = 5.0

    # ------------------ system.env ------------------
    DR_CLOUD: str = "local"
    DR_AWS_APP_REGION: str = "us-east-1"
    DR_UPLOAD_S3_PROFILE: str = "default"
    DR_UPLOAD_S3_BUCKET: str = "deepracer-models-cloud-aws"
    DR_UPLOAD_S3_ROLE: str = "to-be-defined"
    DR_LOCAL_S3_BUCKET: str = "tcc-experiments"
    DR_LOCAL_S3_PROFILE: str = "minio"
    DR_LOCAL_ACCESS_KEY_ID: str = "minioadmin"
    DR_LOCAL_SECRET_ACCESS_KEY: str = "minioadmin123"
    DR_GUI_ENABLE: bool = False
    DR_KINESIS_STREAM_NAME: str = ""
    DR_CAMERA_MAIN_ENABLE: bool = True
    DR_CAMERA_SUB_ENABLE: bool = False
    DR_CAMERA_KVS_ENABLE: bool = True
    DR_SIMAPP_SOURCE: str = "awsdeepracercommunity/deepracer-simapp"
    DR_SIMAPP_VERSION: str = "5.3.3-gpu"
    DR_MINIO_IMAGE: str = "latest"
    DR_ANALYSIS_IMAGE: str = "cpu"
    DR_COACH_IMAGE: str = "5.2.1"
    DR_WORKERS: int = 1
    DR_ROBOMAKER_MOUNT_LOGS: bool = False
    DR_CLOUD_WATCH_ENABLE: bool = False
    DR_CLOUD_WATCH_LOG_STREAM_PREFIX: str = ""
    DR_DOCKER_STYLE: str = "compose"
    DR_HOST_X: bool = False

    # --- Resource Allocation & Ports ---
    DR_WEBVIEWER_PORT: int = 8100
    DR_ROBOMAKER_TRAIN_PORT: str = "8080"
    DR_ROBOMAKER_GUI_PORT: str = "5900"
    DR_SAGEMAKER_CUDA_DEVICES: str = ""
    DR_ROBOMAKER_CUDA_DEVICES: str = ""
    DR_GAZEBO_ARGS: str = ""

    # --- Telemetry ---
    DR_TELEGRAF_HOST: str = ""
    DR_TELEGRAF_PORT: str = ""
    
    DRFC_REPO_ABS_PATH: str = "/home/insightlab/deepracer/deepracer-for-cloud"

    def export_as_env_string(self) -> str:
        """Returns a single string with key=value pairs for all environment variables."""
        env_dict = {k: v for k, v in asdict(self).items() if v is not None}
        return " ".join(f"{key}={value}" for key, value in env_dict.items())

    def load_to_environment(self) -> None:
        """
        Loads all of the environment variables from this dataclass into os.environ.
        Only variables with a non-None value are loaded.
        """
        env_vars = asdict(self)
        for key, value in env_vars.items():
            # Convert boolean values to lowercase strings to be consistent with shell expectations
            if isinstance(value, bool):
                os.environ[key] = str(value).lower()
            elif value is not None:
                os.environ[key] = str(value)