from dataclasses import dataclass, asdict, is_dataclass


@dataclass
class EnvVars:
    # ------------------ run.env ------------------
    DR_RUN_ID: int = 0  # Simulation run ID
    DR_WORLD_NAME: str = "reInvent2019_track_ccw"  # Track name
    DR_RACE_TYPE: str = "TIME_TRIAL"  # Race type (e.g., TIME_TRIAL)
    DR_CAR_NAME: str = "FastCar"  # Car name
    DR_CAR_BODY_SHELL_TYPE: str = "deepracer"  # Car shell type
    DR_CAR_COLOR: str = "Red"  # Car color

    DR_DISPLAY_NAME: str = "FastCar"  # Display name (copied from DR_CAR_NAME)
    DR_RACER_NAME: str = "FastCar"  # Racer name (copied from DR_CAR_NAME)

    DR_ENABLE_DOMAIN_RANDOMIZATION: bool = False  # Toggle domain randomization

    # Evaluation parameters
    DR_EVAL_NUMBER_OF_TRIALS: int = 3
    DR_EVAL_IS_CONTINUOUS: bool = True
    DR_EVAL_MAX_RESETS: int = 100
    DR_EVAL_OFF_TRACK_PENALTY: float = 5.0
    DR_EVAL_COLLISION_PENALTY: float = 5.0
    DR_EVAL_SAVE_MP4: bool = True
    DR_EVAL_CHECKPOINT: str = "last"

    # Opponent configuration
    DR_EVAL_OPP_S3_MODEL_PREFIX: str = "levi-with-speed-wp-3"
    DR_EVAL_OPP_CAR_BODY_SHELL_TYPE: str = "deepracer"
    DR_EVAL_OPP_CAR_NAME: str = "FastestCar"
    DR_EVAL_OPP_DISPLAY_NAME: str = "FastestCar"
    DR_EVAL_OPP_RACER_NAME: str = "FastestCar"

    DR_EVAL_DEBUG_REWARD: bool = False
    DR_EVAL_RESET_BEHIND_DIST: float = 1.0
    DR_EVAL_REVERSE_DIRECTION: bool = False
    # DR_EVAL_RTF: float = 1.0  # Commented out

    # Training configuration
    DR_TRAIN_CHANGE_START_POSITION: bool = True
    DR_TRAIN_REVERSE_DIRECTION: bool = False
    DR_TRAIN_ALTERNATE_DRIVING_DIRECTION: bool = False
    DR_TRAIN_START_POSITION_OFFSET: float = 0.0
    DR_TRAIN_ROUND_ROBIN_ADVANCE_DIST: float = 0.05
    DR_TRAIN_MULTI_CONFIG: bool = False
    DR_TRAIN_MIN_EVAL_TRIALS: int = 5
    DR_TRAIN_BEST_MODEL_METRIC: str = "progress"
    # DR_TRAIN_RTF: float = 1.0  # Commented out
    # DR_TRAIN_MAX_STEPS_PER_ITERATION: int = 10000  # Commented out

    # Model paths
    DR_LOCAL_S3_MODEL_PREFIX: str = "rl-deepracer-sagemaker"
    DR_LOCAL_S3_PRETRAINED: bool = False
    DR_LOCAL_S3_PRETRAINED_PREFIX: str = "rl-sagemaker-pretrained"
    DR_LOCAL_S3_PRETRAINED_CHECKPOINT: str = "last"
    DR_LOCAL_S3_CUSTOM_FILES_PREFIX: str = "custom_files"
    DR_LOCAL_S3_TRAINING_PARAMS_FILE: str = "training_params.yaml"
    DR_LOCAL_S3_EVAL_PARAMS_FILE: str = "evaluation_params.yaml"
    DR_LOCAL_S3_MODEL_METADATA_KEY: str = "custom_files/model_metadata.json"
    DR_LOCAL_S3_HYPERPARAMETERS_KEY: str = "custom_files/hyperparameters.json"
    DR_LOCAL_S3_REWARD_KEY: str = "custom_files/reward_function.py"
    DR_LOCAL_S3_METRICS_PREFIX: str = "levi-with-speed-wp-3/metrics"

    DR_UPLOAD_S3_PREFIX: str = "levi-with-speed-wp-3-1"

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
    DR_UPLOAD_S3_BUCKET: str = "not-defined"
    DR_UPLOAD_S3_ROLE: str = "to-be-defined"
    DR_LOCAL_S3_BUCKET: str = "tcc-experiments"
    DR_LOCAL_S3_PROFILE: str = "minio"
    DR_GUI_ENABLE: bool = False
    DR_KINESIS_STREAM_NAME: str = ""
    DR_CAMERA_MAIN_ENABLE: bool = True
    DR_CAMERA_SUB_ENABLE: bool = False
    DR_CAMERA_KVS_ENABLE: bool = True
    DR_ENABLE_EXTRA_KVS_OVERLAY: bool = False
    DR_SIMAPP_SOURCE: str = "awsdeepracercommunity/deepracer-simapp"
    DR_SIMAPP_VERSION: str = "5.3.3-gpu"
    DR_MINIO_IMAGE: str = "latest"
    DR_ANALYSIS_IMAGE: str = "cpu"
    DR_WORKERS: int = 3
    DR_ROBOMAKER_MOUNT_LOGS: bool = False
    # DR_ROBOMAKER_MOUNT_SIMAPP_DIR: str = ""  # Commented out
    # DR_ROBOMAKER_MOUNT_SCRIPTS_DIR: str = ""  # Commented out
    DR_CLOUD_WATCH_ENABLE: bool = False
    DR_CLOUD_WATCH_LOG_STREAM_PREFIX: str = ""
    DR_DOCKER_STYLE: str = "<DOCKER_STYLE>"
    DR_HOST_X: bool = False
    DR_WEBVIEWER_PORT: int = 8100
    # DR_DISPLAY: str = ":99"  # Commented out
    # DR_REMOTE_MINIO_URL: str = "http://mynas:9000"  # Commented out
    # DR_ROBOMAKER_CUDA_DEVICES: str = "0"  # Commented out
    # DR_SAGEMAKER_CUDA_DEVICES: str = "0"  # Commented out
    # DR_TELEGRAF_HOST: str = "telegraf"  # Commented out
    # DR_TELEGRAF_PORT: int = 8092  # Commented out

    def export_as_env_string(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.__dict__.items())