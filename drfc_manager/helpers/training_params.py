# This script was directly based/copied from https://github.com/aws-deepracer-community/deepracer-for-cloud/blob/master/scripts/training/prepare-config.py

from .files_manager import create_folder
from datetime import datetime
import os
import yaml
from typing import Dict, List, Any
from drfc_manager.types.env_vars import EnvVars

env_vars = EnvVars()

def _setting_envs(train_time: str, model_name: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    config["AWS_REGION"] = env_vars.DR_AWS_APP_REGION
    config["JOB_TYPE"] = "TRAINING"
    config["KINESIS_VIDEO_STREAM_NAME"] = env_vars.DR_KINESIS_STREAM_NAME
    config["METRICS_S3_BUCKET"] = env_vars.DR_LOCAL_S3_BUCKET

    metrics_prefix = f"{env_vars.DR_LOCAL_S3_MODEL_PREFIX}/metrics"
    config["METRICS_S3_OBJECT_KEY"] = f"{metrics_prefix}/TrainingMetrics.json"

    config["MODEL_METADATA_FILE_S3_KEY"] = env_vars.DR_LOCAL_S3_MODEL_METADATA_KEY
    config["REWARD_FILE_S3_KEY"] = env_vars.DR_LOCAL_S3_REWARD_KEY
    config["ROBOMAKER_SIMULATION_JOB_ACCOUNT_ID"] = "Dummy"
    config["NUM_WORKERS"] = str(env_vars.DR_WORKERS)
    config["SAGEMAKER_SHARED_S3_BUCKET"] = env_vars.DR_LOCAL_S3_BUCKET
    config["SAGEMAKER_SHARED_S3_PREFIX"] = model_name
    config["SIMTRACE_S3_BUCKET"] = env_vars.DR_LOCAL_S3_BUCKET
    config["SIMTRACE_S3_PREFIX"] = model_name
    config["TRAINING_JOB_ARN"] = "arn:Dummy"
    
    config["BODY_SHELL_TYPE"] = env_vars.DR_CAR_BODY_SHELL_TYPE
    config["CAR_COLOR"] = env_vars.DR_CAR_COLOR
    config["CAR_NAME"] = env_vars.DR_CAR_NAME
    config["RACE_TYPE"] = env_vars.DR_RACE_TYPE
    config["WORLD_NAME"] = env_vars.DR_WORLD_NAME
    config["DISPLAY_NAME"] = env_vars.DR_DISPLAY_NAME
    config["RACER_NAME"] = env_vars.DR_RACER_NAME

    config["ALTERNATE_DRIVING_DIRECTION"] = str(env_vars.DR_TRAIN_ALTERNATE_DRIVING_DIRECTION)
    config["CHANGE_START_POSITION"] = str(env_vars.DR_TRAIN_CHANGE_START_POSITION)
    config["ROUND_ROBIN_ADVANCE_DIST"] = str(env_vars.DR_TRAIN_ROUND_ROBIN_ADVANCE_DIST)
    config["START_POSITION_OFFSET"] = str(env_vars.DR_TRAIN_START_POSITION_OFFSET)
    config["ENABLE_DOMAIN_RANDOMIZATION"] = str(env_vars.DR_ENABLE_DOMAIN_RANDOMIZATION)
    config["MIN_EVAL_TRIALS"] = str(env_vars.DR_TRAIN_MIN_EVAL_TRIALS)
    config["BEST_MODEL_METRIC"] = env_vars.DR_TRAIN_BEST_MODEL_METRIC
    config["REVERSE_DIR"] = str(env_vars.DR_TRAIN_REVERSE_DIRECTION)

    # Camera configuration
    config["CAMERA_MAIN_ENABLE"] = str(env_vars.DR_CAMERA_MAIN_ENABLE)
    config["CAMERA_SUB_ENABLE"] = str(env_vars.DR_CAMERA_SUB_ENABLE)

    # Handle race type specific configurations
    if env_vars.DR_RACE_TYPE == "OBJECT_AVOIDANCE":
        config["NUMBER_OF_OBSTACLES"] = str(env_vars.DR_OA_NUMBER_OF_OBSTACLES)
        config["MIN_DISTANCE_BETWEEN_OBSTACLES"] = str(env_vars.DR_OA_MIN_DISTANCE_BETWEEN_OBSTACLES)
        config["RANDOMIZE_OBSTACLE_LOCATIONS"] = str(env_vars.DR_OA_RANDOMIZE_OBSTACLE_LOCATIONS)
        config["IS_OBSTACLE_BOT_CAR"] = str(env_vars.DR_OA_IS_OBSTACLE_BOT_CAR)
        config["OBSTACLE_TYPE"] = env_vars.DR_OA_OBSTACLE_TYPE
        if env_vars.DR_OA_OBJECT_POSITIONS:
            config["OBJECT_POSITIONS"] = env_vars.DR_OA_OBJECT_POSITIONS

    elif env_vars.DR_RACE_TYPE == "HEAD_TO_BOT":
        config["IS_LANE_CHANGE"] = str(env_vars.DR_H2B_IS_LANE_CHANGE)
        config["LOWER_LANE_CHANGE_TIME"] = str(env_vars.DR_H2B_LOWER_LANE_CHANGE_TIME)
        config["UPPER_LANE_CHANGE_TIME"] = str(env_vars.DR_H2B_UPPER_LANE_CHANGE_TIME)
        config["LANE_CHANGE_DISTANCE"] = str(env_vars.DR_H2B_LANE_CHANGE_DISTANCE)
        config["NUMBER_OF_BOT_CARS"] = str(env_vars.DR_H2B_NUMBER_OF_BOT_CARS)
        config["MIN_DISTANCE_BETWEEN_BOT_CARS"] = str(env_vars.DR_H2B_MIN_DISTANCE_BETWEEN_BOT_CARS)
        config["RANDOMIZE_BOT_CAR_LOCATIONS"] = str(env_vars.DR_H2B_RANDOMIZE_BOT_CAR_LOCATIONS)
        config["BOT_CAR_SPEED"] = str(env_vars.DR_H2B_BOT_CAR_SPEED)
        config["PENALTY_SECONDS"] = str(env_vars.DR_H2B_BOT_CAR_PENALTY)

    return config


def writing_on_temp_training_yml(model_name: str) -> List[str]:
    try:
        train_time = datetime.now().strftime("%Y%m%d%H%M%S")
        config = _setting_envs(train_time, model_name)

        s3_prefix = config["SAGEMAKER_SHARED_S3_PREFIX"]

        s3_yaml_name = env_vars.DR_LOCAL_S3_TRAINING_PARAMS_FILE
        yaml_key = os.path.normpath(os.path.join(s3_prefix, s3_yaml_name))

        temp_dir = os.path.expanduser("~/dr_temp")
        create_folder(temp_dir)

        local_yaml_path = os.path.join(
            temp_dir, "training-params-" + train_time + ".yaml"
        )

        with open(local_yaml_path, "w") as yaml_file:
            yaml.dump(
                config,
                yaml_file,
                default_flow_style=False,
                default_style="'",
                explicit_start=True,
            )

        return [yaml_key, local_yaml_path]
    except Exception as e:
        raise e
