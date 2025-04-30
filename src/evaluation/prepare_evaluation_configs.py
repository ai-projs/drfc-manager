import os

import datetime
from typing import Dict, Any
from src.types.env_vars import EnvVars
from src.config import settings
from src.utils.str_to_bool import str2bool

def prepare_evaluation_config() -> Dict[str, Any]:
    """
    Generates the evaluation configuration dictionary by reading environment variables
    (previously set by setup_evaluation_env using EnvVars).
    """
    eval_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    config = {}

    # Helper to get env var, falling back to default from EnvVars if needed
    # This ensures we use the loaded values or the dataclass defaults
    env_defaults = EnvVars()
    def get_env(key, default=None):
        # Prioritize os.environ (which should be set by load_to_environment)
        # Fallback to dataclass default if not in os.environ (shouldn't happen often if loaded)
        # Final fallback to provided default
        return os.environ.get(key, getattr(env_defaults, key, default))

    # Initialize lists
    config['CAR_COLOR'] = []
    config['BODY_SHELL_TYPE'] = []
    config['RACER_NAME'] = []
    config['DISPLAY_NAME'] = []
    config['MODEL_S3_PREFIX'] = []
    config['MODEL_S3_BUCKET'] = []
    config['SIMTRACE_S3_BUCKET'] = []
    config['SIMTRACE_S3_PREFIX'] = []
    config['METRICS_S3_BUCKET'] = []
    config['METRICS_S3_OBJECT_KEY'] = []
    config['MP4_S3_BUCKET'] = []
    config['MP4_S3_OBJECT_PREFIX'] = []

    aws_region_fallback = getattr(getattr(settings, 'aws', object()), 'region', 'us-east-1')
    config['AWS_REGION'] = get_env('DR_AWS_APP_REGION', aws_region_fallback)
    config['JOB_TYPE'] = 'EVALUATION'
    config['KINESIS_VIDEO_STREAM_NAME'] = get_env('DR_KINESIS_STREAM_NAME', '')
    config['ROBOMAKER_SIMULATION_JOB_ACCOUNT_ID'] = 'Dummy'

    s3_bucket = get_env('DR_LOCAL_S3_BUCKET')
    model_prefix = get_env('DR_LOCAL_S3_MODEL_PREFIX')

    config['MODEL_S3_PREFIX'].append(model_prefix)
    config['MODEL_S3_BUCKET'].append(s3_bucket)
    config['SIMTRACE_S3_BUCKET'].append(s3_bucket)
    config['SIMTRACE_S3_PREFIX'].append(f'{model_prefix}/evaluation-{eval_time}')

    config['METRICS_S3_BUCKET'].append(s3_bucket)
    metrics_prefix = get_env('DR_LOCAL_S3_METRICS_PREFIX', f'{model_prefix}/metrics')
    config['METRICS_S3_OBJECT_KEY'].append(f'{metrics_prefix}/evaluation/evaluation-{eval_time}.json')

    save_mp4 = str2bool(get_env("DR_EVAL_SAVE_MP4", False))
    if save_mp4:
        config['MP4_S3_BUCKET'].append(s3_bucket)
        config['MP4_S3_OBJECT_PREFIX'].append(f'{model_prefix}/mp4/evaluation-{eval_time}')

    config['EVAL_CHECKPOINT'] = get_env('DR_EVAL_CHECKPOINT')
    config['BODY_SHELL_TYPE'].append(get_env('DR_CAR_BODY_SHELL_TYPE'))
    config['CAR_COLOR'].append(get_env('DR_CAR_COLOR'))
    config['DISPLAY_NAME'].append(get_env('DR_DISPLAY_NAME'))
    config['RACER_NAME'].append(get_env('DR_RACER_NAME'))
    config['RACE_TYPE'] = get_env('DR_RACE_TYPE')
    config['WORLD_NAME'] = get_env('DR_WORLD_NAME')
    config['NUMBER_OF_TRIALS'] = get_env('DR_EVAL_NUMBER_OF_TRIALS')
    config['ENABLE_DOMAIN_RANDOMIZATION'] = get_env('DR_ENABLE_DOMAIN_RANDOMIZATION')
    config['RESET_BEHIND_DIST'] = get_env('DR_EVAL_RESET_BEHIND_DIST')
    config['IS_CONTINUOUS'] = get_env('DR_EVAL_IS_CONTINUOUS')
    config['NUMBER_OF_RESETS'] = get_env('DR_EVAL_MAX_RESETS')
    config['OFF_TRACK_PENALTY'] = get_env('DR_EVAL_OFF_TRACK_PENALTY')
    config['COLLISION_PENALTY'] = get_env('DR_EVAL_COLLISION_PENALTY')
    config['CAMERA_MAIN_ENABLE'] = get_env('DR_CAMERA_MAIN_ENABLE')
    config['CAMERA_SUB_ENABLE'] = get_env('DR_CAMERA_SUB_ENABLE')
    config['REVERSE_DIR'] = str2bool(get_env('DR_EVAL_REVERSE_DIRECTION', False))
    config['ENABLE_EXTRA_KVS_OVERLAY'] = get_env('DR_ENABLE_EXTRA_KVS_OVERLAY', 'False')

    race_type = config['RACE_TYPE']
    if race_type == 'OBJECT_AVOIDANCE':
        config['NUMBER_OF_OBSTACLES'] = get_env('DR_OA_NUMBER_OF_OBSTACLES')
        config['MIN_DISTANCE_BETWEEN_OBSTACLES'] = get_env('DR_OA_MIN_DISTANCE_BETWEEN_OBSTACLES')
        config['RANDOMIZE_OBSTACLE_LOCATIONS'] = get_env('DR_OA_RANDOMIZE_OBSTACLE_LOCATIONS')
        config['IS_OBSTACLE_BOT_CAR'] = get_env('DR_OA_IS_OBSTACLE_BOT_CAR')
        config['OBSTACLE_TYPE'] = get_env('DR_OA_OBSTACLE_TYPE')
        object_position_str = get_env('DR_OA_OBJECT_POSITIONS', "")
        if object_position_str:
            object_positions = [o.strip() for o in object_position_str.split(";") if o.strip()]
            config['OBJECT_POSITIONS'] = object_positions
            config['NUMBER_OF_OBSTACLES'] = str(len(object_positions))

    elif race_type == 'HEAD_TO_BOT':
        config['IS_LANE_CHANGE'] = get_env('DR_H2B_IS_LANE_CHANGE')
        config['LOWER_LANE_CHANGE_TIME'] = get_env('DR_H2B_LOWER_LANE_CHANGE_TIME')
        config['UPPER_LANE_CHANGE_TIME'] = get_env('DR_H2B_UPPER_LANE_CHANGE_TIME')
        config['LANE_CHANGE_DISTANCE'] = get_env('DR_H2B_LANE_CHANGE_DISTANCE')
        config['NUMBER_OF_BOT_CARS'] = get_env('DR_H2B_NUMBER_OF_BOT_CARS')
        config['MIN_DISTANCE_BETWEEN_BOT_CARS'] = get_env('DR_H2B_MIN_DISTANCE_BETWEEN_BOT_CARS')
        config['RANDOMIZE_BOT_CAR_LOCATIONS'] = get_env('DR_H2B_RANDOMIZE_BOT_CAR_LOCATIONS')
        config['BOT_CAR_SPEED'] = get_env('DR_H2B_BOT_CAR_SPEED')
        config['PENALTY_SECONDS'] = get_env('DR_H2B_BOT_CAR_PENALTY')

    elif race_type == 'HEAD_TO_MODEL':
        opp_prefix = get_env('DR_EVAL_OPP_S3_MODEL_PREFIX')
        config['MODEL_S3_PREFIX'].append(opp_prefix)
        config['MODEL_S3_BUCKET'].append(s3_bucket)
        config['SIMTRACE_S3_BUCKET'].append(s3_bucket)
        config['SIMTRACE_S3_PREFIX'].append(f'{opp_prefix}/evaluation-{eval_time}')

        config['METRICS_S3_BUCKET'].append(s3_bucket)
        opp_metrics_prefix = f'{opp_prefix}/metrics'
        config['METRICS_S3_OBJECT_KEY'].append(f'{opp_metrics_prefix}/evaluation/evaluation-{eval_time}.json')

        if save_mp4:
            config['MP4_S3_BUCKET'].append(s3_bucket)
            config['MP4_S3_OBJECT_PREFIX'].append(f'{opp_prefix}/mp4/evaluation-{eval_time}')

        config['DISPLAY_NAME'].append(get_env('DR_EVAL_OPP_DISPLAY_NAME'))
        config['RACER_NAME'].append(get_env('DR_EVAL_OPP_RACER_NAME'))
        config['BODY_SHELL_TYPE'].append(get_env('DR_EVAL_OPP_CAR_BODY_SHELL_TYPE'))
        # config['CAR_COLOR'] = ['Purple', 'Orange'] # Example static colors from script
        config['CAR_COLOR'].append('Orange') # Assign second color for opponent
        config['MODEL_NAME'] = config['DISPLAY_NAME']

    return config