# This script was directly based/copied from https://github.com/aws-deepracer-community/deepracer-for-cloud/blob/master/scripts/training/prepare-config.py

from .files_manager import create_folder
from datetime import datetime
import os
import yaml


def _setting_envs(train_time: str, model_name: str) -> dict:
    config = {}

    config['AWS_REGION'] = os.environ.get('DR_AWS_APP_REGION', 'us-east-1')
    config['JOB_TYPE'] = 'TRAINING'
    config['KINESIS_VIDEO_STREAM_NAME'] = os.environ.get('DR_KINESIS_STREAM_NAME', '')
    config['METRICS_S3_BUCKET'] = os.environ.get('DR_LOCAL_S3_BUCKET', 'bucket')

    metrics_prefix = os.environ.get('DR_LOCAL_S3_METRICS_PREFIX', None)
    if metrics_prefix is not None:
        config['METRICS_S3_OBJECT_KEY'] = '{}/TrainingMetrics.json'.format(metrics_prefix)
    else:
        config['METRICS_S3_OBJECT_KEY'] = 'DeepRacer-Metrics/TrainingMetrics-{}.json'.format(train_time)

    config['MODEL_METADATA_FILE_S3_KEY'] = os.environ.get('DR_LOCAL_S3_MODEL_METADATA_KEY',
                                                          'custom_files/model_metadata.json')
    config['REWARD_FILE_S3_KEY'] = os.environ.get('DR_LOCAL_S3_REWARD_KEY', 'custom_files/reward_function.py')
    config['ROBOMAKER_SIMULATION_JOB_ACCOUNT_ID'] = os.environ.get('', 'Dummy')
    config['NUM_WORKERS'] = os.environ.get('DR_WORKERS', 1)
    config['SAGEMAKER_SHARED_S3_BUCKET'] = os.environ.get('DR_LOCAL_S3_BUCKET', 'bucket')
    config['SAGEMAKER_SHARED_S3_PREFIX'] = model_name
    config['SIMTRACE_S3_BUCKET'] = os.environ.get('DR_LOCAL_S3_BUCKET', 'bucket')
    config['SIMTRACE_S3_PREFIX'] = model_name
    config['TRAINING_JOB_ARN'] = 'arn:Dummy'

    config['BODY_SHELL_TYPE'] = os.environ.get('DR_CAR_BODY_SHELL_TYPE', 'deepracer')
    config['CAR_COLOR'] = os.environ.get('DR_CAR_COLOR', 'Red')
    config['CAR_NAME'] = os.environ.get('DR_CAR_NAME', 'MyCar')
    config['RACE_TYPE'] = os.environ.get('DR_RACE_TYPE', 'TIME_TRIAL')
    config['WORLD_NAME'] = os.environ.get('DR_WORLD_NAME', 'LGSWide')
    config['DISPLAY_NAME'] = os.environ.get('DR_DISPLAY_NAME', 'racer1')
    config['RACER_NAME'] = os.environ.get('DR_RACER_NAME', 'racer1')

    config['REVERSE_DIR'] = os.environ.get('DR_TRAIN_REVERSE_DIRECTION', False)
    config['ALTERNATE_DRIVING_DIRECTION'] = os.environ.get('DR_TRAIN_ALTERNATE_DRIVING_DIRECTION',
                                                           os.environ.get('DR_ALTERNATE_DRIVING_DIRECTION', 'false'))
    config['CHANGE_START_POSITION'] = os.environ.get('DR_TRAIN_CHANGE_START_POSITION',
                                                     os.environ.get('DR_CHANGE_START_POSITION', 'true'))
    config['ROUND_ROBIN_ADVANCE_DIST'] = os.environ.get('DR_TRAIN_ROUND_ROBIN_ADVANCE_DIST', '0.05')
    config['START_POSITION_OFFSET'] = os.environ.get('DR_TRAIN_START_POSITION_OFFSET', '0.00')
    config['ENABLE_DOMAIN_RANDOMIZATION'] = os.environ.get('DR_ENABLE_DOMAIN_RANDOMIZATION', 'false')
    config['MIN_EVAL_TRIALS'] = os.environ.get('DR_TRAIN_MIN_EVAL_TRIALS', '5')
    config['CAMERA_MAIN_ENABLE'] = os.environ.get('DR_CAMERA_MAIN_ENABLE', 'True')
    config['CAMERA_SUB_ENABLE'] = os.environ.get('DR_CAMERA_SUB_ENABLE', 'True')
    config['BEST_MODEL_METRIC'] = os.environ.get('DR_TRAIN_BEST_MODEL_METRIC', 'progress')

    return config


def writing_on_temp_training_yml(model_name: str) -> list:
    try:
        train_time = datetime.now().strftime('%Y%m%d%H%M%S')
        config = _setting_envs(train_time, model_name)

        s3_prefix = config['SAGEMAKER_SHARED_S3_PREFIX']

        s3_yaml_name = os.environ.get('DR_LOCAL_S3_TRAINING_PARAMS_FILE', 'training_params.yaml')
        yaml_key = os.path.normpath(os.path.join(s3_prefix, s3_yaml_name))

        create_folder('/tmp/dr/')

        local_yaml_path = os.path.join('/tmp/dr/', 'training-params-' + train_time + '.yaml')

        with open(local_yaml_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False, default_style='\'', explicit_start=True)

        return [yaml_key, local_yaml_path]
    except Exception as e:
        raise e
