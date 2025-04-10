import inspect
from typing import Callable, Dict

from minio.commonconfig import CopySource
from minio.deleteobjects import DeleteObject

from src.types.hyperparameters import HyperParameters

import io
import os
from minio import Minio as MinioClient
from minio.error import MinioException
from orjson import dumps, OPT_INDENT_2

from src.types.model_metadata import ModelMetadata
from src.utils.minio.exceptions.file_upload_exception import FileUploadException, FunctionConversionException

_bucket_name = os.getenv('BUCKET_NAME')
_custom_files_folder = os.getenv('CUSTOM_FILES_FOLDER_PATH')
_reward_function_path = os.getenv('REWARD_FUNCTION_PATH')


def upload_hyperparameters(
    minio_client: MinioClient,
    hyperparameters: HyperParameters,
    object_name: str = None
):
    """
    Uploads hyperparameters to an S3 bucket.

    Args:
        minio_client (MinioClient): The client of the minio api
        hyperparameters (HyperParameters): The hyperparameters to minio.
        object_name (str, optional): The full path to store the hyperparameters. If None, uses default path.

    Returns:
        bool: True if the minio was successful, False otherwise.
    """

    try:
        hyperparameters_serialized = dumps(hyperparameters, option=OPT_INDENT_2)
        object_size = len(hyperparameters_serialized)
        
        # Use provided path or default to custom_files_folder
        if object_name is None:
            object_name = f'{_custom_files_folder}/hyperparameters.json'

        result = minio_client.put_object(
            _bucket_name,
            object_name,
            io.BytesIO(hyperparameters_serialized),
            length=object_size,
            content_type="application/json"
        )

        return True if result else False
    except MinioException:
        raise FileUploadException(message=f'Error uploading hyperparameters file to S3 bucket')
    except Exception as e:
        raise FileUploadException(original_exception=e)


def upload_reward_function(
    minio_client: MinioClient,
    reward_function_buffer: io.BytesIO,
    object_name: str = None
):
    try:
        buffer_size = reward_function_buffer.getbuffer().nbytes
        
        # Use provided path or default to custom_files_folder
        if object_name is None:
            object_name = f'{_custom_files_folder}/reward_function.py'

        result = minio_client.put_object(
            _bucket_name,
            object_name,
            reward_function_buffer,
            length=buffer_size,
            content_type="text/plain"
        )

        return True if result else False
    except MinioException:
        raise FileUploadException(message=f'Error uploading reward function file to S3 bucket')
    except Exception as e:
        raise FileUploadException(original_exception=e)


def function_to_bytes_buffer(func: Callable[[Dict], float]) -> io.BytesIO:
    try:
        source_code = inspect.getsource(func)
        return io.BytesIO(source_code.encode('utf-8'))
    except Exception as e:
        raise FunctionConversionException(
            message="Failed to convert reward function to BytesIO.",
            original_exception=e
        )

def upload_metadata(
    minio_client: MinioClient,
    model_metadata: ModelMetadata,
    object_name: str = None
):
    """
    Uploads metadata to an S3 bucket.

    Args:
        minio_client (MinioClient): The client of the minio api
        model_metadata (Model Metadata): The metadata to minio.
        object_name (str, optional): The full path to store the metadata. If None, uses default path.

    Returns:
        bool: True if the minio was successful, False otherwise.
    """

    try:
        model_metadata_serialized = dumps(model_metadata, option=OPT_INDENT_2)
        object_size = len(model_metadata_serialized)
        
        # Use provided path or default to custom_files_folder
        if object_name is None:
            object_name = f'{_custom_files_folder}/model_metadata.json'

        result = minio_client.put_object(
            _bucket_name,
            object_name,
            io.BytesIO(model_metadata_serialized),
            length=object_size,
            content_type="application/json"
        )

        return True if result else False
    except MinioException:
        raise FileUploadException(message=f'Error uploading model metadata file to S3 bucket')
    except Exception as e:
        raise FileUploadException(original_exception=e)


def upload_local_data(minio_client: MinioClient, local_data_path: str, object_name):
    try:
        result = minio_client.fput_object(
            _bucket_name,
            object_name=object_name,
            file_path=local_data_path,
        )

        return True if result else False
    except MinioException:
        raise FileUploadException(message=f'Error uploading {object_name}  file to S3 bucket')
    except Exception as e:
        raise FileUploadException(original_exception=e)


def check_if_object_exists(minio_client: MinioClient, object_name: str):
    try:
        minio_client.stat_object(_bucket_name, object_name)
        return True
    except Exception:
        return False


def copy_object(minio_client: MinioClient, source_object_name: str, dest_object_name: str):
    try:
        minio_client.copy_object(
            _bucket_name,
            dest_object_name,
            CopySource(_bucket_name, source_object_name)
        )
        return True
    except MinioException:
        raise FileUploadException(message=f'Error copying {source_object_name} to {dest_object_name}')
    except Exception as e:
        raise FileUploadException(original_exception=e)


def remove_objects_folder(minio_client: MinioClient, object_name: str):
    try:
        objects = minio_client.list_objects(_bucket_name, prefix=object_name)
        objects_names = [object.object_name for object in objects]
        delete_objects = [DeleteObject(object_name) for object_name in objects_names]

        minio_client.remove_objects(
            _bucket_name,
            delete_objects,
        )
        return True
    except MinioException:
        raise FileUploadException(message=f'Error deleting {object_name} folder')
    except Exception as e:
        raise FileUploadException(original_exception=e)

