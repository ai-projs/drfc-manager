from io import BytesIO
from typing import Callable, Dict, Optional, Union

from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource

from src.config import settings
from src.types.hyperparameters import HyperParameters
from src.types.model_metadata import ModelMetadata
from src.utils.minio.utilities import (
    function_to_bytes_buffer,
    serialize_hyperparameters,
    serialize_model_metadata
)
from src.utils.minio.exceptions.file_upload_exception import FileUploadException, FunctionConversionException

class StorageError(Exception):
    """Custom exception for storage-related errors."""
    pass

class MinioStorageManager:
    """Handles interactions with MinIO S3 storage."""

    def __init__(self, config: settings = settings):
        self.config = config.minio
        try:
            self.client = Minio(
                endpoint=str(self.config.server_url).replace('http://', '').replace('https://', ''), # Minio client needs host:port
                access_key=self.config.access_key,
                secret_key=self.config.secret_key.get_secret_value() if hasattr(self.config.secret_key, 'get_secret_value') else self.config.secret_key,
                secure=str(self.config.server_url).startswith('https')
            )
            # Check connection/bucket
            found = self.client.bucket_exists(self.config.bucket_name)
            if not found:
                self.client.make_bucket(self.config.bucket_name)
                print(f"Created MinIO bucket: {self.config.bucket_name}")
            else:
                print(f"Using existing MinIO bucket: {self.config.bucket_name}")

        except S3Error as e:
            raise StorageError(f"MinIO S3 Error: {e}") from e
        except Exception as e:
            raise StorageError(f"Failed to initialize MinIO client for endpoint {self.config.server_url}: {e}") from e

    def _upload_data(self, object_name: str, data: Union[bytes, BytesIO], length: int, content_type: str = 'application/octet-stream'):
        """Helper to upload data."""
        if isinstance(data, bytes):
            data = BytesIO(data)
        try:
            self.client.put_object(
                self.config.bucket_name,
                object_name,
                data,
                length=length,
                content_type=content_type
            )
            print(f"Successfully uploaded {object_name} to bucket {self.config.bucket_name}")
        except S3Error as e:
            raise StorageError(f"Failed to upload {object_name} to MinIO: {e}") from e
        except Exception as e: # Catch broader exceptions
             raise StorageError(f"Unexpected error during upload of {object_name}: {e}") from e

    def upload_hyperparameters(self, hyperparameters: HyperParameters, object_name: Optional[str] = None):
        """Uploads hyperparameters JSON."""
        if object_name is None:
            object_name = f"{self.config.custom_files_folder}/hyperparameters.json"
        try:
            data_bytes = serialize_hyperparameters(hyperparameters)
            self._upload_data(object_name, data_bytes, len(data_bytes), 'application/json')
        except Exception as e:
            raise FileUploadException("hyperparameters.json", str(e)) from e # Keep specific exception for compatibility

    def upload_model_metadata(self, model_metadata: ModelMetadata, object_name: Optional[str] = None):
        """Uploads model metadata JSON."""
        if object_name is None:
            object_name = f"{self.config.custom_files_folder}/model_metadata.json"
        try:
            data_bytes = serialize_model_metadata(model_metadata)
            self._upload_data(object_name, data_bytes, len(data_bytes), 'application/json')
        except Exception as e:
            raise FileUploadException("model_metadata.json", str(e)) from e

    def upload_reward_function(self, reward_function: Callable[[Dict], float], object_name: Optional[str] = None):
        """Uploads reward function Python code."""
        if object_name is None:
            object_name = f"{self.config.custom_files_folder}/reward_function.py"
        try:
            buffer = function_to_bytes_buffer(reward_function)
            self._upload_data(object_name, buffer, buffer.getbuffer().nbytes, 'text/x-python')
        except FunctionConversionException as e: # Catch specific conversion error
            raise e
        except Exception as e:
            raise FileUploadException("reward_function.py", str(e)) from e

    def upload_local_file(self, local_path: str, object_name: str):
        """Uploads a file from the local filesystem."""
        try:
            self.client.fput_object(self.config.bucket_name, object_name, local_path)
            print(f"Successfully uploaded local file {local_path} to {object_name}")
        except S3Error as e:
            raise StorageError(f"Failed to upload local file {local_path} to MinIO: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error uploading local file {local_path}: {e}") from e

    def object_exists(self, object_name: str) -> bool:
        """Checks if an object exists in the bucket."""
        try:
            self.client.stat_object(self.config.bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            raise StorageError(f"Failed to check object status for {object_name}: {e}") from e
        except Exception as e:
             raise StorageError(f"Unexpected error checking object {object_name}: {e}") from e

    def copy_object(self, source_object_name: str, dest_object_name: str):
        """Copies an object within the bucket."""
        try:
            # Create a proper CopySource object
            source = CopySource(self.config.bucket_name, source_object_name)
            
            self.client.copy_object(
                self.config.bucket_name,
                dest_object_name,
                source
            )
            print(f"Successfully copied {source_object_name} to {dest_object_name}")
        except Exception as e:
            raise StorageError(f"Unexpected error copying {source_object_name}: {str(e)}") from e
