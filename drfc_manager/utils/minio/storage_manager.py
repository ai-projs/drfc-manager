from io import BytesIO
from typing import Callable, Dict, Optional, Union, Any
import json
import re

from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource

from drfc_manager.config_env import settings
from drfc_manager.types.hyperparameters import HyperParameters
from drfc_manager.types.model_metadata import ModelMetadata
from drfc_manager.utils.minio.utilities import (
    function_to_bytes_buffer,
    serialize_hyperparameters,
    serialize_model_metadata
)
from drfc_manager.utils.minio.exceptions.file_upload_exception import FileUploadException, FunctionConversionException
from drfc_manager.utils.minio.storage_client import StorageClient
from drfc_manager.utils.logging import logger

class StorageError(Exception):
    """Custom exception for storage-related errors."""
    pass

class MinioStorageManager(StorageClient):
    """MinIO implementation of the storage client interface."""

    def __init__(self, config: settings = settings):
        self._config = config.minio
        try:
            self.client = Minio(
                endpoint=str(self._config.server_url).replace('http://', '').replace('https://', ''),
                access_key=self._config.access_key,
                secret_key=self._config.secret_key.get_secret_value() if hasattr(self._config.secret_key, 'get_secret_value') else self._config.secret_key,
                secure=str(self._config.server_url).startswith('https')
            )
            # Check connection/bucket
            found = self.client.bucket_exists(self._config.bucket_name)
            if not found:
                self.client.make_bucket(self._config.bucket_name)
                logger.info(f"Created MinIO bucket: {self._config.bucket_name}")
            else:
                logger.info(f"Using existing MinIO bucket: {self._config.bucket_name}")

        except S3Error as e:
            raise StorageError(f"MinIO S3 Error: {e}") from e
        except Exception as e:
            raise StorageError(f"Failed to initialize MinIO client for endpoint {self._config.server_url}: {e}") from e

    @property
    def config(self) -> Any:
        """Get the storage configuration."""
        return self._config

    def _upload_data(self, object_name: str, data: Union[bytes, BytesIO], length: int, content_type: str = 'application/octet-stream'):
        """Helper to upload data."""
        if isinstance(data, bytes):
            data = BytesIO(data)
        try:
            self.client.put_object(
                self._config.bucket_name,
                object_name,
                data,
                length=length,
                content_type=content_type
            )
            logger.info(f"Successfully uploaded {object_name} to bucket {self._config.bucket_name}")
        except S3Error as e:
            raise StorageError(f"Failed to upload {object_name} to MinIO: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error during upload of {object_name}: {e}") from e

    def upload_hyperparameters(self, hyperparameters: HyperParameters, object_name: Optional[str] = None) -> None:
        """Upload hyperparameters JSON."""
        if object_name is None:
            object_name = f"{self._config.custom_files_folder}/hyperparameters.json"
        try:
            data_bytes = serialize_hyperparameters(hyperparameters)
            self._upload_data(object_name, data_bytes, len(data_bytes), 'application/json')
        except Exception as e:
            raise FileUploadException("hyperparameters.json", str(e)) from e

    def upload_model_metadata(self, model_metadata: ModelMetadata, object_name: Optional[str] = None) -> None:
        """Upload model metadata JSON."""
        if object_name is None:
            object_name = f"{self._config.custom_files_folder}/model_metadata.json"
        try:
            data_bytes = serialize_model_metadata(model_metadata)
            self._upload_data(object_name, data_bytes, len(data_bytes), 'application/json')
        except Exception as e:
            raise FileUploadException("model_metadata.json", str(e)) from e

    def upload_reward_function(self, reward_function: Union[Callable[[Dict], float], str], object_name: Optional[str] = None) -> None:
        """Upload reward function Python code."""
        if object_name is None:
            object_name = f"{self._config.custom_files_folder}/reward_function.py"
        try:
            if isinstance(reward_function, str):
                match = re.search(r'^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', reward_function, flags=re.MULTILINE)
                if match:
                    func_name = match.group(1)
                    alias = f"\n\n# Alias user-defined function to required name\nreward_function = {func_name}\n"
                    reward_str = reward_function + alias
                else:
                    reward_str = reward_function
                data_bytes = reward_str.encode('utf-8')
                self._upload_data(object_name, data_bytes, len(data_bytes), 'text/x-python')
            else:
                buffer = function_to_bytes_buffer(reward_function)
                self._upload_data(object_name, buffer, buffer.getbuffer().nbytes, 'text/x-python')
        except FunctionConversionException as e:
            raise e
        except Exception as e:
            raise FileUploadException("reward_function.py", str(e)) from e

    def upload_local_file(self, local_path: str, object_name: str):
        """Uploads a file from the local filesystem."""
        try:
            self.client.fput_object(self._config.bucket_name, object_name, local_path)
            logger.info(f"Successfully uploaded local file {local_path} to {object_name}")
        except S3Error as e:
            raise StorageError(f"Failed to upload local file {local_path} to MinIO: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error uploading local file {local_path}: {e}") from e

    def object_exists(self, object_name: str) -> bool:
        """Checks if an object exists in the bucket."""
        try:
            self.client.stat_object(self._config.bucket_name, object_name)
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
            source = CopySource(self._config.bucket_name, source_object_name)
            
            self.client.copy_object(
                self._config.bucket_name,
                dest_object_name,
                source
            )
            logger.info(f"Successfully copied {source_object_name} to {dest_object_name}")
        except Exception as e:
            raise StorageError(f"Unexpected error copying {source_object_name}: {str(e)}") from e

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in the storage by looking for any object with the model prefix.
        """
        try:
            objects = self.client.list_objects(
                self._config.bucket_name,
                prefix=f"{model_name}/",
                recursive=True
            )
            for _ in objects:
                return True
            return False
        except Exception as e:
            raise StorageError(f"Error checking if model {model_name} exists: {e}") from e

    def download_json(self, object_name: str) -> Dict:
        """Download and parse a JSON object."""
        try:
            response = self.client.get_object(
                self._config.bucket_name, 
                object_name
            )
            data = response.read().decode('utf-8')
            return json.loads(data)
        except Exception as e:
            raise StorageError(f"Error downloading object {object_name}: {e}")
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()
    
    def download_py_object(self, object_name: str) -> str:
        """Download a Python file as text."""
        try:
            response = self.client.get_object(
                self._config.bucket_name, 
                object_name
            )
            return response.read().decode('utf-8')
        except Exception as e:
            raise StorageError(f"Error downloading object {object_name}: {e}")
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()
