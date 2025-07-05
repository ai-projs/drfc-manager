from io import BytesIO
from typing import Callable, Dict, Optional, Union
import json
import re

from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource

from drfc_manager.config_env import AppConfig
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.types.hyperparameters import HyperParameters
from drfc_manager.types.model_metadata import ModelMetadata
from drfc_manager.utils.minio.utilities import (
    function_to_bytes_buffer,
    serialize_hyperparameters,
    serialize_model_metadata,
)
from drfc_manager.utils.minio.exceptions.file_upload_exception import (
    FileUploadException,
    FunctionConversionException,
)
from drfc_manager.utils.minio.storage_client import StorageClient
from drfc_manager.utils.logging import logger
from drfc_manager.config_env import settings


env_vars = EnvVars()


class StorageError(Exception):
    """Custom exception for storage-related errors."""

    pass


class MinioStorageManager(StorageClient):
    """MinIO implementation of the storage client interface."""

    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            config = settings
        try:
            self.client = Minio(
                endpoint=str(env_vars.DR_MINIO_URL_API)
                .replace("http://", "")
                .replace("https://", ""),
                access_key=env_vars.DR_LOCAL_ACCESS_KEY_ID,
                secret_key=env_vars.DR_LOCAL_SECRET_ACCESS_KEY,
                secure=str(env_vars.DR_MINIO_URL_API).startswith("https"),
            )
            # Check connection/bucket
            found = self.client.bucket_exists(env_vars.DR_LOCAL_S3_BUCKET)
            if not found:
                self.client.make_bucket(env_vars.DR_LOCAL_S3_BUCKET)
                logger.info(f"Created MinIO bucket: {env_vars.DR_LOCAL_S3_BUCKET}")
            else:
                logger.info(f"Using existing MinIO bucket: {env_vars.DR_LOCAL_S3_BUCKET}")

        except S3Error as e:
            raise StorageError(f"MinIO S3 Error: {e}") from e
        except Exception as e:
            raise StorageError(
                f"Failed to initialize MinIO client for endpoint {env_vars.DR_MINIO_URL_API}: {e}"
            ) from e

    def _upload_data(
        self,
        object_name: str,
        data: Union[bytes, BytesIO],
        length: int,
        content_type: str = "application/octet-stream",
    ):
        """Helper to upload data."""
        if isinstance(data, bytes):
            data = BytesIO(data)
        try:
            self.client.put_object(
                env_vars.DR_LOCAL_S3_BUCKET,
                object_name,
                data,
                length=length,
                content_type=content_type,
            )
            logger.info(
                f"Successfully uploaded {object_name} to bucket {env_vars.DR_LOCAL_S3_BUCKET}"
            )
        except S3Error as e:
            raise StorageError(f"Failed to upload {object_name} to MinIO: {e}") from e
        except Exception as e:
            raise StorageError(
                f"Unexpected error during upload of {object_name}: {e}"
            ) from e

    def upload_hyperparameters(
        self, hyperparameters: HyperParameters, object_name: Optional[str] = None
    ) -> None:
        """Upload hyperparameters JSON."""
        if object_name is None:
            object_name = f"{env_vars.DR_LOCAL_S3_CUSTOM_FILES_PREFIX}/hyperparameters.json"
        try:
            data_bytes = serialize_hyperparameters(hyperparameters)
            self._upload_data(
                object_name, data_bytes, len(data_bytes), "application/json"
            )
        except Exception as e:
            raise FileUploadException("hyperparameters.json", str(e)) from e

    def upload_model_metadata(
        self, model_metadata: ModelMetadata, object_name: Optional[str] = None
    ) -> None:
        """Upload model metadata JSON."""
        if object_name is None:
            object_name = f"{env_vars.DR_LOCAL_S3_CUSTOM_FILES_PREFIX}/model_metadata.json"
        try:
            data_bytes = serialize_model_metadata(model_metadata)
            self._upload_data(
                object_name, data_bytes, len(data_bytes), "application/json"
            )
        except Exception as e:
            raise FileUploadException("model_metadata.json", str(e)) from e

    def upload_reward_function(
        self,
        reward_function: Union[Callable[[Dict], float], str],
        object_name: Optional[str] = None,
    ) -> None:
        """Upload reward function Python code."""
        if object_name is None:
            object_name = f"{env_vars.DR_LOCAL_S3_CUSTOM_FILES_PREFIX}/reward_function.py"
        try:
            if isinstance(reward_function, str):
                match = re.search(
                    r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                    reward_function,
                    flags=re.MULTILINE,
                )
                if match:
                    func_name = match.group(1)
                    alias = f"\n\n# Alias user-defined function to required name\nreward_function = {func_name}\n"
                    reward_str = reward_function + alias
                else:
                    reward_str = reward_function
                data_bytes = reward_str.encode("utf-8")
                self._upload_data(
                    object_name, data_bytes, len(data_bytes), "text/x-python"
                )
            else:
                buffer = function_to_bytes_buffer(reward_function)
                self._upload_data(
                    object_name, buffer, buffer.getbuffer().nbytes, "text/x-python"
                )
        except FunctionConversionException as e:
            raise e
        except Exception as e:
            raise FileUploadException("reward_function.py", str(e)) from e

    def upload_local_file(self, local_path: str, object_name: str):
        """Uploads a file from the local filesystem."""
        try:
            self.client.fput_object(env_vars.DR_LOCAL_S3_BUCKET, object_name, local_path)
            logger.info(
                f"Successfully uploaded local file {local_path} to {object_name}"
            )
        except S3Error as e:
            raise StorageError(
                f"Failed to upload local file {local_path} to MinIO: {e}"
            ) from e
        except Exception as e:
            raise StorageError(
                f"Unexpected error uploading local file {local_path}: {e}"
            ) from e

    def object_exists(self, object_name: str) -> bool:
        """Checks if an object exists in the bucket."""
        try:
            self.client.stat_object(env_vars.DR_LOCAL_S3_BUCKET, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise StorageError(
                f"Failed to check object status for {object_name}: {e}"
            ) from e
        except Exception as e:
            raise StorageError(
                f"Unexpected error checking object {object_name}: {e}"
            ) from e

    def copy_object(self, source_object_name: str, dest_object_name: str):
        """Copies an object within the bucket."""
        try:
            # Create a proper CopySource object
            source = CopySource(env_vars.DR_LOCAL_S3_BUCKET, source_object_name)

            self.client.copy_object(env_vars.DR_LOCAL_S3_BUCKET, dest_object_name, source)
            logger.info(
                f"Successfully copied {source_object_name} to {dest_object_name}"
            )
        except Exception as e:
            raise StorageError(
                f"Unexpected error copying {source_object_name}: {str(e)}"
            ) from e

    def copy_model_files(self, prefix: str, dest_prefix: str) -> None:
        """Copy model files from source prefix to destination prefix in S3."""
        objects = self.client.list_objects(
            env_vars.DR_LOCAL_S3_BUCKET, prefix=prefix, recursive=True
        )
        for obj in objects:
            src = obj.object_name
            dst = src.replace(prefix, dest_prefix, 1)
            self.copy_object(src, dst)
            logger.info(f"Copied {src} to {dst}")

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in the storage by looking for any object with the model prefix.
        """
        try:
            objects = self.client.list_objects(
                env_vars.DR_LOCAL_S3_BUCKET, prefix=f"{model_name}/", recursive=True
            )
            for _ in objects:
                return True
            return False
        except Exception as e:
            raise StorageError(
                f"Error checking if model {model_name} exists: {e}"
            ) from e

    def download_json(self, object_name: str) -> Dict:
        """Download and parse a JSON object."""
        try:
            response = self.client.get_object(env_vars.DR_LOCAL_S3_BUCKET, object_name)
            data = response.read().decode("utf-8")
            return json.loads(data)
        except Exception as e:
            raise StorageError(f"Error downloading object {object_name}: {e}")
        finally:
            if "response" in locals():
                response.close()
                response.release_conn()

    def download_py_object(self, object_name: str) -> str:
        """Download a Python file as text."""
        try:
            response = self.client.get_object(env_vars.DR_LOCAL_S3_BUCKET, object_name)
            return response.read().decode("utf-8")
        except Exception as e:
            raise StorageError(f"Error downloading object {object_name}: {e}")
        finally:
            if "response" in locals():
                response.close()
                response.release_conn()
