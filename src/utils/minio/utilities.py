import inspect
import io
from typing import Callable, Dict
from src.types.hyperparameters import HyperParameters
from orjson import dumps, OPT_INDENT_2

from src.types.model_metadata import ModelMetadata
from src.utils.minio.exceptions.file_upload_exception import FileUploadException, FunctionConversionException

def serialize_hyperparameters(hyperparameters: HyperParameters) -> bytes:
    """Convert hyperparameters to JSON bytes."""
    return dumps(hyperparameters, option=OPT_INDENT_2)

def serialize_model_metadata(model_metadata: ModelMetadata) -> bytes:
    """Convert model metadata to JSON bytes."""
    return dumps(model_metadata, option=OPT_INDENT_2)

def function_to_bytes_buffer(func: Callable[[Dict], float]) -> io.BytesIO:
    try:
        source_code = inspect.getsource(func)
        alias_code = f"\n\n# Alias user-defined function to required name\nreward_function = {func.__name__}\n"
        combined_code = source_code + alias_code
        return io.BytesIO(combined_code.encode('utf-8'))
    except Exception as e:
        raise FunctionConversionException(
            message="Failed to convert reward function to BytesIO.",
            original_exception=e
        )