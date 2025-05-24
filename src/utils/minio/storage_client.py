# src/storage/storage_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from src.types.hyperparameters import HyperParameters
from src.types.model_metadata import ModelMetadata
from typing import Callable

class StorageClient(ABC):
    """Abstract base class defining the storage client interface."""
    
    @property
    @abstractmethod
    def config(self) -> Any:
        """Get the storage configuration."""
        pass
    
    @abstractmethod
    def download_json(self, object_name: str) -> Dict:
        """Download and parse a JSON object."""
        pass
    
    @abstractmethod
    def download_py_object(self, object_name: str) -> str:
        """Download a Python file as text."""
        pass
    
    @abstractmethod
    def upload_hyperparameters(self, hyperparameters: HyperParameters, object_name: Optional[str] = None) -> None:
        """Upload hyperparameters."""
        pass
    
    @abstractmethod
    def upload_model_metadata(self, metadata: ModelMetadata, object_name: Optional[str] = None) -> None:
        """Upload model metadata."""
        pass
    
    @abstractmethod
    def upload_reward_function(self, reward_function: Union[Callable[[Dict], float], str], object_name: Optional[str] = None) -> None:
        """Upload reward function."""
        pass