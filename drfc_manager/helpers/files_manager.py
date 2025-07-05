from glob import glob
import os
from typing import Optional


def create_folder(folder_name: str, mode: Optional[int] = None) -> None:
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name) if mode is None else os.makedirs(
                folder_name, mode=mode
            )
    except PermissionError:
        raise PermissionError(
            f"You don't have permission to create folder {folder_name} with permission {mode}"
        )
    except Exception as e:
        raise e


def delete_files_on_folder(folder_name: str) -> None:
    try:
        if os.path.exists(folder_name):
            files = glob(f"{folder_name}/*")
            for file in files:
                os.remove(file)
    except PermissionError:
        raise PermissionError(
            f"You don't have permission to delete folder {folder_name}"
        )
    except Exception as e:
        raise e
