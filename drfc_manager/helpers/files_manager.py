from glob import glob
import os


def create_folder(folder_name: str, mode: int = None):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name) if mode is None else os.makedirs(folder_name, mode=mode)
    except PermissionError:
        raise PermissionError(f'You don\'t have permission to create folder {folder_name} with permission {mode}')
    except Exception as e:
        raise e


def delete_files_on_folder(folder_name: str):
    try:
        if os.path.exists(folder_name):
            files = glob(f'{folder_name}/*')
            [os.remove(file) for file in files]
    except PermissionError:
        raise PermissionError(f'You don\'t have permission to delete folder {folder_name}')
    except Exception as e:
        raise e
