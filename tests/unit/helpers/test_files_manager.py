import os
from drfc_manager.helpers.files_manager import create_folder, delete_files_on_folder


def test_create_folder(tmp_path):
    folder = tmp_path / "folder"
    create_folder(str(folder))
    assert os.path.isdir(str(folder))


def test_delete_files_on_folder(tmp_path):
    folder = tmp_path / "folder"
    os.makedirs(folder)
    # create files
    file1 = folder / "a.txt"
    file1.write_text("x")
    file2 = folder / "b.txt"
    file2.write_text("y")
    delete_files_on_folder(str(folder))
    assert list(folder.iterdir()) == []


def test_delete_nonexistent_folder(tmp_path):
    # Should not raise if folder does not exist
    delete_files_on_folder(str(tmp_path / "noexist"))
