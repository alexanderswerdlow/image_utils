import shutil
from pathlib import Path
import random
import string
from datetime import datetime
import pickle
from image_utils import Im


def delete_create_folder(path: Path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def get_rand_hex():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))


def get_date_time_str():
    return datetime.now().strftime("%Y_%m_%d-%H_%M")


def strip_unsafe(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c == '']).rstrip()


def save_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_pickle(obj, file_path: Path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def get_file_list(**kwargs): [f for f in get_files(**kwargs)]


def get_files(path: Path, recursive: bool = False, return_folders: bool = False, allowed_extensions=None):
    path = Path(path)

    if allowed_extensions or recursive:
        glob_str = "*" if allowed_extensions is None else f"*.[{''.join(allowed_extensions)}]*"
        iterator = path.rglob(glob_str) if recursive else path.glob(glob_str)
    else:
        iterator = path.iterdir()

    for file in iterator:
        if file.is_file() or return_folders:
            yield file


def get_images(recursive: bool = False, allowed_extensions=[".png", ".jpg", ".jpeg"]):
    for file in get_files(recursive=recursive, allowed_extensions=allowed_extensions):
        yield Im.open(file)
