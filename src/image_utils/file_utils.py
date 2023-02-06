import shutil
from pathlib import Path
import random
import string
from datetime import datetime
import pickle


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
