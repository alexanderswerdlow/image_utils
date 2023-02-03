from typing import Union
from image_utils import Im, default_lib_ops, delete_create_folder
from PIL import Image
import torch
import numpy as np
import pytest
from pathlib import Path
from einops import rearrange

def pytest_sessionstart(session):
    default_lib_ops()
    delete_create_folder(Path(__file__).parent / 'output')