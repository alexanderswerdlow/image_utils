from typing import Union
from image_utils import Im, strip_unsafe
from PIL import Image
import torch
import numpy as np
import pytest
from pathlib import Path
from einops import rearrange, repeat

img_path = Path('tests/flower.jpg')
save_path = Path(__file__).parent / 'output'

def get_img(img_type: Union[np.ndarray, Image.Image, torch.Tensor], hwc_order = True, dtype=None, normalize=False, device=None, bw_img = False, batch_shape=None):
    if bw_img:
        if dtype is None:
            img = Image.fromarray(np.random.rand(128, 128) > 0.5)
        else:
            img = Image.fromarray(np.random.randint(256, size=(128, 128)).astype(dtype))
    else:
        img = Image.open(img_path)
    if img_type == Image.Image:
        return img
    
    img = np.array(img)
    if img_type == torch.Tensor:
        img = torch.from_numpy(img)

    if not hwc_order:
        img = rearrange(img, 'h w c -> c h w')

    if dtype is not None:
        img = img / 255.0
        if img_type == torch.Tensor:
            img = img.to(dtype=dtype)
        else:
            img = img.astype(dtype)

    if normalize:
        pass

    if device is not None and img_type == torch.Tensor:
        img = img.to(device=device)

    if batch_shape is not None:
        img = repeat(img, f'... -> {" ".join(sorted(list(batch_shape)))} ...', **batch_shape)

    return img

@pytest.mark.parametrize("dim_size", [4, 10, 100])
def test_single_arg_even(dim_size):
    dims = (dim_size, dim_size)
    rand_float_tensor = torch.FloatTensor(*dims).uniform_()
    rand_bool_tensor = torch.FloatTensor(*dims).uniform_() > 0.5
    rant_int_tensor = torch.randint(0, 100, dims)

    rand_float_array = np.random.rand(*dims)
    rand_bool_array = np.random.rand(*dims) > 0.5
    rand_int_array = np.random.randint(100, size=dims)

    print(rand_float_tensor)
    print(rand_bool_tensor)
    print(rant_int_tensor)

    print(rand_float_array)
    print(rand_bool_array)
    print(rand_int_array)

valid_configs = [
    {'img_type': Image.Image},
    {'img_type': np.ndarray},
    {'img_type': np.ndarray, 'hwc_order': False,},
    {'img_type': np.ndarray, 'dtype': np.float16,},
    {'img_type': np.ndarray, 'hwc_order': False, 'dtype': np.float16},
    {'img_type': np.ndarray, 'hwc_order': False, 'dtype': np.float32, 'normalize': True},
    {'img_type': torch.Tensor},
    {'img_type': torch.Tensor, 'hwc_order': False,},
    {'img_type': torch.Tensor, 'dtype': torch.float32,},
    {'img_type': torch.Tensor, 'hwc_order': False, 'dtype': torch.float16},
    {'img_type': torch.Tensor, 'hwc_order': False, 'dtype': torch.float, 'normalize': True},
    {'img_type': torch.Tensor, 'hwc_order': False, 'dtype': torch.float, 'normalize': True},
    {'img_type': np.ndarray, 'bw_img': True},
    {'img_type': np.ndarray, 'bw_img': True, 'dtype': np.uint8},

    {'img_type': np.ndarray, 'batch_shape': {'a': 2}},
    {'img_type': np.ndarray, 'batch_shape': {'a': 2, 'b': 3, 'c': 4}},
    {'img_type': np.ndarray, 'batch_shape': {'a': 2, 'b': 3}},
]

@pytest.mark.parametrize("img_params", valid_configs)
def test_write_text(img_params):
    img = Im(get_img(**img_params))
    file_path = save_path / strip_unsafe('__'.join([f'{k}_{v}' for k,v in img_params.items()]))
    img.copy.write_text('test').save(file_path.parent / f"{file_path.name}_text")

@pytest.mark.parametrize("img_params", valid_configs)
def test_add_border(img_params):
    img = Im(get_img(**img_params))
    file_path = save_path / strip_unsafe('__'.join([f'{k}_{v}' for k,v in img_params.items()]))
    img.copy.add_border(border = 5, color=(128, 128, 128)).save(file_path.parent / f"{file_path.name}_border")

@pytest.mark.parametrize("img_params", valid_configs)
def test_resize(img_params):
    img = Im(get_img(**img_params))
    file_path = save_path / strip_unsafe('__'.join([f'{k}_{v}' for k,v in img_params.items()]))
    img.copy.resize(128, 128).save(file_path.parent / f"{file_path.name}_resize")
    img.copy.scale(0.25).save(file_path.parent / f"{file_path.name}_downscale")
    img.copy.scale_to_width(128).save(file_path.parent / f"{file_path.name}_scale_width")
    img.copy.scale_to_height(128).save(file_path.parent / f"{file_path.name}_scale_height")

@pytest.mark.parametrize("img_params", valid_configs)
def test_normalization(img_params):
    img = Im(get_img(**img_params))
    file_path = save_path / strip_unsafe('__'.join([f'{k}_{v}' for k,v in img_params.items()]))
    img.normalize().denormalize().save(file_path.parent / f"{file_path.name}_normalize")
    img.denormalize().normalize().save(file_path.parent / f"{file_path.name}_normalize")