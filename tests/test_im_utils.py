from typing import Iterable, Union
from image_utils import Im, strip_unsafe, concat_horizontal, concat_vertical
from PIL import Image
import torch
import numpy as np
import pytest
from pathlib import Path
from einops import rearrange, repeat

img_path = Path('tests/flower.jpg')
save_path = Path(__file__).parent / 'output'


def get_img(img_type: Union[np.ndarray, Image.Image, torch.Tensor], hwc_order=True, dtype=None, normalize=False, device=None, bw_img=False, batch_shape=None):
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
    {'img_type': np.ndarray, 'hwc_order': False, },
    {'img_type': np.ndarray, 'dtype': np.float16, },
    {'img_type': np.ndarray, 'hwc_order': False, 'dtype': np.float16},
    {'img_type': np.ndarray, 'hwc_order': False, 'dtype': np.float32, 'normalize': True},
    {'img_type': torch.Tensor},
    {'img_type': torch.Tensor, 'hwc_order': False, },
    {'img_type': torch.Tensor, 'dtype': torch.float32, },
    {'img_type': torch.Tensor, 'hwc_order': False, 'dtype': torch.float16},
    {'img_type': torch.Tensor, 'hwc_order': False, 'dtype': torch.bfloat16},
    {'img_type': torch.Tensor, 'hwc_order': False, 'dtype': torch.float, 'normalize': True},
    {'img_type': torch.Tensor, 'hwc_order': False, 'dtype': torch.float16, 'normalize': True},
    {'img_type': torch.Tensor, 'hwc_order': False, 'dtype': torch.bfloat16, 'normalize': True},
    {'img_type': np.ndarray, 'bw_img': True},
    {'img_type': np.ndarray, 'bw_img': True, 'dtype': np.uint8},
    {'img_type': np.ndarray, 'batch_shape': {'a': 2}},
    {'img_type': np.ndarray, 'batch_shape': {'a': 2, 'b': 3, 'c': 4}},
    {'img_type': np.ndarray, 'batch_shape': {'a': 2, 'b': 3}},
]

def get_file_path(img_params: dict, name: str):
    file_path = save_path / strip_unsafe('__'.join([f'{k}_{v}' for k, v in img_params.items()]))
    return file_path.parent / f"{file_path.name}_{name}"

@pytest.mark.parametrize("img_params", valid_configs)
def test_save(img_params):
    img = Im(get_img(**img_params))
    img.copy.save(get_file_path(img_params, 'save'))

@pytest.mark.parametrize("img_params", valid_configs)
def test_write_text(img_params):
    img = Im(get_img(**img_params))
    img.copy.write_text('test').save(get_file_path(img_params, 'text'))


@pytest.mark.parametrize("img_params", valid_configs)
def test_add_border(img_params):
    img = Im(get_img(**img_params))
    img.copy.add_border(border=5, color=(128, 128, 128)).save(get_file_path(img_params, 'border'))


@pytest.mark.parametrize("img_params", valid_configs)
def test_resize(img_params):
    img = Im(get_img(**img_params))
    img.copy.resize(128, 128).save(get_file_path(img_params, 'resize'))
    img.copy.scale(0.25).save(get_file_path(img_params, 'downscale'))
    img.copy.scale_to_width(128).save(get_file_path(img_params, 'scale_width'))
    img.copy.scale_to_height(128).save(get_file_path(img_params, 'scale_height'))


@pytest.mark.parametrize("img_params", valid_configs)
def test_normalization(img_params):
    img = Im(get_img(**img_params))
    if img_params.get('bw_img', False):
        return
    img.normalize().denormalize().save(get_file_path(img_params, 'normalize0'))
    img.denormalize().normalize().save(get_file_path(img_params, 'normalize1'))


@pytest.mark.parametrize("img_params", valid_configs)
def test_format(img_params):
    img = Im(get_img(**img_params))
    pil_img = img.pil
    torch_img = img.torch
    np_img = img.np
    cv_img = img.opencv


@pytest.mark.parametrize("img_params", valid_configs)
def test_concat(img_params):
    img = Im(get_img(**img_params))

    input_data = [img, img, img]
    if img_params.get('batch_shape', False):
        return

    concat_horizontal(*input_data, spacing=5)
    concat_vertical(*input_data, spacing=0)
