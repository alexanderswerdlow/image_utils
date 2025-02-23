# Image Utils

[![Run tests](https://github.com/alexanderswerdlow/image_utils/actions/workflows/ci.yml/badge.svg)](https://github.com/alexanderswerdlow/image_utils/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://image-utils.readthedocs.io/)
![Supported python versions](https://raw.githubusercontent.com/alexanderswerdlow/image_utils/master/docs/python_badge.svg)


Are you tired of having to constantly switch between NumPy arrays, PyTorch Tensors, and PIL images? Simply wrap your NumPy array, PyTorch Tensor, or PIL image with `Im()` and let it handle conversions between formats.

For example, we can replace this:
```
numpy_img = (torch.rand(10, 3, 256, 256).permute(0, 2, 3, 1).detach().float().cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(numpy_img[0]).save('output.png')
```

With this:
```
Im(torch.rand(10, 3, 256, 256)).save()
```

The powerful part is not that this works for this specific input shape/dtype/range combination, but [almost] any combination.

## Features

- Supports NumPy arrays, PyTorch Tensors, and PIL Images
- Handles arbitrary shapes `[..., H, W, C] or [..., C, H, W]` and preserves the input shape, batching all necessary transformations.
- Handles all common data types `[Float, Integer, Boolean]`, and Ranges `[0, 1], [0, 255]`
- Vertical/Horizontal concatenation of images with automatic padding, device conversion, and even batching
- Writing text on images
- Video encoding `[mp4, gif]` of a sequence of images
- Image normalization, resizing, and much more!

## Installation

_Warning_: The library is currently in alpha and the API is subject to change. If you use this library as part of another application, consider pinning to a specific commit, adding as a submodule, or even just taking the `src/image_utils/im.py` file as it works standalone!

This package is not currently on PyPI. To install, use the git url:

```
pip install git+https://github.com/alexanderswerdlow/image_utils.git
```

## Usage

Below is an example of using the primary `Im` class:

```
from image_utils import Im

img = np.random.randint(0, 256, (2, 10, 256, 256, 3), np.uint8)
img = Im(img)
img = img.write_text("Hello World!") # Writes the text on all 20 images
img = img.scale(2) # Scales image, preserving aspect ratio. Use resize(), scale_to_width(), or scale_to_width() for more control.
img = img.crop(200, 300, 0, 100)

# The Im class supports regular array slicing and unpacking! Here we concatenate the two [10, ...] into a single [10, ...] sequence of images
img = Im.concat_horizontal(*img, spacing=15) # Concatenation even works with varying shapes with automatic padding!
img.save() # Batched images are saved as a grid by default. Uses a timestamp for the name and PNG format by default. 
img.save_video() # We now have a 10 frame video!
```

## Extra Goodies

Another handy feature is provided by `library_ops`. This overrides the `__repr__` for NumPy arrays and PyTorch Tensors. For example:

```console
>>> import torch
>>> torch.randn(5, 5)
tensor([[-0.5524,  1.2306,  1.3209,  0.0336, -0.2458],
        [ 0.0448, -0.5564,  1.7019,  1.3689, -2.7115],
        [ 0.3842, -0.9593, -1.3799,  0.8625, -0.4071],
        [ 1.1263,  0.8479, -0.0585,  0.2687, -1.1983],
        [-0.5371, -0.5553, -0.7780, -0.8373,  0.2803]])
>>> from image_utils import library_ops
>>> torch.randn(5, 5)
[5,5] torch.float32 cpu finite
elems: 25, avg: 0.306, min: -1.399, max: 2.467
tensor([[ 0.782,  1.755,  0.975,  2.467, -0.646],
        [ 0.899,  2.344, -1.178, -0.291, -1.399],
        [ 0.676,  1.095,  0.289,  0.104, -0.294],
        [-0.152,  1.120, -0.844,  0.698,  0.647],
        [ 0.158, -0.048,  0.338, -0.838, -1.008]])
[5,5] torch.float32 cpu finite
```

Instead of only seeing the array contents, we can now view the shape, dtype, device, and more. `finite` or `infinite` signifies whether the array contains any `NaN` of `Inf` values.

If you want a dedicated library for this, check out [lovely-tensors](https://github.com/xl0/lovely-tensors)!

## When you should use image_utils

If you need to quickly visualize and work with images in a flexible way and aren't concerned with maximum efficiency

## When you shouldn't use image_utils

Currently, you shouldn't use image_utils in your pre-processing pipeline for a machine learning model. There is no guarantee that a given operation [e.g., resize] will have bit-perfect consistency between versions. Furthermore, image_utils focuses on flexibility over a wide-range of formats which in practice means frequent internal conversions and thus incurs additional overhead.

## Tests

*Note*: If you want to know more about how to use specific methods or which formats we test on, check out `tests/test_im_utils.py`.

To run all tests, simply run: `pytest`

To break with pdb on error, use: `pytest --pdb -s`

To run a specific test use: `pytest -k 'test_concat' --pdb -s`

## Local Installation

To install locally in a self-contained environment with [UV](https://docs.astral.sh/uv/):

```
git clone https://github.com/alexanderswerdlow/image_utils.git
uv sync --extra video --extra dev --extra cpu
```