# Image Utils

_Warning: This is a very alpha level library and was built for my personal use. The API is unstable and subject to change. Be careful when using this library for preprocessing._

Are you tired of having to constantly switch between NumPy arrays, PyTorch Tensors, and PIL images? Gone are the days of arr.transpose(1, 2, 0) or arr.permute(2, 0, 1) with image utils.

Simply wrap your NumPy array, PyTorch Tensor, or PIL image with `Im()` and let it handle conversions between formats.

## Installation

This package is not currently on PyPI. To install, use the git url:

```
pip install git+https://github.com/alexanderswerdlow/image_utils.git
```

## Usage

Below is an example of using the primary `Im` class:

```
import cv2
image = cv2.imread("tests/flower.jpg") # (h w c), np.uint8, [0, 255]

from image_utils import Im
image = Im(image).get_torch # (c h w), torch.float32, [0, 1]

from PIL import Image
image = Im(image).get_pil # PIL Image.Image
```

_Note:_ The `Im` class tries to preserve the input format when possible, but many functions will convert the input data to a different datatype (e.g. float -> uint8) causing loss of precision. For visualization this should not be an issue, but take care when using the library for pre-processing.

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

## Tests

To run tests run `pytest`

To break with pdb on error, run: `pytest --pdb -s`