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

_Note:_ The `Im` class tries to preserve the input format when possible, but many functions will convert the input data to a different datatype, possibly causing loss of precision. For the primary purpose of vizualization this should not be an issue, but take care when using the library for pre-processing.

## Tests

To run tests run `pytest`

To break with pdb on error, run: `pytest --pdb -s`