# Image Utils

_Warning: This is a very alpha level library and was built for my personal use. The API is unstable and subject to change._


Are you tired of having to constantly switch between NumPy arrays, PyTorch Tensors, and PIL images?

Gone are the days of arr.transpose(1, 2, 0) or arr.permute(2, 0, 1) with image utils.

Simply wrap your NumPy array, PyTorch Tensor, or PIL image with `Im()` and let it handle conversions between formats.

For example:

```
import cv2
image = cv2.imread("tests/flower.jpg") # (h w c), np.uint8, [0, 255]

from image_utils import Im
image = Im(image).get_torch # (c h w), torch.float32, [0, 1]

from PIL import Image
image = Im(image).get_pil # PIL Image.Image
```

To run tests run `pytest`
To break with pdb on error, run: `pytest --pdb -s`