from __future__ import annotations

import copy
import string
import tempfile
import warnings
from enum import auto
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Type, TypeAlias, Union, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from einops import pack, rearrange, repeat
from jaxtyping import Bool, Float, Integer
from numpy import ndarray
from PIL import Image, ImageOps
from strenum import StrEnum
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode, resize

from image_utils.file_utils import get_date_time_str, load_cached_from_url
from image_utils.standalone_image_utils import pca, torch_to_numpy

if int(Image.__version__.split(".")[0]) >= 9 and int(Image.__version__.split(".")[1]) > 0:  # type: ignore
    resampling_module = Image.Resampling
else:
    resampling_module = Image

colorize_weights = {}
ImArr: TypeAlias = Union[ndarray, Tensor]  # The actual array itself
ImArrType: TypeAlias = Type[Union[ndarray, Tensor]]  # The object itself is just a type
ImDtype: TypeAlias = Union[torch.dtype, np.dtype]


def is_dtype(arr: ImArr, dtype: Union[Float, Integer, Bool]):
    return isinstance(arr, dtype[ndarray, "..."]) or isinstance(arr, dtype[Tensor, "..."])


def is_tensor(obj: ImArr):
    return torch.is_tensor(obj)


def is_ndarray(obj: ImArr):
    return isinstance(obj, ndarray)


def is_pil(obj: ImArr):
    return isinstance(obj, Image.Image)


def is_arr(obj: ImArr):
    return torch.is_tensor(obj) | isinstance(obj, ndarray)


def dispatch_op(obj: ImArr, np_op, torch_op, *args):
    if is_ndarray(obj):
        return np_op(obj, *args)
    elif is_tensor(obj):
        return torch_op(obj, *args)
    else:
        raise ValueError(f"obj must be numpy array or torch tensor, not {type(obj)}")


class ChannelOrder(StrEnum):
    HWC = auto()
    CHW = auto()


class ChannelRange(StrEnum):
    UINT8 = auto()
    FLOAT = auto()
    BOOL = auto()


class Im:
    """
    This class is a helper class to easily convert between formats (PIL/NumPy ndarray/PyTorch Tensor)
    and perform common operations, regardless of input dtype, batching, normalization, etc.

    Note: Be careful when using this class directly as part of a training pipeline. Many operations will cause the underlying data to convert between formats (e.g., Tensor -> Pillow) and move the data back to system memory and/or incur loss of precision (e.g., float -> uint8). In addition, we do not guarantee bit-consistency over different versions as we sometimes modify the internal representation or backend computation of a function.

    Some operations are in place operations even if they return an Im object.

    Specifically, we make the following design choices:

    - All inputs are internally represented by either a ndarray or tensor with shape (B, H, W, C) or (B, C, H, W)
    - Convinience is prioritized over efficiency. We make copies or perform in-place operation with little consistency. We may re-evaluate this design decision in the future.
    """

    default_normalize_mean = [0.4265, 0.4489, 0.4769]
    default_normalize_std = [0.2053, 0.2206, 0.2578]

    def __init__(self, arr: Union["Im", Tensor, Image.Image, ndarray], channel_range: Optional[ChannelRange] = None, **kwargs):
        # TODO: Add real URL checking here
        if isinstance(arr, (str, Path)) and Path(arr).exists():
            arr = Im.open(arr)  # type: ignore
        elif isinstance(arr, str):
            arr = Image.open(load_cached_from_url(arr))

        if isinstance(arr, Im):
            for attr in dir(arr):
                if not attr.startswith("__"):
                    setattr(self, attr, getattr(arr, attr))
            return

        self.device: torch.device
        self.arr_type: ImArrType

        # To handle things in a unified manner, we choose to always convert PIL Images -> NumPy internally
        if isinstance(arr, Image.Image):
            arr = np.array(arr)

        assert isinstance(arr, (ndarray, Tensor)), f"arr must be numpy array, pillow image, or torch tensor, not {type(arr)}"
        self.arr: ImArr = arr
        if isinstance(self.arr, ndarray):
            self.arr_type = ndarray
            self.device = torch.device("cpu")
        elif isinstance(self.arr, Tensor):
            self.device = self.arr.device
            self.arr_type = Tensor
            if self.arr.requires_grad:
                warnings.warn("Input tensor has requires_grad=True. We are detaching the tensor.")
                self.arr = self.arr.detach()
        else:
            raise ValueError("Must be numpy array, pillow image, or torch tensor")

        if len(self.arr.shape) == 2:  # Normalize to HWC
            self.arr = self.arr[..., None]

        # TODO: Consider normalizing to HWC order for all arrays, similar to how arr_transform works
        # These views should be very efficient and make things more unified
        self.channel_order: ChannelOrder = ChannelOrder.HWC if self.arr.shape[-1] < min(self.arr.shape[-3:-1]) else ChannelOrder.CHW
        self.dtype: ImDtype = self.arr.dtype
        self.shape = self.arr.shape

        # We normalize all arrays to (B, H, W, C) and record their original shape so
        # we can re-transform then when we need to output them
        if len(self.shape) == 3:
            self.arr_transform = lambda x: rearrange(x, "() a b c -> a b c")
        elif len(self.shape) == 4:
            self.arr_transform = lambda x: x
        elif len(self.shape) >= 5:
            extra_dims = self.shape[:-3]
            mapping = {k: v for k, v in zip(string.ascii_uppercase, extra_dims)}
            transform_str = f'({" ".join(sorted(list(mapping.keys())))}) a b c -> {" ".join(sorted(list(mapping.keys())))} a b c'
            self.arr_transform = lambda x: rearrange(x, transform_str, **mapping)
        else:
            raise ValueError("Must be between 3-5 dims")

        self.arr = rearrange(self.arr, "... a b c -> (...) a b c")

        # We use some very simple hueristics to guess what kind of image we have
        if channel_range is not None:  # Optionally specify the type
            self.channel_range = channel_range
        elif is_dtype(arr, Integer):
            assert self.arr.max() >= 0, "Integer array must be non-negative"
            if self.arr.max() > 1:
                self.channel_range = ChannelRange.UINT8
            else:  # We assume an integer array with 0s and 1s is a BW image
                self.channel_range = ChannelRange.BOOL
        elif is_dtype(arr, Float):
            if -128 <= self.arr.min() <= self.arr.max() <= 128:
                self.channel_range = ChannelRange.FLOAT
            else:
                raise ValueError("Not supported")
        elif is_dtype(arr, Bool):
            self.channel_range = ChannelRange.BOOL
        else:
            raise ValueError("Invalid Type")

    def __repr__(self):
        if self.arr_type == ndarray:
            arr_name = "ndarray"
        elif self.arr_type == Tensor:
            arr_name = "tensor"
        else:
            raise ValueError("Must be numpy array, pillow image, or torch tensor")

        if is_pil(self.arr):
            shape_str = repr(self.arr)
        else:
            shape_str = f"type: {arr_name}, shape: {self.shape}"

        return f"Im of {shape_str}, device: {self.device}"

    def _convert(
        self, desired_datatype: ImArrType, desired_order: ChannelOrder = ChannelOrder.HWC, desired_range: ChannelRange = ChannelRange.UINT8
    ) -> Im:
        if self.arr_type != desired_datatype or self.channel_order != desired_order or self.channel_range != desired_range:
            # We preserve the original dtype, shape, and device
            orig_shape, orig_transform, orig_device, orig_dtype = self.shape, self.arr_transform, self.device, self.arr.dtype

            if desired_datatype == ndarray:
                self = Im(self.get_np(order=desired_order, range=desired_range))
            elif desired_datatype == Tensor:
                self = Im(self.get_torch(order=desired_order, range=desired_range))

            self.device = orig_device
            self.arr_transform = orig_transform
            self.dtype = orig_dtype
            self.shape = orig_shape

        return self

    @staticmethod
    def _convert_to_datatype(desired_datatype: ImArrType, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8):
        def custom_decorator(func):
            def wrapper(self: Im, *args, **kwargs):
                self = self._convert(desired_datatype, desired_order, desired_range)
                return func(self, *args, **kwargs)

            return wrapper

        return custom_decorator

    def _handle_order_transform(self, im, desired_order: ChannelOrder, desired_range: ChannelRange, select_batch=None):
        if select_batch is not None:
            im = im[select_batch]
        else:
            im = self.arr_transform(im)

        if desired_order == ChannelOrder.CHW and self.channel_order == ChannelOrder.HWC:
            im = rearrange(im, "... h w c -> ... c h w")
        elif desired_order == ChannelOrder.HWC and self.channel_order == ChannelOrder.CHW:
            im = rearrange(im, "... c h w -> ... h w c")

        start_cur_order = "h w ()" if desired_order == ChannelOrder.HWC else "() h w"
        end_cur_order = start_cur_order.replace("()", "c")

        if self.channel_range != desired_range:
            assert is_ndarray(im) or is_tensor(im)
            if self.channel_range == ChannelRange.FLOAT and desired_range == ChannelRange.UINT8:
                im = im * 255
            elif self.channel_range == ChannelRange.UINT8 and desired_range == ChannelRange.FLOAT:
                im = im / 255.0
            elif self.channel_range == ChannelRange.BOOL and desired_range == ChannelRange.UINT8:
                assert self.channels == 1
                im = repeat(im, f"... {start_cur_order} -> ... {end_cur_order}", c=3) * 255
            elif self.channel_range == ChannelRange.BOOL and desired_range == ChannelRange.FLOAT:
                assert self.channels == 1
                im = repeat(im, f"... {start_cur_order} -> ... {end_cur_order}", c=3)
            else:
                raise ValueError("Not supported")

            if desired_range == ChannelRange.UINT8:
                im = im.astype(np.uint8) if is_ndarray(im) else im.to(torch.uint8)
            elif desired_range == ChannelRange.FLOAT:
                im = im.astype(np.float32) if is_ndarray(im) else im.to(torch.float32)

        return im

    def get_np(self, order=ChannelOrder.HWC, range=ChannelRange.UINT8) -> ndarray:
        arr = self.arr
        if is_tensor(arr):
            arr = torch_to_numpy(arr)

        arr = self._handle_order_transform(arr, order, range)

        return arr

    def get_torch(self, order=ChannelOrder.CHW, range=ChannelRange.FLOAT) -> Tensor:
        arr = self.arr
        if is_ndarray(arr):
            arr = torch.from_numpy(arr)

        arr = self._handle_order_transform(arr, order, range)
        if self.device is not None:
            arr = arr.to(self.device)
        return arr

    def get_pil(self) -> Union[Image.Image, list[Image.Image]]:
        if len(self.shape) == 3:
            return Image.fromarray(self.get_np())
        else:
            img = rearrange(self.get_np(), "... h w c -> (...) h w c")
            if img.shape[0] == 1:
                return Image.fromarray(img[0])
            else:
                return [Image.fromarray(img[i]) for i in range(img.shape[0])]

    @property
    def copy(self):
        return copy.deepcopy(self)

    @property
    def height(self):
        return self.image_shape[0]

    @property
    def width(self):
        return self.image_shape[1]

    @property
    def batch_size(self):
        # TODO: This is a bit hacky
        return self.arr.shape[0]

    @property
    def channels(self):
        """
        Returns number of channels in the image (e.g., 3 for RGB or 1 for BW)
        """
        if self.channel_order == ChannelOrder.HWC:
            return self.arr.shape[-1]
        else:
            return self.arr.shape[-3]

    @property
    def range_max(self):
        assert self.channel_range in (ChannelRange.UINT8, ChannelRange.FLOAT)
        return 255 if self.channel_range == ChannelRange.UINT8 else 1.0

    @property
    def image_shape(self):  # returns h,w
        if self.channel_order == ChannelOrder.HWC:
            return (self.arr.shape[-3], self.arr.shape[-2])
        else:
            return (self.arr.shape[-2], self.arr.shape[-1])

    @staticmethod
    def open(filepath: Path, use_imageio=False) -> Im:
        if use_imageio:
            from imageio import v3 as iio

            img = iio.imread(filepath)
        else:
            img = Image.open(filepath)
        return Im(img)

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.CHW, desired_range=ChannelRange.FLOAT)
    def resize(self, height, width, resampling_mode=InterpolationMode.BILINEAR):
        assert isinstance(self.arr, Tensor)
        return Im(resize(self.arr, [height, width], resampling_mode, antialias=True))

    def scale(self, scale) -> Im:
        width, height = self.width, self.height
        return self.resize(int(height * scale), int(width * scale))

    def scale_to_width(self, new_width) -> Im:
        width, height = self.width, self.height
        wpercent = new_width / float(width)
        hsize = int((float(height) * float(wpercent)))
        return self.resize(hsize, new_width)

    def scale_to_height(self, new_height) -> Im:
        width, height = self.width, self.height
        hpercent = new_height / float(height)
        wsize = int((float(width) * float(hpercent)))
        return self.resize(new_height, wsize)

    @staticmethod
    def _save_data(filepath: Path = Path(get_date_time_str()), filetype="png"):
        filepath = Path(filepath)
        if filepath.suffix == "":
            filepath = filepath.with_suffix(f".{filetype}")

        if len(filepath.parents) == 1:
            filepath = Path("output") / filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)

        return filepath

    def save(self, filepath: Path = Path(get_date_time_str()), filetype="png", optimize=False, quality=None):
        img = self.get_torch()

        filepath = Im._save_data(filepath, filetype)

        if len(img.shape) > 3:
            from torchvision import utils

            img = rearrange(img, "... h w c -> (...) h w c")
            img = utils.make_grid(img)
            img = Im(img).get_pil()
        else:
            img = self.get_pil()

        assert isinstance(img, Image.Image)

        flags = {"optimize": True, "quality": quality if quality else 0.95} if optimize or quality else {}

        img.save(filepath, **flags)

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def write_text(self, text: str) -> Im:
        for i in range(self.arr.shape[0]):
            text_to_write = text[i] if isinstance(text, list) else text
            assert isinstance(self.arr[i], ndarray)
            im = cv2.cvtColor(cast(ndarray, self.arr[i]), cv2.COLOR_RGB2BGR)
            im = cv2.putText(
                im,
                text_to_write,
                (0, im.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.002 * min(self.arr.shape[-3:-1]),
                (255, 0, 0),
                max(1, round(min(self.arr.shape[-3:-1]) / 150)),
                cv2.LINE_AA,
            )
            self.arr[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return self

    def add_border(self, border: int, color: Tuple[int, int, int]):
        """
        Adds solid color border to all sides of an image
        Args:
            border: size in px
            color: RGB tuple
        """
        imgs = self.pil
        if isinstance(imgs, Iterable):
            imgs = Im(np.stack([Im(ImageOps.expand(img, border=border, fill=color)).np for img in imgs], axis=0))
        else:
            imgs = Im(ImageOps.expand(imgs, border=border, fill=color))
        return imgs

    # def crop_image(self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0):
    #     raise NotImplementedError("")
    #     imgs = self.pil
    #     if isinstance(imgs, Iterable):
    #         imgs = Im(np.stack([Im(ImageOps.expand(ImageOps.crop(img, border=border), border=border, fill=color)).np for img in imgs], axis=0))
    #     else:
    #         imgs = Im(ImageOps.expand(ImageOps.crop(imgs, border=border), border=border, fill=color))

    def normalize_setup(self, mean=default_normalize_mean, std=default_normalize_std):
        def convert_instance_np(arr_1, arr_2):
            assert isinstance(self.dtype, np.dtype)
            return np.array(arr_2).astype(self.dtype)

        def convert_instance_torch(arr_1, arr_2):
            assert isinstance(self.dtype, torch.dtype)
            if self.dtype in (torch.float16, torch.bfloat16, torch.half):
                return Tensor(arr_2).to(dtype=torch.float, device=self.device)
            else:
                return Tensor(arr_2).to(dtype=self.dtype, device=self.device)

        if is_ndarray(self.arr):
            self = Im(self.get_np(ChannelOrder.HWC, ChannelRange.FLOAT))
        elif is_tensor(self.arr):
            self = Im(self.get_torch(ChannelOrder.HWC, ChannelRange.FLOAT))

        mean = dispatch_op(self.arr, convert_instance_np, convert_instance_torch, mean)
        std = dispatch_op(self.arr, convert_instance_np, convert_instance_torch, std)

        return self, mean, std

    def normalize(self, normalize_min_max: bool = False, **kwargs) -> Im:
        if normalize_min_max:
            # TODO: Make this more general
            self = Im(self.get_np(ChannelOrder.HWC, ChannelRange.FLOAT))
            self.arr = self.arr / self.arr.max() - self.arr.min()
        else:
            self, mean, std = self.normalize_setup(**kwargs)
            self.arr = (self.arr - mean) / std
        return self

    def denormalize(self, clamp: Union[bool, tuple[float, float]] = (0, 1.0), **kwargs) -> Im:
        self, mean, std = self.normalize_setup(**kwargs)
        self.arr = (self.arr * std) + mean
        if isinstance(self.arr, ndarray):
            self.arr = self.arr.clip(*clamp) if clamp else self.arr
        elif isinstance(self.arr, Tensor):
            self.arr = self.arr.clamp(*clamp) if clamp else self.arr
        return self

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def get_opencv(self):
        return self.arr

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def convert_opencv_color(self, color: int):
        """E.g.,cv2.COLOR_RGB2BGR"""
        assert isinstance(self.arr, ndarray)
        self.arr = cv2.cvtColor(self.arr, color)

    @staticmethod
    def concat_vertical(*args, **kwargs) -> Im:
        """Concatenates images vertically (i.e. stacked on top of each other)"""
        return concat_variable(concat_vertical_, *args, **kwargs)

    @staticmethod
    def concat_horizontal(*args, **kwargs) -> Im:
        """Concatenates images horizontally (i.e. left to right)"""
        return concat_variable(concat_horizontal_, *args, **kwargs)

    def save_video(self, filepath: Path, fps: int, format="mp4"):
        filepath = Im._save_data(filepath, format)
        byte_stream = self.encode_video(fps, format)
        with open(filepath, "wb") as f:
            f.write(byte_stream.getvalue())

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def encode_video(self, fps: int, format="mp4") -> BytesIO:
        assert len(self.arr.shape) == 4, "Video data must be 4D (time, height, width, channels)"
        import imageio

        byte_stream = BytesIO()

        # TODO: We shouldn't need to write -> read. An imageio/ffmpeg issue is causing this.
        with tempfile.NamedTemporaryFile(suffix=f".{format}") as ntp:
            if format == "webm":
                writer = imageio.get_writer(
                    ntp.name, format="webm", codec="libvpx-vp9", pixelformat="yuv420p", output_params=["-lossless", "1"], fps=fps
                )
            elif format == "gif":
                writer = imageio.get_writer(ntp.name, format="GIF", mode="I", duration=(1000 * 1 / fps))
            elif format == "mp4":
                writer = imageio.get_writer(ntp.name, quality=10, pixelformat="yuv420p", codec="libx264", fps=fps)
            else:
                raise NotImplementedError(f"Format {format} not implemented.")

            for frame in self.arr:
                writer.append_data(frame)

            writer.close()
            with open(ntp.name, "rb") as f:
                byte_stream.write(f.read())

        byte_stream.seek(0)
        return byte_stream

    def to(self, device: torch.device):
        assert isinstance(self.arr, Tensor), "Can only convert to device if array is a Tensor"
        self.arr = self.arr.to(device)
        return self

    @staticmethod
    def stack_imgs(*args: Im):
        imgs = [img._convert(desired_datatype=ndarray) if img.arr_type == Image.Image else img for img in args]
        return Im(
            rearrange(
                [img._handle_order_transform(img.arr, desired_order=ChannelOrder.HWC, desired_range=img.channel_range) for img in imgs],
                "b ... -> b ...",
            )
        )

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.CHW, desired_range=ChannelRange.FLOAT)
    def colorize(self) -> Im:
        if self.channels not in colorize_weights:
            colorize_weights[self.channels] = torch.randn(3, self.channels, 1, 1)

        assert isinstance(self.arr, Tensor)
        self.arr = F.conv2d(self.arr, weight=colorize_weights[self.channels])
        self.arr = (self.arr - self.arr.min()) / (self.arr.max() - self.arr.min())
        return self

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.FLOAT)
    def pca(self, **kwargs) -> Im:
        """
        Computes principal components over all "pixels" in the batched image array.
        You may optionally specify principal components.
        """
        assert isinstance(self.arr, Tensor)
        b, h, w, _ = self.arr.shape
        pca_arr: Tensor = rearrange(self.arr, "... c -> (...) c")
        output = pca(pca_arr, **kwargs)
        output: Tensor = rearrange(output, "(b h w) c -> b h w c", b=b, h=h, w=w)
        return Im(output)

    pil = property(get_pil)
    np = property(get_np)
    torch = property(get_torch)
    opencv = property(get_opencv)


def concat_variable(concat_func: Callable[..., Im], *args: Im, **kwargs) -> Im:
    if len(args) == 1 and isinstance(args[0], Iterable):
        args = args[0]

    output_img = None
    for img in args:
        if not isinstance(img, Im):
            img = Im(img)

        if img.channel_range == ChannelRange.BOOL:
            warnings.warn("Concatenating boolean images. We are converting to NumPy. This may cause unexpected behavior.")
            img = Im(img.np)

        if output_img is None:
            output_img = img
        else:
            if output_img.arr_type != img.arr_type or output_img.channel_order != img.channel_order or output_img.channel_range != img.channel_range:
                img = img._convert(
                    desired_datatype=output_img.arr_type, desired_order=output_img.channel_order, desired_range=output_img.channel_range
                )

            if output_img.device != img.device:
                warnings.warn("Concatenating images on different devices. We are moving both to CPU. This may cause unexpected behavior.")
                img = img.to(torch.device("cpu"))
                output_img = output_img.to(torch.device("cpu"))

            output_img = concat_func(output_img, img, **kwargs)

    assert isinstance(output_img, Im)
    return output_img


def get_arr_hwc(im: Im):
    assert im.batch_size == 1, "Must have batch size of 1"
    return im._handle_order_transform(im.arr, desired_order=ChannelOrder.HWC, desired_range=im.channel_range, select_batch=0)


def new_like(arr, shape, fill: Optional[tuple[int]] = None):
    if is_ndarray(arr):
        new_arr = np.zeros_like(arr, shape=shape)
    elif is_tensor(arr):
        new_arr = arr.new_zeros(shape)
    else:
        raise ValueError("Must be numpy array or torch tensor")

    if fill is not None:
        assert len(fill) == 3 and new_arr.shape[-1] == 3
        fill_ = fill if is_dtype(arr, Integer) else tuple(f / 255 for f in fill)
        new_arr[..., 0] = fill_[0]
        new_arr[..., 1] = fill_[1]
        new_arr[..., 2] = fill_[2]

    return new_arr


def concat_horizontal_(im1: Im, im2: Im, spacing: int = 0, **kwargs) -> Im:
    im1_arr = get_arr_hwc(im1)
    im2_arr = get_arr_hwc(im2)
    if im1.height != im2.height:
        warnings.warn(f"Images should have same height. Got {im1.height} and {im2.height}. Padding to match height.")
        if im1.height > im2.height:
            new_im2_arr = new_like(im1_arr, (im1_arr.shape[0], im2_arr.shape[1], im2_arr.shape[2]), **kwargs)
            new_im2_arr[: im2.height] = im2_arr
            im2_arr = new_im2_arr
        else:
            new_im1_arr = new_like(im2_arr, (im2_arr.shape[0], im1_arr.shape[1], im1_arr.shape[2]), **kwargs)
            new_im1_arr[: im1.height] = im1_arr
            im1_arr = new_im1_arr

    if spacing > 0:
        new_im2_arr = new_like(im2_arr, (im2_arr.shape[0], im2_arr.shape[1] + spacing, im2_arr.shape[2]), **kwargs)
        new_im2_arr[:, spacing:] = im2_arr
        im2_arr = new_im2_arr

    return Im(pack([im1_arr, im2_arr], "h * c")[0])


def concat_vertical_(im1: Im, im2: Im, spacing: int = 0, fill: Optional[tuple[int]] = None, **kwargs) -> Im:
    im1_arr = get_arr_hwc(im1)
    im2_arr = get_arr_hwc(im2)
    if im1.width != im2.width:
        warnings.warn(f"Images should have same width. Got {im1.width} and {im2.width}. Padding to match width.")
        if im1.width > im2.width:
            new_im2_arr = new_like(im1_arr, (im2_arr.shape[0], im1_arr.shape[1], im2_arr.shape[2]), **kwargs)
            new_im2_arr[:, : im2.width] = im2_arr
            im2_arr = new_im2_arr
        else:
            new_im1_arr = new_like(im2_arr, (im1_arr.shape[0], im2_arr.shape[1], im1_arr.shape[2]), **kwargs)
            new_im1_arr[:, : im1.width] = im1_arr
            im1_arr = new_im1_arr

    if spacing > 0:
        new_im2_arr = new_like(im2_arr, (im2_arr.shape[0] + spacing, im2_arr.shape[1], im2_arr.shape[2]), **kwargs)
        new_im2_arr[spacing:, :] = im2_arr
        im2_arr = new_im2_arr

    return Im(pack([im1_arr, im2_arr], "* w c")[0])
