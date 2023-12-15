from __future__ import annotations

import colorsys
import copy
import string
import tempfile
from enum import auto
from io import BytesIO
from pathlib import Path
from typing import (Callable, Iterable, Optional, Tuple, Type, TypeAlias,
                    Union, cast)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from numpy import ndarray
import torchvision.transforms.functional as T
from einops import pack, rearrange, repeat
from jaxtyping import AbstractDtype, Float, Integer, Bool
from PIL import Image, ImageOps
from strenum import StrEnum
from torchvision.transforms.functional import InterpolationMode

from image_utils.file_utils import get_date_time_str

if int(Image.__version__.split(".")[0]) >= 9 and int(Image.__version__.split(".")[1]) > 0: # type: ignore
    resampling_module = Image.Resampling
else:
    resampling_module = Image

colorize_weights = {}

ImArr: TypeAlias = Union[ndarray, Tensor] # The actual array itself
ImArrType: TypeAlias = Type[Union[ndarray, Tensor]] # The object itself is just a type
ImDtype: TypeAlias = Union[torch.dtype, np.dtype]

def is_dtype(arr: ImArr, dtype: Union[Float, Integer, Bool]):
    return isinstance(arr, dtype[ndarray, '...']) or isinstance(arr, dtype[Tensor, '...'])

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
        raise ValueError(f'obj must be numpy array or torch tensor, not {type(obj)}')


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

    def __init__(self, arr: Union['Im', Tensor, Image.Image, ndarray], channel_range: Optional[ChannelRange] = None, **kwargs):
        # TODO: Add real URL checking here
        if isinstance(arr, (str, Path)) and Path(arr).exists():
            arr = Im.open(arr) # type: ignore
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

        assert isinstance(arr, (ndarray, Tensor)), f'arr must be numpy array, pillow image, or torch tensor, not {type(arr)}'
        self.arr: ImArr = arr
        if isinstance(self.arr, ndarray):
            self.arr_type = ndarray
            self.device = torch.device('cpu')
        elif isinstance(self.arr, Tensor):
            self.device = self.arr.device
            self.arr_type = Tensor
        else:
            raise ValueError('Must be numpy array, pillow image, or torch tensor')

        if len(self.arr.shape) == 2: # Normalize to HWC
            self.arr = self.arr[..., None]

        # TODO: Consider normalizing to HWC order for all arrays, similar to how arr_transform works
        # These views should be very efficient and make things more unified
        self.channel_order: ChannelOrder = ChannelOrder.HWC if self.arr.shape[-1] < min(self.arr.shape[-3:-1]) else ChannelOrder.CHW
        self.dtype: ImDtype = self.arr.dtype
        self.shape = self.arr.shape
        
        # We normalize all arrays to (B, H, W, C) and record their original shape so
        # we can re-transform then when we need to output them
        if len(self.shape) == 3:
            self.arr_transform = lambda x: rearrange(x, '() a b c -> a b c')
        elif len(self.shape) == 4:
            self.arr_transform = lambda x: x
        elif len(self.shape) >= 5:
            extra_dims = self.shape[:-3]
            mapping = {k: v for k, v in zip(string.ascii_uppercase, extra_dims)}
            transform_str = f'({" ".join(sorted(list(mapping.keys())))}) a b c -> {" ".join(sorted(list(mapping.keys())))} a b c'
            self.arr_transform = lambda x: rearrange(x, transform_str, **mapping)
        else:
            raise ValueError('Must be between 3-5 dims')

        self.arr = rearrange(self.arr, '... a b c -> (...) a b c')
        
        # We use some very simple hueristics to guess what kind of image we have
        if channel_range is not None: # Optionally specify the type
            self.channel_range = channel_range
        elif is_dtype(arr, Integer):
            assert self.arr.max() > 0, "Integer array must be non-negative"
            if self.arr.max() > 1:
                self.channel_range = ChannelRange.UINT8
            else: # We assume an integer array with 0s and 1s is a BW image
                self.channel_range = ChannelRange.BOOL
        elif is_dtype(arr, Float):
            if -128 <= self.arr.min() <= self.arr.max() <= 128:
                self.channel_range = ChannelRange.FLOAT
            else:
                raise ValueError('Not supported')
        elif is_dtype(arr, Bool):
            self.channel_range = ChannelRange.BOOL
        else:
            raise ValueError('Invalid Type')

    def __repr__(self):
        if self.arr_type == ndarray:
            arr_name = 'ndarray'
        elif self.arr_type == Tensor:
            arr_name = 'tensor'
        else:
            raise ValueError('Must be numpy array, pillow image, or torch tensor')

        if is_pil(self.arr):
            shape_str = repr(self.arr)
        else:
            shape_str = f'type: {arr_name}, shape: {self.shape}'

        return f'Im of {shape_str}, device: {self.device}'

    def _convert(self, desired_datatype: ImArrType, desired_order: ChannelOrder = ChannelOrder.HWC, desired_range: ChannelRange = ChannelRange.UINT8) -> Im:
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
        if select_batch:
            im = im[select_batch]
        else:
            im = self.arr_transform(im)

        if desired_order == ChannelOrder.CHW and self.channel_order == ChannelOrder.HWC:
            im = rearrange(im, '... h w c -> ... c h w')
        elif desired_order == ChannelOrder.HWC and self.channel_order == ChannelOrder.CHW:
            im = rearrange(im, '... c h w -> ... h w c')

        start_cur_order = 'h w ()' if desired_order == ChannelOrder.HWC else '() h w'
        end_cur_order = start_cur_order.replace('()', 'c')

        if self.channel_range != desired_range:
            if is_ndarray(im):
                if self.channel_range == ChannelRange.FLOAT and desired_range == ChannelRange.UINT8:
                    im = (im * 255).astype(np.uint8)
                elif self.channel_range == ChannelRange.UINT8 and desired_range == ChannelRange.FLOAT:
                    im = (im / 255.0).astype(np.float32)
                elif self.channel_range == ChannelRange.BOOL and desired_range == ChannelRange.UINT8:
                    assert self.channels == 1
                    im = (repeat(im, f"... {start_cur_order} -> ... {end_cur_order}", c=3) * 255).astype(np.uint8)
                else:
                    raise ValueError("Not supported")
            elif is_tensor(im):
                if self.channel_range == ChannelRange.FLOAT and desired_range == ChannelRange.UINT8:
                    im = (im * 255).to(torch.uint8)
                elif self.channel_range == ChannelRange.UINT8 and desired_range == ChannelRange.FLOAT:
                    im = (im / 255.0).to(torch.float32)
                elif self.channel_range == ChannelRange.BOOL and desired_range == ChannelRange.UINT8:
                    assert self.channels == 1
                    im = (repeat(im, f"... {start_cur_order} -> ... {end_cur_order}", c=3) * 255).to(torch.uint8)
                elif self.channel_range == ChannelRange.BOOL and desired_range == ChannelRange.FLOAT:
                    assert self.channels == 1
                    im = repeat(im, f"... {start_cur_order} -> ... {end_cur_order}", c=3).to(torch.float32)
                else:
                    print(self.channel_range, desired_range)
                    raise ValueError("Not supported")

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
            img = rearrange(self.get_np(), '... h w c -> (...) h w c')
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
    def channels(self):
        """
        Returns number of channels in the image (e.g., 3 for RGB or 1 for BW)
        """
        if self.channel_order == ChannelOrder.HWC:
            return self.arr.shape[-1] 
        else:
            return self.arr.shape[-3]
    
    @property
    def image_shape(self): # returns h,w
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
        return Im(T.resize(self.arr, [height, width], resampling_mode, antialias=True))

    def scale(self, scale) -> Im:
        width, height = self.width, self.height
        return self.resize(int(height * scale), int(width * scale))

    def scale_to_width(self, new_width) -> Im:
        width, height = self.width, self.height
        wpercent = (new_width/float(width))
        hsize = int((float(height)*float(wpercent)))
        return self.resize(hsize, new_width)

    def scale_to_height(self, new_height) -> Im:
        width, height = self.width, self.height
        hpercent = (new_height/float(height))
        wsize = int((float(width)*float(hpercent)))
        return self.resize(new_height, wsize)

    @staticmethod
    def _save_data(filepath: Path = Path(get_date_time_str()), filetype='png'):
        filepath = Path(filepath)
        if filepath.suffix == '':
            filepath = filepath.with_suffix(f'.{filetype}')

        if len(filepath.parents) == 1:
            filepath = Path('output') / filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)

        return filepath

    def save(self, filepath: Path = Path(get_date_time_str()), filetype='png', optimize=False, quality=None):
        img = self.get_torch()

        filepath = Im._save_data(filepath, filetype)

        if len(img.shape) > 3:
            from torchvision import utils
            img = rearrange(img, '... h w c -> (...) h w c')
            img = utils.make_grid(img)
            img = Im(img).get_pil()
        else:
            img = self.get_pil()
        
        assert isinstance(img, Image.Image)

        flags = {'optimize': True, 'quality': quality if quality else 0.95} if optimize or quality else {}

        img.save(filepath, **flags)

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def write_text(self, text: str) -> Im:
        for i in range(self.arr.shape[0]):
            text_to_write = text[i] if isinstance(text, list) else text
            assert isinstance(self.arr[i], ndarray)
            im = cv2.cvtColor(cast(ndarray, self.arr[i]), cv2.COLOR_RGB2BGR)
            im = cv2.putText(im, text_to_write, (0, im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.002 *
                             min(self.arr.shape[-3:-1]), (255, 0, 0), max(1, round(min(self.arr.shape[-3:-1]) / 150)), cv2.LINE_AA)
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
        """E.g.,cv2.COLOR_RGB2BGR """
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

    def save_video(self, filepath: Path, fps: int, format='mp4'):
        filepath = Im._save_data(filepath, format)
        byte_stream = self.encode_video(fps, format)
        with open(filepath, "wb") as f:
            f.write(byte_stream.getvalue())

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def encode_video(self, fps: int, format='mp4') -> BytesIO:
        assert len(self.arr.shape) == 4, "Video data must be 4D (time, height, width, channels)"
        import imageio
        byte_stream = BytesIO()

        # TODO: We shouldn't need to write -> read. An imageio/ffmpeg issue is causing this.
        with tempfile.NamedTemporaryFile(suffix=f'.{format}') as ntp:
            if format == 'webm':
                writer = imageio.get_writer(ntp.name, format='webm', codec='libvpx-vp9', pixelformat='yuv420p', output_params=['-lossless', '1'], fps=fps)
            elif format == 'gif':
                writer = imageio.get_writer(ntp.name, format='GIF', mode="I", duration=(1000 * 1/fps))
            elif format == 'mp4':
                writer = imageio.get_writer(ntp.name, quality=10, pixelformat='yuv420p', codec='libx264', fps=fps)
            else:
                raise NotImplementedError(f'Format {format} not implemented.')

            for frame in self.arr:
                writer.append_data(frame)

            writer.close()
            with open(ntp.name, 'rb') as f:
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
        return Im(rearrange([img._handle_order_transform(img.arr, desired_order=ChannelOrder.HWC, desired_range=img.channel_range) for img in imgs], 'b ... -> b ...'))
    
    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.CHW, desired_range=ChannelRange.FLOAT)
    def colorize(self) -> Im:
        if self.channels not in colorize_weights:
            colorize_weights[self.channels] = torch.randn(3, self.channels, 1, 1)
        
        assert isinstance(self.arr, Tensor)
        self.arr = F.conv2d(self.arr, weight=colorize_weights[self.channels])
        self.arr = (self.arr-self.arr.min())/(self.arr.max()-self.arr.min())
        return self
    
    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.FLOAT)
    def pca(self, **kwargs) -> Im:
        """
        Computes principal components over all "pixels" in the batched image array.
        You may optionally specify principal components.
        """
        assert isinstance(self.arr, Tensor)
        b, h, w, _ = self.arr.shape
        pca_arr: Tensor = rearrange(self.arr, '... c -> (...) c')
        output = pca(pca_arr, **kwargs)
        output: Tensor = rearrange(output, '(b h w) c -> b h w c', b=b, h=h, w=w)
        return Im(output)

    pil = property(get_pil)
    np = property(get_np)
    torch = property(get_torch)
    opencv = property(get_opencv)


def torch_to_numpy(arr: Tensor):
    # Sadly, NumPy does not support these types
    if arr.dtype == torch.bfloat16 or arr.dtype == torch.float16:
        return arr.float().cpu().detach().numpy()
    else:
        return arr.cpu().detach().numpy()

def concat_variable(concat_func: Callable[..., Im], *args: Im, **kwargs) -> Im:
    output_img = None
    for img in args:
        if output_img is None:
            output_img = img
        else:
            output_img = concat_func(output_img, img, **kwargs)

    assert isinstance(output_img, Im)
    return output_img

def get_arr_hwc(im: Im): return im._handle_order_transform(im.arr, desired_order=ChannelOrder.HWC, desired_range=im.channel_range)


def concat_horizontal_(im1: Im, im2: Im) -> Im:
    if im1.height != im2.height:
        raise ValueError(f'Images must have same height. Got {im1.height} and {im2.height}')
    return Im(pack([get_arr_hwc(im1), get_arr_hwc(im2)], 'h * c')[0])

def concat_vertical_(im1: Im, im2: Im) -> Im:
    if im1.width != im2.width:
        raise ValueError(f'Images must have same width. Got {im1.width} and {im2.width}')
    return Im(pack([get_arr_hwc(im1), get_arr_hwc(im2)], '* w c')[0])

def get_layered_image_from_binary_mask(masks, flip=False):
    if torch.is_tensor(masks):
        masks = torch_to_numpy(masks)
    if flip:
        masks = np.flipud(masks)

    masks = masks.astype(np.bool_)

    colors = np.asarray(list(get_n_distinct_colors(masks.shape[2])))
    img = np.zeros((*masks.shape[:2], 3))
    for i in range(masks.shape[2]):
        img[masks[..., i]] = colors[i]

    return Image.fromarray(img.astype(np.uint8))


def get_img_from_binary_masks(masks, flip=False):
    """H W C"""
    arr = encode_binary_labels(masks)
    if flip:
        arr = np.flipud(arr)

    colors = np.asarray(list(get_n_distinct_colors(2 ** masks.shape[2])))
    return Image.fromarray(colors[arr].astype(np.uint8))


def encode_binary_labels(masks):
    if torch.is_tensor(masks):
        masks = torch_to_numpy(masks)

    masks = masks.transpose(2, 0, 1)
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)


def get_n_distinct_colors(n):
    def HSVToRGB(h, s, v):
        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        return (int(255 * r), int(255 * g), int(255 * b))

    huePartition = 1.0 / (n + 1)
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))


def square_pad(image, h, w):
    h_1, w_1 = image.shape[-2:]
    ratio_f = w / h
    ratio_1 = w_1 / h_1

    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_1, 2) != round(ratio_f, 2):

        # padding to preserve aspect ratio
        hp = int(w_1/ratio_f - h_1)
        wp = int(ratio_f * h_1 - w_1)
        if hp > 0 and wp < 0:
            hp = hp // 2
            image = T.pad(image, (0, hp, 0, hp), 0, "constant")
            return T.resize(image, [h, w], antialias=True)

        elif hp < 0 and wp > 0:
            wp = wp // 2
            image = T.pad(image, (wp, 0, wp, 0), 0, "constant")
            return T.resize(image, [h, w], antialias=True)

    else:
        return T.resize(image, [h, w], antialias=True)

def calculate_principal_components(embeddings, num_components=3):
    """Calculates the principal components given the embedding features.

    Args:
        embeddings: A 2-D float tensor of shape `[num_pixels, embedding_dims]`.
        num_components: An integer indicates the number of principal
        components to return.

    Returns:
        A 2-D float tensor of shape `[num_pixels, num_components]`.
    """
    embeddings = embeddings - torch.mean(embeddings, 0, keepdim=True)
    _, _, v = torch.svd(embeddings)
    return v[:, :num_components]


def pca(
        embeddings: Tensor, 
        num_components: int = 3, 
        principal_components: Optional[Tensor] = None
    ) -> Tensor:
    """Conducts principal component analysis on the embedding features.

    This function is used to reduce the dimensionality of the embedding.

    Args:
        embeddings: An N-D float tensor with shape with the 
        last dimension as `embedding_dim`.
        num_components: The number of principal components.
        principal_components: A 2-D float tensor used to convert the
        embedding features to PCA'ed space, also known as the U matrix
        from SVD. If not given, this function will calculate the
        principal_components given inputs.

    Returns:
        A N-D float tensor with the last dimension as  `num_components`.
    """
    shape = embeddings.shape
    embeddings = embeddings.view(-1, shape[-1])

    if principal_components is None:
        principal_components = calculate_principal_components(
            embeddings, num_components)
    embeddings = torch.mm(embeddings, principal_components)

    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.view(new_shape)

    return embeddings

def download_image(url) -> BytesIO:
    import urllib
    try:
        with urllib.request.urlopen(url) as response:
            return BytesIO(response.read())
    except urllib.error.URLError as e:
        raise Exception("Error downloading the image: " + str(e))

def load_cached_from_url(url: str) -> BytesIO:
    import hashlib
    cache_dir = Path.home() / '.cache' / 'image_utils'
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest()
    local_path = cache_dir / filename

    if local_path.exists():
        return BytesIO(local_path.read_bytes())
    else:
        image_bytesio = download_image(url)
        print(f'Downloading image from {url} and caching in {local_path}')
        with open(local_path, 'wb') as file:
            file.write(image_bytesio.getvalue())
        return image_bytesio