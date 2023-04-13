from __future__ import annotations
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageOps
import colorsys
import cv2
import torch.nn.functional as F
from einops import rearrange, repeat, pack
import torchvision.transforms.functional as T
import tempfile
from io import BytesIO
import copy
from pathlib import Path
from strenum import StrEnum
from enum import auto
import string


def is_tensor(obj):
    return torch.is_tensor(obj)


def is_ndarray(obj):
    return isinstance(obj, np.ndarray)


def is_pil(obj):
    return isinstance(obj, Image.Image)


def is_arr(obj):
    return torch.is_tensor(obj) | isinstance(obj, np.ndarray)


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

    Note: Be careful when using this class directly as part of a training pipeline. Many operations will cause the underlying data to convert between formats (e.g., Tensor -> Pillow) and move the data back to system memory and/or incur loss of precision (e.g., float -> uint8)
    """

    default_normalize_mean = [0.4265, 0.4489, 0.4769]
    default_normalize_std = [0.2053, 0.2206, 0.2578]

    def __init__(self, arr, channel_range: ChannelRange = None, **kwargs):
        self.arr_device = None
        self.arr = arr
        self.arr_type: Union[Image.Image, np.ndarray, torch.Tensor]
        if isinstance(arr, np.ndarray):
            self.arr_type = np.ndarray
        elif isinstance(arr, Image.Image):
            self.arr_type = Image.Image
        elif isinstance(arr, torch.Tensor):
            self.arr_device = arr.device
            self.arr_type = torch.Tensor
        else:
            raise ValueError('Must be numpy array, pillow image, or torch tensor')

        if is_arr(self.arr):
            if len(self.arr.shape) == 2:
                self.arr = self.arr[..., None]

            self.channel_order = ChannelOrder.HWC if self.arr.shape[-1] < min(self.arr.shape[-3:-1]) else ChannelOrder.CHW
            self.dtype = self.arr.dtype
            self.shape = self.arr.shape

            if self.channel_order != ChannelOrder.HWC:
                self.arr = rearrange(self.arr, '... c h w -> ... h w c')

            if len(self.shape) == 3:
                self.arr_transform = lambda x: rearrange(x, '() h w c -> h w c')
            elif len(self.shape) == 4:
                self.arr_transform = lambda x: x
            elif len(self.shape) >= 5:
                extra_dims = self.shape[:-3]
                mapping = {k: v for k, v in zip(string.ascii_uppercase, extra_dims)}
                self.arr_transform = lambda x: rearrange(x, f'({" ".join(sorted(list(mapping.keys())))}) h w c -> {" ".join(sorted(list(mapping.keys())))} h w c', **mapping)
            else:
                raise ValueError('Must be between 3-5 dims')

            self.arr = rearrange(self.arr, '... h w c -> (...) h w c')
        else:
            self.channel_order = ChannelOrder.HWC
            self.shape = (*self.arr.size[::-1], len(self.arr.getbands()))
            self.dtype = np.array(self.arr).dtype
            self.arr_transform = lambda x: rearrange(x, 'h w c -> h w c')

        if channel_range is not None:
            self.channel_range = channel_range
        elif self.dtype == np.uint8 or self.dtype == torch.uint8:
            if self.arr_type == Image.Image or self.arr.max() > 1:
                self.channel_range = ChannelRange.UINT8
            else:
                self.channel_range = ChannelRange.BOOL
        elif self.dtype == np.float16 or self.dtype == np.float32 or self.dtype == torch.float16 or self.dtype == torch.bfloat16 or self.dtype == torch.float32:
            if -10 <= self.arr.min() <= self.arr.max() <= 10:
                self.channel_range = ChannelRange.FLOAT
            else:
                raise ValueError('Not supported')
        elif self.dtype == np.bool_ or torch.bool:
            self.channel_range = ChannelRange.BOOL
        else:
            raise ValueError('Invalid Type')

    def __repr__(self):
        if self.arr_type == np.ndarray:
            arr_name = 'ndarray'
        elif self.arr_type == torch.Tensor:
            arr_name = 'tensor'
        elif self.arr_type == Image.Image:
            arr_name = 'PIL Image'
        else:
            raise ValueError('Must be numpy array, pillow image, or torch tensor')

        if is_pil(self.arr):
            shape_str = repr(self.arr)
        else:
            shape_str = f'type: {arr_name}, shape: {self.shape}'
        return f'Im of {shape_str}, device: {self.arr_device}'

    @property
    def height(self):
        return self.arr.shape[-3]
    
    @property
    def width(self):
        return self.arr.shape[-2]

    def convert(self, desired_datatype: Union[Image.Image, np.ndarray, torch.Tensor], desired_order: ChannelOrder = ChannelOrder.HWC, desired_range: ChannelRange = ChannelRange.UINT8) -> Im:
        if self.arr_type != desired_datatype or self.channel_order != desired_order or self.channel_range != desired_range:
            if desired_datatype == np.ndarray:
                self = Im(self.get_np(order=desired_order, range=desired_range))
            elif desired_datatype == Image.Image:
                self = Im(self.get_pil())
            elif desired_datatype == torch.Tensor:
                self = Im(self.get_torch(order=desired_order, range=desired_range))

        return self

    def convert_to_datatype(desired_datatype: Union[Image.Image, np.ndarray, torch.Tensor], desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8):
        def custom_decorator(func):
            def wrapper(self, *args, **kwargs):
                if desired_datatype == Image.Image and is_arr(self.arr) and self.arr.shape[0] > 1:
                    return Im(np.stack([func(Im(img), *args, **kwargs).get_np() for img in self.get_pil()]))
                else:
                    self = self.convert(desired_datatype, desired_order, desired_range)
                    return func(self, *args, **kwargs)
            return wrapper
        return custom_decorator

    def handle_order_transform(self, im, desired_order: ChannelOrder, desired_range: ChannelRange, select_batch=None):
        if select_batch:
            im = im[select_batch]
        else:
            im = self.arr_transform(im)

        if desired_order == ChannelOrder.CHW:
            im = rearrange(im, '... h w c -> ... c h w')

        if self.channel_range != desired_range:
            if is_ndarray(im):
                if self.channel_range == ChannelRange.FLOAT and desired_range == ChannelRange.UINT8:
                    im = (im * 255).astype(np.uint8)
                elif self.channel_range == ChannelRange.UINT8 and desired_range == ChannelRange.FLOAT:
                    im = (im / 255.0).astype(np.float32)
                elif self.channel_range == ChannelRange.BOOL and desired_range == ChannelRange.UINT8:
                    im = (repeat(im[..., 0], "... -> ... c", c=3) * 255).astype(np.uint8)
                else:
                    raise ValueError("Not supported")
            elif is_tensor(im):
                if self.channel_range == ChannelRange.FLOAT and desired_range == ChannelRange.UINT8:
                    im = (im * 255).to(torch.uint8)
                elif self.channel_range == ChannelRange.UINT8 and desired_range == ChannelRange.FLOAT:
                    im = (im / 255.0).to(torch.float32)
                elif self.channel_range == ChannelRange.BOOL and desired_range == ChannelRange.UINT8:
                    im = (repeat(im[..., 0], "... -> ... c", c=3) * 255).to(torch.uint8)
                elif self.channel_range == ChannelRange.BOOL and desired_range == ChannelRange.FLOAT:
                    im = repeat(im[..., 0], "... -> ... c", c=3).to(torch.float32)
                else:
                    print(self.channel_range, desired_range)
                    raise ValueError("Not supported")

        return im

    def get_np(self, order=ChannelOrder.HWC, range=ChannelRange.UINT8) -> np.ndarray:
        if is_pil(self.arr):
            arr = pil_to_numpy(self.arr)
        elif is_tensor(self.arr):
            arr = torch_to_numpy(self.arr)
        else:
            arr = self.arr

        arr = self.handle_order_transform(arr, order, range)

        return arr

    def get_torch(self, order=ChannelOrder.CHW, range=ChannelRange.FLOAT) -> torch.Tensor:
        if is_tensor(self.arr):
            arr = self.arr
        elif is_pil(self.arr):
            arr = torch.from_numpy(pil_to_numpy(self.arr))
        else:
            arr = torch.from_numpy(self.arr)

        arr = self.handle_order_transform(arr, order, range)
        if self.arr_device is not None:
            arr = arr.to(self.arr_device)
        return arr

    def get_pil(self) -> Image.Image:
        if is_pil(self.arr):
            return self.arr
        elif len(self.shape) == 3:
            return Image.fromarray(self.get_np())
        else:
            img = self.convert(np.ndarray, ChannelOrder.HWC, ChannelRange.UINT8).get_np()
            img = rearrange(img, '... h w c -> (...) h w c')
            if img.shape[0] == 1:
                return Image.fromarray(img[0])
            else:
                return [Image.fromarray(img[i]) for i in range(img.shape[0])]

    @property
    def copy(self):
        return copy.deepcopy(self)

    @property
    def image_shape(self):
        return (self.shape[-3], self.shape[-2])

    def open(filepath: Path, use_imageio=False) -> Im:
        if use_imageio:
            import imageio.v3 as iio
            img = iio.imread(filepath)
        else:
            img = Image.open(filepath)
        return Im(img)

    # Taken from: https://github.com/GaParmar/clean-fid/blob/9c9dded6758fc4b575c27c4958dbc87b9065ec6e/cleanfid/resize.py#L41
    @convert_to_datatype(desired_datatype=Image.Image)
    def resize(self, height, width, resampling_mode=Image.Resampling.LANCZOS):
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize((width, height), resample=resampling_mode)
            return np.asarray(img).clip(0, 255).reshape(height, width, 1)

        self.arr = [resize_single_channel(np.array(self.arr)[:, :, idx]) for idx in range(3)]
        self.arr = np.concatenate(self.arr, axis=2).astype(np.float32) / 255.0
        return Im(self.arr)

    @convert_to_datatype(desired_datatype=Image.Image)
    def scale(self, scale) -> Im:
        width, height = self.arr.size
        return self.resize(int(height * scale), int(width * scale))

    @convert_to_datatype(desired_datatype=Image.Image)
    def scale_to_width(self, new_width) -> Im:
        width, height = self.arr.size
        wpercent = (new_width/float(width))
        hsize = int((float(height)*float(wpercent)))
        return self.resize(hsize, new_width)

    @convert_to_datatype(desired_datatype=Image.Image)
    def scale_to_height(self, new_height) -> Im:
        width, height = self.arr.size
        hpercent = (new_height/float(height))
        wsize = int((float(width)*float(hpercent)))
        return self.resize(new_height, wsize)
    
    @staticmethod
    def _save_data(filepath: Path = None, filetype='png'):
        if filepath is None:
            from image_utils.file_utils import get_date_time_str
            filepath = get_date_time_str()

        filepath = Path(filepath)
        if filepath.suffix == '':
            filepath = filepath.with_suffix(f'.{filetype}')

        if len(filepath.parents) == 1:
            filepath = Path('output') / filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)

        return filepath

    def save(self, filepath: Path = None, filetype='png', optimize=False, quality=None):
        img = self.get_torch()

        filepath = Im._save_data(filepath, filetype)

        if len(img.shape) > 3:
            from torchvision import utils
            img = rearrange(img, '... h w c -> (...) h w c')
            img = utils.make_grid(img)
            img = Im(img).get_pil()
        else:
            img = self.get_pil()

        flags = {'optimize': True, 'quality': quality if quality else 0.95} if optimize or quality else {}
        img.save(filepath, **flags)

    @convert_to_datatype(desired_datatype=np.ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def write_text(self, text: str) -> Im:
        for i in range(self.arr.shape[0]):
            text_to_write = text[i] if isinstance(text, list) else text
            im = cv2.cvtColor(self.arr[i], cv2.COLOR_RGB2BGR)
            im = cv2.putText(im, text_to_write, (0, im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.002 *
                             min(self.arr.shape[-3:-1]), (255, 0, 0), max(1, round(min(self.arr.shape[-3:-1]) / 150)), cv2.LINE_AA)
            self.arr[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return self

    def add_border(self, border: int, color: Tuple[int, int, int]):
        imgs = self.pil
        if isinstance(imgs, Iterable):
            imgs = Im(np.stack([Im(ImageOps.expand(ImageOps.crop(img, border=border), border=border, fill=color)).np for img in imgs], axis=0))
        else:
            imgs = Im(ImageOps.expand(ImageOps.crop(imgs, border=border), border=border, fill=color))
        return imgs

    def normalize_setup(self, mean=default_normalize_mean, std=default_normalize_std):
        if self.arr_type == np.ndarray or self.arr_type == Image.Image:
            self = Im(self.get_np(ChannelOrder.HWC, ChannelRange.FLOAT))
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(self.dtype)
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(self.dtype)
        elif self.arr_type == torch.Tensor:
            self = Im(self.get_torch(ChannelOrder.HWC, ChannelRange.FLOAT))
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std)

            mean, std = mean.to(device=self.arr_device), std.to(device=self.arr_device)
            
        return self, mean, std

    def normalize(self, **kwargs) -> Im:
        self, mean, std = self.normalize_setup(**kwargs)
        self.arr = (self.arr - mean) / std
        return self

    def denormalize(self, clamp: Union[bool, tuple(float, float)] = (0, 1.0), **kwargs) -> Im:
        self, mean, std = self.normalize_setup(**kwargs)
        self.arr = (self.arr * std) + mean
        if self.arr_type == np.ndarray:
            self.arr = self.arr.clip(*clamp) if clamp else self.arr
        elif self.arr_type == torch.Tensor:
            self.arr = self.arr.clamp(*clamp) if clamp else self.arr
        return self

    @convert_to_datatype(desired_datatype=np.ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def get_opencv(self):
        return self.arr

    @convert_to_datatype(desired_datatype=np.ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def convert_opencv_color(self, color: int):
        """E.g.,cv2.COLOR_RGB2BGR """
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

    @convert_to_datatype(desired_datatype=np.ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def encode_video(self, fps: int, format='mp4') -> BytesIO:
        assert len(self.arr.shape) == 4, "Video data must be 4D (time, height, width, channels)"
        import imageio
        byte_stream = BytesIO()

        if format == 'webm':
            writer = imageio.get_writer(byte_stream, format='webm', codec='libvpx-vp9', pixelformat='yuv420p', output_params=['-lossless', '1'], fps=fps)
        elif format == 'gif':
            writer = imageio.get_writer(byte_stream, format='GIF', mode="I", fps=fps)
        elif format == 'mp4':
            with tempfile.NamedTemporaryFile(suffix='.mp4') as ntp:
                writer = imageio.get_writer(ntp.name, quality=10, pixelformat='yuv420p', codec ='libx264', fps=fps)
                for frame in self.arr:
                    writer.append_data(frame)
                writer.close()
                with open(ntp.name, 'rb') as f:
                    byte_stream.write(f.read())
        else:
            raise NotImplementedError(f'Format {format} not implemented.')

        if format != 'mp4':
            for frame in self.arr:
                writer.append_data(frame)

            writer.close()
        
        byte_stream.seek(0)
        return byte_stream

    def to(self, device):
        assert self.arr_type == torch.Tensor, "Can only convert to device if array is a torch.Tensor"
        self.arr = self.arr.to(device)
        return self

    @staticmethod
    def stack_imgs(*args):
        imgs = [img.convert(desired_datatype=np.ndarray) if img.arr_type == Image.Image else img for img in args]
        return Im(rearrange([img.handle_order_transform(img.arr, desired_order=ChannelOrder.HWC, desired_range=img.channel_range) for img in imgs], 'b ... -> b ...'))

    pil = property(get_pil)
    np = property(get_np)
    torch = property(get_torch)
    opencv = property(get_opencv)


def torch_to_numpy(arr):
    if arr.dtype == torch.bfloat16 or arr.dtype == torch.float16:
        return arr.float().cpu().detach().numpy()
    else:
        return arr.cpu().detach().numpy()


def pil_to_numpy(arr):
    return np.array(arr.convert('RGB'))


def concat_variable(concat_func, *args, **kwargs) -> Im:
    output_img = None
    for img in args:
        if output_img is None:
            output_img = img
        else:
            output_img = concat_func(output_img, img, **kwargs)
            
    return output_img

def get_arr_hwc(im: Im): return im.handle_order_transform(im.arr, desired_order=ChannelOrder.HWC, desired_range=im.channel_range)

def concat_horizontal_(im1: Im, im2: Im, spacing=0) -> Im:
    if is_arr(im1.arr) and is_arr(im2.arr) and im1.height == im2.height:
        if is_tensor(im1.arr):
            return Im(torch.cat([get_arr_hwc(im1), get_arr_hwc(im2)], dim=-2))
        elif is_ndarray(im1.arr):
            return Im(np.concatenate([get_arr_hwc(im1), get_arr_hwc(im2)], axis=-2))
    else:
        im1, im2 = im1.get_pil(), im2.get_pil()
        dst = Image.new("RGBA", (im1.width + spacing + im2.width, max(im2.height, im1.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (spacing + im1.width, 0))
        return Im(dst)


def concat_vertical_(im1: Im, im2: Im, spacing=0) -> Im:
    if is_arr(im1.arr) and is_arr(im2.arr) and im1.width == im2.width:
        if is_tensor(im1.arr):
            return Im(torch.cat([get_arr_hwc(im1), get_arr_hwc(im2)], dim=-3))
        elif is_ndarray(im1.arr):
            return Im(np.concatenate([get_arr_hwc(im1), get_arr_hwc(im2)], axis=-3))
    else:
        im1, im2 = im1.get_pil(), im2.get_pil()
        dst = Image.new('RGBA', (im1.width, im1.height + spacing + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, spacing + im1.height))
        return Im(dst)


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


colorize_weights = {}


def colorize(x):
    if x.shape[0] not in colorize_weights:
        colorize_weights[x.shape[0]] = torch.randn(3, x.shape[0], 1, 1)

    x = F.conv2d(x, weight=colorize_weights[x.shape[0]])
    x = (x-x.min())/(x.max()-x.min())
    return x


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
            return T.resize(image, [h, w])

        elif hp < 0 and wp > 0:
            wp = wp // 2
            image = T.pad(image, (wp, 0, wp, 0), 0, "constant")
            return T.resize(image, [h, w])

    else:
        return T.resize(image, [h, w])
