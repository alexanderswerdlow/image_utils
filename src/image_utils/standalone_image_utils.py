from __future__ import annotations

import colorsys
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as T
from PIL import Image
from torch import Tensor


def torch_to_numpy(arr: Tensor):
    # Sadly, NumPy does not support these types
    if arr.dtype == torch.bfloat16 or arr.dtype == torch.float16:
        return arr.float().cpu().detach().numpy()
    else:
        return arr.cpu().detach().numpy()


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
        hp = int(w_1 / ratio_f - h_1)
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


def pca(embeddings: Tensor, num_components: int = 3, principal_components: Optional[Tensor] = None) -> Tensor:
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
        principal_components = calculate_principal_components(embeddings, num_components)
    embeddings = torch.mm(embeddings, principal_components)

    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.view(new_shape)

    return embeddings
