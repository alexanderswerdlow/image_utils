import torch
import numpy as np
import random
from . import is_tensor, is_ndarray, is_arr


def generic_print(self, arr_values):
    assert is_arr(self)

    if len(self.shape) == 0: return arr_values
    
    if is_ndarray(self):
        lib = np
        num_elements = lib.prod(self.shape)
        device = ''
    else:
        lib = torch
        num_elements = lib.prod(torch.tensor(list(self.shape))).item()
        device = self.device.type

    if self.dtype in (np.bool_, torch.bool):
        specific_data = f' sum: {self.sum()}, unique: {len(lib.unique(self))},'
    elif (is_ndarray(self) and np.issubdtype(self.dtype, np.integer)) or (is_tensor(self) and not torch.is_floating_point(self)):
        specific_data = f' unique: {len(lib.unique(self))},'
    else:
        specific_data = f' avg: {self.mean():.3f},'

    shape_str = ",".join([str(self.shape[i]) for i in range(len(self.shape))])
    finite_str = "finite" if lib.isfinite(self).all() else "non-finite"
    basic_info = f'[{shape_str}] {self.dtype} {device} {finite_str}'
    numerical_info = f'\nelems: {num_elements},{specific_data} min: {self.min():.3f}, max: {self.max().item():.3f}'
    return basic_info + numerical_info + f'\n{arr_values}\n' + basic_info


normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: generic_print(self, normal_repr(self))
torch.set_printoptions(sci_mode=False, precision=3)

np.set_string_function(lambda self: generic_print(self, np.ndarray.__repr__(self)), repr=False)
np.set_printoptions(suppress=True, precision=3)


def set_random_seeds():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
