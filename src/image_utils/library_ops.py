try:
    import torch
    import numpy as np

    torch.set_printoptions(sci_mode=False, precision=3, threshold=10, edgeitems=2, linewidth=120)
    np.set_printoptions(suppress=True, precision=3, threshold=10, edgeitems=2, linewidth=120)
    import lovely_tensors as lt
    from lovely_numpy import lovely, set_config

    lt.monkey_patch()
    set_config(repr=lovely)
except ImportError:
    print("lovely_tensors is not installed. Run `pip install lovely-tensors` if you wish to use it.")
