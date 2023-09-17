try:
    import lovely_tensors as lt
    from lovely_numpy import lovely, set_config
    lt.monkey_patch()
    set_config(repr=lovely)
except ImportError:
    print("lovely_tensors is not installed. Run `pip install lovely-tensors` if you wish to use it.")
