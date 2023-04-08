from functools import partial
from multiprocessing import Pool
import random
import time
from pathlib import Path
from joblib import dump, load
import time
import numpy as np
from joblib import Parallel, delayed
from joblib import Memory


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def parallelize(func=None, num_processes=1, shuffle=False, custom_chunking=False, use_joblib=False):
    def wrapper(iterable, **kwargs):
        if use_joblib:
            executor = Parallel(n_jobs=num_processes, max_nbytes=1e6)
            tasks = (delayed(func)(iterable, idx, **kwargs) for idx in range(len(iterable)))
            return executor(tasks)
        else:
            with Pool(num_processes) as p:
                num_arr = list(range(len(iterable)))
                if shuffle:
                    random.shuffle(num_arr)
                for _ in p.imap_unordered(
                    partial(func, **kwargs),
                    split_range(num_arr, num_processes) if custom_chunking else iterable
                ):
                    pass

    return wrapper
