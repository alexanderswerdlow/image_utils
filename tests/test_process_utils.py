from image_utils.process_utils import parallelize
from image_utils.file_utils import get_rand_hex
import pytest
from pathlib import Path
from

img_path = Path('tests/flower.jpg')
save_path = Path(__file__).parent / 'output'


# memory = Memory(location='./cachedir', verbose=0, mmap_mode='r')
# @memory.cache
# def costly_compute(data, column):
#     """Emulate a costly function by sleeping and returning a column."""
#     time.sleep(1)
#     return data[column] + 1

def test_func():
    pass


def get_test_list(list_size):
    return [get_rand_hex() for _ in range(list_size)]


@pytest.mark.parametrize("num_processes", [1, 2, 10])
@pytest.mark.parametrize("list_size", [1, 10, 10000])
def test_parallelize(num_processes, list_size):
    test_list = get_test_list(list_size)
    test_parallel = parallelize(test_func, num_processes=num_processes, shuffle=False)(test_list)
