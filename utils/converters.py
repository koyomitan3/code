import numpy as np
from utils.constants import REACTOR_BLOCK_ID


def convert_array(array):
    max_length = max(len(value) for value in REACTOR_BLOCK_ID.values())
    converted_array = np.empty(array.shape, dtype='U{}'.format(max_length))
    for index, value in np.ndenumerate(array):
        converted_array[index] = REACTOR_BLOCK_ID.get(value, str(value)).ljust(max_length)
    return converted_array


def pad_array(array, pad_value=-1):
    """Pad the 3D array with the specified pad_value on all sides."""
    padded_shape = (array.shape[0] + 2, array.shape[1] + 2, array.shape[2] + 2)
    padded_array = np.full(padded_shape, pad_value, dtype=int)
    padded_array[1:-1, 1:-1, 1:-1] = array
    return padded_array