import numpy as np
from numba import njit
from plot_utils import plot_grid
from constants import REACTOR_BLOCK_ID

def convert_array(array):
    """Convert the array elements to their corresponding string representations with aligned width."""
    # Determine the maximum length of any string in REACTOR_BLOCK_ID
    max_length = max(len(value) for value in REACTOR_BLOCK_ID.values())
    
    # Create a new array of strings with aligned width
    converted_array = np.empty(array.shape, dtype='U{}'.format(max_length))
    
    # Populate the converted_array with aligned strings
    for index, value in np.ndenumerate(array):
        converted_array[index] = REACTOR_BLOCK_ID.get(value, str(value)).ljust(max_length)
    
    return converted_array


def pad_array(array, pad_value=-1):
    """Pad the 3D array with the specified pad_value on all sides."""
    padded_shape = (array.shape[0] + 2, array.shape[1] + 2, array.shape[2] + 2)
    padded_array = np.full(padded_shape, pad_value, dtype=int)
    padded_array[1:-1, 1:-1, 1:-1] = array
    return padded_array

@njit
def get_neighbors(array, x, y, z):
    return [
        array[x-1, y, z], array[x+1, y, z], array[x, y-1, z],
        array[x, y+1, z], array[x, y, z-1], array[x, y, z+1]
    ]

@njit
def is_valid(element, neighbors):
    if element == 0:
        return True
    elif element == 1:
        return True
    elif element == 2:
        return neighbors.count(2) + neighbors.count(3) > 0
    elif element == 2:
        return neighbors.count(0) >= 1 or neighbors.count(5) >= 1
    elif element == 100:
        return neighbors.count(6) >= 2 and (
            neighbors[0] == neighbors[1] == 6 or 
            neighbors[2] == neighbors[3] == 6 or 
            neighbors[4] == neighbors[5] == 6
        )
    elif element == 101:
        return neighbors.count(-1) >= 3
    elif element == 102:
        return neighbors.count(8) >= 2 and neighbors.count(3) >= 1
    elif element == 103:
        return neighbors.count(3) >= 1 or neighbors.count(5) >= 1
    else:
        return True  # For elements 6 to 17, assume they are always valid

@njit
def validate_array(array):
    """Validate the entire array and turn invalid elements into 0."""
    changes = True
    while changes:
        changes = False
        invalid_elements = []
        for x in range(1, array.shape[0] - 1):
            for y in range(1, array.shape[1] - 1):
                for z in range(1, array.shape[2] - 1):
                    element = array[x, y, z]
                    neighbors = get_neighbors(array, x, y, z)
                    if not is_valid(element, neighbors):
                        invalid_elements.append((x, y, z))
        
        if invalid_elements:
            changes = True
            for x, y, z in invalid_elements:
                array[x, y, z] = 0

@njit
def score_array(array):
    """Dummy function to score the array."""
    # Placeholder for actual scoring logic
    return np.sum(array)

def main():
    # Example 3x3x3 array
    array = np.random.randint(0, 18, size=(2, 5, 5))
    #padded_array = pad_array(array)
    validate_array(array)
    print("Validated Array:")
    print(convert_array(array))
    
    score = score_array(array)
    print("Array Score:", score)
    plot_grid(array, save_path='plot2.png')
    
if __name__ == "__main__":
    main()