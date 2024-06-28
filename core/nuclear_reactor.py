import numpy as np
from numba import njit

@njit
def get_neighbors(array, x, y, z):
    neighbors = np.empty(6, dtype=array.dtype)
    directions = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)])
    
    for i, direction in enumerate(directions):
        new_x, new_y, new_z = x + direction[0], y + direction[1], z + direction[2]
        if (0 <= new_x < array.shape[0] and
            0 <= new_y < array.shape[1] and
            0 <= new_z < array.shape[2]):
            neighbors[i] = array[new_x, new_y, new_z]
        else:
            neighbors[i] = -1
    return neighbors

@njit
def is_valid(element, neighbors):
    neighbor_counts = np.bincount(neighbors[neighbors >= 0], minlength=18)
    
    if element == 0 or element == -1:
        return True
    elif element == 1:
        return True
    elif element == 2:
        return neighbor_counts[1] >= 1
    elif element == 3:
        return neighbor_counts[1] > 0 or neighbor_counts[2] > 0
    elif element == 4:
        return neighbor_counts[1] > 0
    elif element == 5:
        return neighbor_counts[2] > 0
    elif element == 6:
        return neighbor_counts[3] > 0 and neighbor_counts[4] > 0
    elif element == 7:
        return neighbor_counts[2] >= 2
    elif element == 8:
        return neighbor_counts[1] > 0 and neighbor_counts[-1] > 0
    elif element == 9:
        return neighbor_counts[3] > 0 and neighbor_counts[5] > 0
    elif element == 10:
        return neighbor_counts[4] == 1 and neighbor_counts[-1] > 0
    elif element == 11:
        return neighbor_counts[-1] >= 3
    elif element == 12:
        return neighbor_counts[1] >= 2
    elif element == 13:
        return neighbor_counts[6] > 0
    elif element == 14:
        return neighbor_counts[2] > 0 and neighbor_counts[-1] > 0
    elif element == 15:
        return neighbor_counts[7] > 0
    elif element == 16:
        return (neighbor_counts[8] >= 2 and 
                (neighbors[0] == neighbors[1] == 8 or 
                 neighbors[2] == neighbors[3] == 8 or 
                 neighbors[4] == neighbors[5] == 8))
    elif element == 17:
        return neighbor_counts[-1] > 0 and neighbor_counts[2] > 0
    else:
        return False

@njit
def validate_array(array):
    queue = [(x, y, z) for x in range(array.shape[0]) for y in range(array.shape[1]) for z in range(array.shape[2])]
    changes = True
    
    while changes:
        changes = False
        next_queue = []
        
        for x, y, z in queue:
            element = array[x, y, z]
            neighbors = get_neighbors(array, x, y, z)
            
            if not is_valid(element, neighbors):
                array[x, y, z] = 0  # Mark as invalid
                changes = True
                next_queue.extend([(x, y, z)])
        
        queue = next_queue
    
    return array



@njit
def is_array_valid(array):
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            for z in range(array.shape[2]):
                element = array[x, y, z]
                neighbors = get_neighbors(array, x, y, z)
                if not is_valid(element, neighbors):
                    return False
    return True


