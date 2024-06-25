import numpy as np
from numba import njit

@njit
def get_neighbors(array, x, y, z):
    neighbors = np.empty(6, dtype=array.dtype)
    # Define the six possible moves (up, down, left, right, front, back)
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
    neighbor_counts = np.zeros(18, dtype=np.int32)
    for neighbor in neighbors:
        neighbor_counts[neighbor] += 1
    
    if element == 0:  # Air, any
        return True
    elif element == 1:  # Reactor Cell, any
        return True
    elif element == 2:  # Moderator, must be active
        return neighbor_counts[1] >= 1
    elif element == 3:  # Water, at least one Reactor Cell or active Moderator
        return neighbor_counts[1] > 0 or neighbor_counts[2] > 0
    elif element == 4:  # Redstone ...
        return neighbor_counts[1] > 0
    elif element == 5:  # Quartz
        return neighbor_counts[2] > 0 
    elif element == 6:  # Gold
        return neighbor_counts[3] > 0 and neighbor_counts[4] > 0
    elif element == 7:  # Glowstone
        return neighbor_counts[2] >= 2
    elif element == 8:  # Lapis
        return neighbor_counts[1] > 0 and neighbor_counts[-1] > 0
    elif element == 9:  # Diamond
        return neighbor_counts[3] > 0 and neighbor_counts[5] > 0
    elif element == 10:  # Helium
        return neighbor_counts[4] == 1 and neighbor_counts[-1] > 0
    elif element == 11:  # Enderium
        return neighbor_counts[-1] >= 3
    elif element == 12:  # Cryotheum
        return neighbor_counts[1] >= 2
    elif element == 13:  # Iron
        return neighbor_counts[6] > 0
    elif element == 14:  # Emerald
        return neighbor_counts[2] > 0 and neighbor_counts[-1] > 0
    elif element == 15:  # Copper
        return neighbor_counts[7] > 0
    elif element == 16:  # Tin for two lapis on opposite sides in the same axis
        return (neighbor_counts[8] >= 2 and 
                (neighbors[0] == neighbors[1] == 8 or 
                 neighbors[2] == neighbors[3] == 8 or 
                 neighbors[4] == neighbors[5] == 8))
    elif element == 17:  # Placeholder
        return neighbor_counts[-1] > 0 and neighbor_counts[2] > 0
    else:
        return True  # Reactor Casings for later

@njit
def validate_array(array):
    changes = True
    while changes:
        changes = False
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                for z in range(array.shape[2]):
                    element = array[x, y, z]
                    neighbors = get_neighbors(array, x, y, z)
                    if not is_valid(element, neighbors):
                        array[x, y, z] = 0  # Mark as invalid
                        changes = True
    return array