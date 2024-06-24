import numpy as np
from numba import njit
from plot_utils import plot_grid
from converters import convert_array, pad_array

@njit
def get_neighbors(array, x, y, z) -> list:
    return [
        array[x-1, y, z], array[x+1, y, z], array[x, y-1, z],
        array[x, y+1, z], array[x, y, z-1], array[x, y, z+1]
    ]

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
        return neighbor_counts[2] > 0
    elif element == 5:  # Quartz
        return neighbor_counts[3] > 0 and neighbor_counts[4] > 0
    elif element == 6:  # Gold
        return neighbor_counts[1] >= 2
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
    elif element == 14:  # Placeholder Lapis
        return neighbor_counts[1] > 0 and neighbor_counts[-1] > 0
    elif element == 15:  # Placeholder Lapis
        return neighbor_counts[1] > 0 and neighbor_counts[-1] > 0
    elif element == 16:  # Tin for two lapis on opposite sides in the same axis
        return (neighbor_counts[8] >= 2 and 
                (neighbors[0] == neighbors[1] == 8 or 
                 neighbors[2] == neighbors[3] == 8 or 
                 neighbors[4] == neighbors[5] == 8))
    elif element == 17:  # Placeholder
        return neighbor_counts[-1] >= 3
    elif element == 102:  # Placeholder
        return neighbor_counts[8] >= 2 and neighbor_counts[3] >= 1
    elif element == 103:
        return neighbor_counts[3] >= 1 or neighbor_counts[5] >= 1
    else:
        return True  # Reactor Casings for later

@njit
def validate_array(array) -> None:
    changes = True
    while changes:
        changes = False
        for x in range(1, array.shape[0] - 1):
            for y in range(1, array.shape[1] - 1):
                for z in range(1, array.shape[2] - 1):
                    element = array[x, y, z]
                    neighbors = get_neighbors(array, x, y, z)
                    if not is_valid(element, neighbors):
                        array[x, y, z] = 0
                        changes = True

@njit
def score_array(array):
    """Dummy function to score the array."""
    # Placeholder for actual scoring logic
    return np.sum(array)

def main():
    array = np.random.randint(0, 18, size=(3, 5, 5))
    print("Before:")
    print(array)
    padded_array = pad_array(array)
    validate_array(padded_array)
    print("Validated Array:")
    print(padded_array)
    score = score_array(padded_array)
    print("Array Score:", score)
    plot_grid(padded_array, save_path='plot2.png')

if __name__ == "__main__":
    main()