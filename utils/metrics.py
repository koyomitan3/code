from numba import njit
import numpy as np
from .constants import REACTOR_BLOCK_ID, REACTOR_BLOCK_COLOR, REACTOR_FUEL_TYPE, REACTOR_BLOCK_PROPERTIES

@njit
def count_neighbors(array, x, y, z, element):
    count = 0
    max_x, max_y, max_z = array.shape

    directions = np.array([
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1)
    ])

    for direction in directions:
        nx, ny, nz = x + direction[0], y + direction[1], z + direction[2]
        if 0 <= nx < max_x and 0 <= ny < max_y and 0 <= nz < max_z:
            if array[nx, ny, nz] == element:
                count += 1

    return count

@njit
def count_neighbors_opposite(array, x, y, z, element):
    count = 0
    max_x, max_y, max_z = array.shape

    directions = np.array([
        ((-1, 0, 0), (1, 0, 0)),
        ((0, -1, 0), (0, 1, 0)),
        ((0, 0, -1), (0, 0, 1))
    ])

    for dir_pair in directions:
        nx1, ny1, nz1 = x + dir_pair[0][0], y + dir_pair[0][1], z + dir_pair[0][2]
        nx2, ny2, nz2 = x + dir_pair[1][0], y + dir_pair[1][1], z + dir_pair[1][2]
        if 0 <= nx1 < max_x and 0 <= ny1 < max_y and 0 <= nz1 < max_z:
            if array[nx1, ny1, nz1] == element:
                count += 1
        if 0 <= nx2 < max_x and 0 <= ny2 < max_y and 0 <= nz2 < max_z:
            if array[nx2, ny2, nz2] == element:
                count += 1

    return count



def reactor_metrics(array, fuel):
    x_max, y_max, z_max = array.shape
    total_fuel_cells = 0
    total_neighbor = 0.0
    total_heat_multiplier = 0.0
    power = 0.0
    heat = 0.0
    total_cooling = 0.0
    
    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                element = int(array[x, y, z])
                if element in REACTOR_BLOCK_PROPERTIES:
                    total_cooling += REACTOR_BLOCK_PROPERTIES[element]['cooling_value']
                    if element == 1:
                        total_fuel_cells += 1
                        adjacent_cells = count_neighbors(array, x, y, z, 1)
                        mod = count_neighbors(array, x, y, z, 2)
                        base_energy_gen = REACTOR_FUEL_TYPE[fuel]['energy_gen']
                        base_heat_gen = REACTOR_FUEL_TYPE[fuel]['heat_gen']
                        if adjacent_cells > 0:
                            energy_multiplier = (adjacent_cells + 1) * base_energy_gen
                            power += energy_multiplier
                            heat_multiplier = (adjacent_cells + 1) * (adjacent_cells + 2) / 2 * base_heat_gen
                            heat += heat_multiplier
                            total_neighbor += adjacent_cells + 1
                            total_heat_multiplier += (adjacent_cells + 1) * (adjacent_cells + 2) / 2
                        elif adjacent_cells == 0 and mod == 0:
                            power += base_energy_gen
                            heat += base_heat_gen
                            total_neighbor += 1
                            total_heat_multiplier += 1 * 2 / 2
                        elif mod > 0:
                            energy_multiplier = base_energy_gen * (1 + (mod / 6))
                            power += energy_multiplier
                            heat_multiplier = base_heat_gen * (1 + (mod / 3))
                            heat += heat_multiplier
                            total_neighbor += (1 + (mod / 6))
                            total_heat_multiplier += (1 + (mod / 3))

    efficiency = 100 * total_neighbor / total_fuel_cells if total_fuel_cells != 0 else 0
    heat_multiplier = 100 * total_heat_multiplier / total_fuel_cells if total_fuel_cells != 0 else 0

    metrics = {
        "heat_gen": heat,
        "cooling": total_cooling,
        "fuel_cells": total_fuel_cells,
        "energy_gen": power,
        "efficiency": int(efficiency),
        "heat_multiplier": heat_multiplier,
        "total_energy": power * total_fuel_cells,
        "heat_diff": int(heat - total_cooling)
    }
    return metrics



array2 = np.array([
    [[1, 3, 0],
     [1, 2, 0],
     [1, 0, 0]],

    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],

    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]
], dtype=int)

#print(reactor_metrics(array, "TBU"))
#print(reactor_metrics(array2, "TBU"))