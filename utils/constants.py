import numpy as np


REACTOR_BLOCK_ID = {
    -1: "C",  # Reactor Casing
    0 : "0",  # Air
    1 : "[]", # Reactor Cell
    2 : "##", # Moderator
    3 : "Wt", # Water
    4 : "Rs", # Redstone
    5 : "Qz", # Quartz
    6 : "Au", # Gold
    7 : "Gs", # Glowstone
    8 : "Lp", # Lapis
    9 : "Dm", # Diamond
    10 : "He", # Helium
    11: "Ed", # Enderium
    12: "Cr", # Cryotheum
    13: "Fe", # Iron
    14: "Em", # Emerald
    15: "Cu", # Copper
    16: "Sn", # Tin
    17: "Mg", # Magnesium
}

REACTOR_BLOCK_COLOR = {
    -1: 'gray',         # Reactor Casing
    0: 'white',         # Air
    1: 'lightgray',     # Reactor Cell
    2: 'black',         # Moderator
    3: 'blue',          # Water
    4: 'red',           # Redstone
    5: 'lightgoldenrodyellow', # Quartz
    6: 'gold',          # Gold
    7: 'yellow',        # Glowstone
    8: 'blueviolet',    # Lapis
    9: 'deepskyblue',   # Diamond
    10:'lightblue',     # Helium
    11:'purple',        # Enderium
    12:'lightgreen',    # Cryotheum
    13:'gray',          # Iron
    14:'green',         # Emerald
    15:'orange',        # Copper
    16:'brown',         # Tin
    17:'darkcyan'       # Magnesium
}


# Units in order: H/t, s, s, RF, RF/t

REACTOR_FUEL_TYPE = {
    "TBU": {
        "heat_gen": 18,
        "fuel_pellet_duration": 7200,
        "meltdown_time": 1875,
        "energy_per_pellet": 8640000,
        "energy_gen": 60
    },
    "TBU-Oxide": {
        "heat_gen": 200,
        "fuel_pellet_duration": 200,
        "meltdown_time": 400,
        "energy_per_pellet": 75,
        "energy_gen": 1500
    },
    "LEU-233": {
        "heat_gen": 150,
        "fuel_pellet_duration": 400,
        "meltdown_time": 600,
        "energy_per_pellet": 60,
        "energy_gen": 1200
    },
    "LEU-233-Oxide": {
        "heat_gen": 250,
        "fuel_pellet_duration": 100,
        "meltdown_time": 300,
        "energy_per_pellet": 90,
        "energy_gen": 2000
    },
    "LEU-235": {
        "heat_gen": 50,
        "fuel_pellet_duration": 3600,
        "meltdown_time": 450,
        "energy_per_pellet": 8640000,
        "energy_gen": 120
    }
}


REACTOR_BLOCK_PROPERTIES = {
    -1: {
        'cooling_value': 0,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    0: {
        'cooling_value': 0,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    1: {
        'cooling_value': 0,   # Placeholder
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 1
    },
    2: {
        'cooling_value': 0,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    3: {
        'cooling_value': 60,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    4: {
        'cooling_value': 90,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    5: {
        'cooling_value': 90,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    6: {
        'cooling_value': 120,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    7: {
        'cooling_value': 130,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    8: {
        'cooling_value': 120,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    9: {
        'cooling_value': 150,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    10: {
        'cooling_value': 140,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    11: {
        'cooling_value': 120,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    12: {
        'cooling_value': 160,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    13: {
        'cooling_value': 80,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    14: {
        'cooling_value': 160,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    15: {
        'cooling_value': 80,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    16: {
        'cooling_value': 120,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    },
    17: {
        'cooling_value': 110,
        'heat_gen': 0,
        'energy_gen': 0,
        'efficiency': 0
    }
}
