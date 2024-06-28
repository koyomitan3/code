import os
import sys
from deep_learning_main import run_drl_optimization, generate_random_size
from utils.constants import REACTOR_FUEL_TYPE
import random

def _getrandom_fuel():
    return random.choice(list(REACTOR_FUEL_TYPE.keys()))

def get_latest_checkpoint(checkpoint_dir='./models/checkpoints'):
    try:
        files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        if not files:
            print("No checkpoint files found.")
            return None
        latest_file = max(files, key=lambda f: int(f.split('_')[2].split('.')[0]))
        return os.path.join(checkpoint_dir, latest_file)
    except Exception as e:
        print(f"Error while searching for the latest checkpoint: {e}")
        return None

def main():
    SIZE = generate_random_size()
    CURRENT_FUEL = _getrandom_fuel()
    POPULATION_SIZE = 100
    GENERATIONS = 10000
    resume_checkpoint = None

    if len(sys.argv) > 1 and sys.argv[1] == 'resume':
        resume_checkpoint = get_latest_checkpoint()
        if resume_checkpoint:
            print(f"Resuming from latest checkpoint: {resume_checkpoint}")
        else:
            print("No valid checkpoint found. Starting from scratch.")

    run_drl_optimization(SIZE, CURRENT_FUEL, population_size=POPULATION_SIZE, generations=GENERATIONS, resume_checkpoint=resume_checkpoint)

if __name__ == "__main__":
    main()
