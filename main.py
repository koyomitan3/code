import numpy as np
from metrics import reactor_metrics
from plot_utils import plot_grid
from nuclear_reactor import validate_array
import time
from numba import njit

POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_RATE = 0.01
CROSSOVER_RATE = 3
TOURNAMENT_SIZE = 10
SIZE = (3, 3, 3)
CURRENT_FUEL = "TBU"


@njit
def generate_array():
    return np.random.randint(1, 17, SIZE)

@njit
def initialize_population(size):
    return [generate_array() for _ in range(size)] 


def fitness(individual):
    validate_array(individual)
    metrics = reactor_metrics(individual, CURRENT_FUEL)
    if metrics['heat_diff'] > 0:
        return 0
    else:
        return (1.9 * metrics['energy_gen']) + 0.8 * metrics['heat_gen'] + (metrics['heat_diff'] * -1) + 1.3 * (metrics['efficiency'] / 100)


def tournament_selection(population, fitnesses):
    selected = [np.random.randint(0, len(population)) for _ in range(TOURNAMENT_SIZE)]
    best = sorted(selected, key=lambda idx: fitnesses[idx], reverse=True)[0]
    return population[best]


def crossover(parent1, parent2, crossover_rate=CROSSOVER_RATE):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, np.prod(SIZE))
        child = np.concatenate((parent1.flat[:point], parent2.flat[point:])).reshape(SIZE)
        return child
    else:
        return parent1.copy()


def mutate(individual, mutation_rate=MUTATION_RATE):
    mask = np.random.rand(*individual.shape) < mutation_rate
    individual[mask] = np.random.randint(1, 17, size=np.count_nonzero(mask))
    return individual

def genetic_algorithm():
    # Initialize population
    population = initialize_population(POPULATION_SIZE)
    
    for generation in range(GENERATIONS):
        # Evaluate fitness of the population
        fitnesses = [fitness(ind) for ind in population]

        # Create new population
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            try:
                validate_array(child)  # Validate the child array to catch any issues early
            except Exception as e:
                print(f"Validation error in generation {generation}, replacing invalid child: {e}")
                child = generate_array()  # Replace invalid child with a new random array
            new_population.append(child)
        
        population = new_population
        
        # Optionally print or record statistics
        best_fitness = max(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Generation {generation}: Best Fitness: {best_fitness}, Average Fitness: {average_fitness}")

    # Return the best solution found
    fitnesses = [fitness(ind) for ind in population]
    best_index = fitnesses.index(max(fitnesses))
    return population[best_index], fitnesses[best_index]

# Run the genetic algorithm
t1 = time.perf_counter()
best_solution, best_fitness = genetic_algorithm()
new_matrix = validate_array(best_solution)
print("Best solution found:", new_matrix)
print("Best solution fitness:", best_fitness)
t2 = time.perf_counter()
print(f"Time taken {t2 - t1} seconds")
print(reactor_metrics(new_matrix, CURRENT_FUEL))
plot_grid(validate_array(new_matrix), 'new_array_cpu.png')
print(f"Saved to {plot_grid.__name__}")