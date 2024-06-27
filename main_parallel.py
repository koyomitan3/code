import numpy as np
from metrics import reactor_metrics
from plot_utils import plot_grid
from nuclear_reactor import validate_array
import time
from multiprocessing import Pool, cpu_count

POPULATION_SIZE = 48
GENERATIONS = 1000
MUTATION_RATE = 0.01
CROSSOVER_RATE = 3
TOURNAMENT_SIZE = 10
SIZE = (3, 3, 3)
CURRENT_FUEL = "TBU"

def generate_array():
    return np.random.randint(1, 17, SIZE)

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

def crossover(parents):
    parent1, parent2 = parents
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, np.prod(SIZE))
        child = np.concatenate((parent1.flat[:point], parent2.flat[point:])).reshape(SIZE)
        return child
    else:
        return parent1.copy()

def mutate(individual):
    mask = np.random.rand(*individual.shape) < MUTATION_RATE
    individual[mask] = np.random.randint(1, 17, size=np.count_nonzero(mask))
    return individual

def evolve_population(population):
    with Pool(processes=cpu_count()) as pool:
        fitnesses = pool.map(fitness, population)

        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover((parent1, parent2))
            child = pool.apply(mutate, (child,))  # Apply mutation in parallel
            
            try:
                validate_array(child)
            except Exception as e:
                print(f"Validation error, replacing invalid child: {e}")
                child = generate_array()
                
            new_population.append(child)
    
    return new_population

def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)
    
    for generation in range(GENERATIONS):
        population = evolve_population(population)
        
        fitnesses = [fitness(ind) for ind in population]
        best_fitness = max(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Generation {generation}: Best Fitness: {best_fitness}, Average Fitness: {average_fitness}")
    
    fitnesses = [fitness(ind) for ind in population]
    best_index = fitnesses.index(max(fitnesses))
    return population[best_index], fitnesses[best_index]

if __name__ == "__main__":
    t1 = time.perf_counter()
    best_solution, best_fitness = genetic_algorithm()
    new_matrix = validate_array(best_solution)
    print("Best solution found:", new_matrix)
    print("Best solution fitness:", best_fitness)
    t2 = time.perf_counter()
    print(f"Time taken {t2 - t1} seconds")
    print(reactor_metrics(new_matrix, CURRENT_FUEL))
    plot_grid(new_matrix, 'new_array_parallel.png')
    print(f"Saved to {plot_grid.__name__}")
