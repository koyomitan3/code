import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from metrics import reactor_metrics
from nuclear_reactor import validate_array
from plot_utils import plot_grid
import time
from converters import pad_array

torch.set_default_device('cuda')

# Constants
SIZE = (3, 3, 3)
CURRENT_FUEL = "TBU"
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.11
CROSSOVER_RATE = 0.04
TOURNAMENT_SIZE = 5

def save_model(model, optimizer, epoch, path='underhaul_optimizer.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Initialize policy network (example using PyTorch)
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dense1 = nn.Linear(np.prod(SIZE), 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)  # Output layer for action probability

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        return torch.sigmoid(self.output_layer(x))

# Initialize the policy network and optimizer
policy_network = PolicyNetwork()
optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

def generate_population(size):
    population = [np.zeros(SIZE, dtype=int) for _ in range(size)]
    return population

def fitness(individual):
    validate_array(pad_array(individual))
    metrics = reactor_metrics(individual, CURRENT_FUEL)
    if metrics['heat_diff'] > 0:
        return 0
    else:
        return (1.9 * metrics['energy_gen']) + (0.8 * metrics['heat_gen']) + (metrics['heat_diff'] * -1) + (1.3 * (metrics['efficiency'] / 100))

# Function to mutate an individual configuration
def mutate(individual):
    mask = np.random.rand(*individual.shape) < MUTATION_RATE
    individual[mask] = np.random.randint(0, 18, size=np.count_nonzero(mask))
    return individual

# Function to perform crossover between two individual configurations
def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, np.prod(SIZE))
        child = np.concatenate((parent1.flat[:point], parent2.flat[point:])).reshape(SIZE)
        return child
    else:
        return parent1.copy()

# Function to perform tournament selection for parent selection
def tournament_selection(population, fitnesses):
    selected = [np.random.randint(0, len(population)) for _ in range(TOURNAMENT_SIZE)]
    best = sorted(selected, key=lambda idx: fitnesses[idx], reverse=True)[0]
    return population[best]

# Function to run the DRL-based optimization
def run_drl_optimization():
    t1 = time.perf_counter()
    population = generate_population(POPULATION_SIZE)
    best_individual = None
    best_fitness = -float('inf')

    for generation in range(GENERATIONS):
        new_population = []
        fitnesses = [fitness(individual) for individual in population]

        for individual in population:
            state = torch.tensor(individual.flatten(), dtype=torch.float32).unsqueeze(0)
            
            # Perform action (mutation or crossover) based on policy network output
            action_prob = policy_network(state)
            if np.random.rand() < action_prob.item():
                mutated_individual = mutate(np.copy(individual))
            else:
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                crossover_individual = crossover(parent1, parent2)
                # Assign mutated_individual before the conditional expression:
                mutated_individual = mutate(crossover_individual) if np.random.rand() < action_prob.item() else crossover_individual

            try:
                validate_array(mutated_individual)
            except Exception as e:
                print(f"Validation error in generation {generation}, replacing invalid child: {e}")
                mutated_individual = generate_population(POPULATION_SIZE)

            new_population.append(mutated_individual)

        population = new_population
        best_fitness = max(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)

    # Return the best solution found
    best_index = np.argmax(fitnesses)
    best_individual = population[best_index]
    new_matrix = validate_array(best_individual)

    print("Best solution found:", new_matrix)
    print("Best solution fitness:", best_fitness)

    t2 = time.perf_counter()
    print(f"Time taken {t2 - t1} seconds")

    # Additional optional: Plotting and metrics display
    print(reactor_metrics(new_matrix, CURRENT_FUEL))
    plot_grid(new_matrix, 'new_array_cuda.png')
    print(f"Saved to {plot_grid.__name__}")

# Run the DRL-based optimization
run_drl_optimization()
