import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import reactor_metrics
from nuclear_reactor import validate_array
from plot_utils import plot_grid
import time
from converters import pad_array
import optuna

torch.set_default_device('cuda')

device = torch.device('cuda')


# Constants
SIZE = (3, 3, 3)
CURRENT_FUEL = "TBU"
POPULATION_SIZE = 48
GENERATIONS = 100

def save_model(model, optimizer, epoch, path='underhaul_optimizer.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Initialize policy network (example using PyTorch)
class PolicyNetwork(nn.Module):
    def __init__(self, num_hidden_layers, num_neurons):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(np.prod(SIZE), num_neurons))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
        self.output_layer = nn.Linear(num_neurons, 1)  # Output layer for action probability

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return torch.sigmoid(self.output_layer(x))

def generate_population(size):
    population = [np.random.randint(0, 18, SIZE) for _ in range(size)]
    return population

def fitness(individual):
    validate_array(pad_array(individual))
    metrics = reactor_metrics(individual, CURRENT_FUEL)
    if metrics['heat_diff'] > 0:
        return 0
    else:
        return (1.9 * metrics['energy_gen']) + (0.8 * metrics['heat_gen']) + (metrics['heat_diff'] * -1) + (1.3 * (metrics['efficiency'] / 100))

# Function to mutate an individual configuration
def mutate(individual, mutation_rate):
    mask = np.random.rand(*individual.shape) < mutation_rate
    individual[mask] = np.random.randint(0, 18, size=np.count_nonzero(mask))
    return individual

# Function to perform crossover between two individual configurations
def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, np.prod(SIZE))
        child = np.concatenate((parent1.flat[:point], parent2.flat[point:])).reshape(SIZE)
        return child
    else:
        return parent1.copy()

# Function to perform tournament selection for parent selection
def tournament_selection(population, fitnesses, tournament_size):
    selected = [np.random.randint(0, len(population)) for _ in range(tournament_size)]
    best = sorted(selected, key=lambda idx: fitnesses[idx], reverse=True)[0]
    return population[best]

# Objective function for Optuna
def objective(trial):
    # Define hyperparameter search space
    mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.2)
    crossover_rate = trial.suggest_float("crossover_rate", 0.1, 0.9)
    tournament_size = trial.suggest_int("tournament_size", 2, 10)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
    num_neurons = trial.suggest_int("num_neurons", 16, 128, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

    # Initialize policy network and optimizer based on chosen hyperparameters
    policy_network = PolicyNetwork(num_hidden_layers, num_neurons).to(device)
    optimizer = getattr(torch.optim, optimizer_name)(policy_network.parameters(), lr=0.001)

    t1 = time.perf_counter()
    population = generate_population(POPULATION_SIZE)
    best_individual = None
    best_fitness = -float('inf')

    for generation in range(GENERATIONS):
        new_population = []
        fitnesses = [fitness(individual) for individual in population]

        for individual in population:
            state = torch.tensor(individual.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

            # Perform action (mutation or crossover) based on policy network output
            with torch.no_grad():
                action_prob = policy_network(state)
                if np.random.rand() < action_prob.item():
                    mutated_individual = mutate(np.copy(individual), mutation_rate)
                else:
                    parent1 = tournament_selection(population, fitnesses, tournament_size)
                    parent2 = tournament_selection(population, fitnesses, tournament_size)
                    crossover_individual = crossover(parent1, parent2, crossover_rate)
                    mutated_individual = mutate(crossover_individual, mutation_rate) if np.random.rand() < action_prob.item() else crossover_individual

            try:
                validate_array(mutated_individual)
            except Exception as e:
                print(f"Validation error in generation {generation}, replacing invalid child: {e}")
                mutated_individual = generate_population(1)[0]

            new_population.append(mutated_individual)

        population = new_population
        best_fitness = max(fitnesses)
        new_best_index = np.argmax(fitnesses)
        best_individual = population[new_best_index]

        average_fitness = sum(fitnesses) / len(fitnesses)

        # Report intermediate results to Optuna
        trial.report(best_fitness, generation)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the best fitness achieved
    return best_fitness

if __name__ == "__main__":
    # Create an Optuna study object
    study = optuna.create_study(direction="maximize")

    # Run Optuna optimization
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters and best fitness
    print("Best hyperparameters: ", study.best_params)
    print("Best fitness: ", study.best_value)

    # You can access the best individual from the trial that achieved the best fitness
    # It's not directly returned by Optuna, so you need to store it during the optimization process.