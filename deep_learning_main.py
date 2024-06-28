import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

# Custom Imports
from utils.metrics import reactor_metrics
from visualization.plot_utils import plot_grid
from core.nuclear_reactor import is_array_valid, is_valid, get_neighbors
# Set the default device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)


def generate_random_size():
    # Generate random dimensions a, b, c
    a = random.randint(3, 4)
    b = random.randint(3, 4)
    c = random.randint(3, 4)

    # Return the tuple (a, b, c)
    return (a, b, c)


# Constants
SIZE = generate_random_size()
#SIZE =  (3, 3, 3)
CURRENT_FUEL = "TBU"
POPULATION_SIZE = 75
GENERATIONS = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.04
TOURNAMENT_SIZE = 7
IMAGE_PATH = 'visualization/img'

# Initialize policy network (example using PyTorch)
class PolicyNetwork(nn.Module):

    def __init__(self, input_shape):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, padding=1)
        flattened_size = self._get_flattened_size(input_shape)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1)

    def _get_flattened_size(self, input_shape):
        x = torch.randn(1, 1, *input_shape)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x.view(1, -1).size(1)


def save_checkpoint(epoch,
                    model,
                    optimizer,
                    population,
                    best_individual,
                    best_fitness,
                    checkpoint_dir='./models/checkpoints'):
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(
            f"Checkpoint directory '{checkpoint_dir}' created or already exists."
        )
    except Exception as e:
        print(f"Error creating checkpoint directory '{checkpoint_dir}': {e}")
        return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'population': population,
        'best_individual': best_individual,
        'best_fitness': best_fitness
    }

    checkpoint_path = os.path.join(checkpoint_dir,
                                   f'checkpoint_epoch_{epoch}.pt')

    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint at '{checkpoint_path}': {e}")


def save_model(model, model_dir='./models'):
    try:
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory '{model_dir}' created or already exists.")
    except Exception as e:
        print(f"Error creating model directory '{model_dir}': {e}")
        return

    model_path = os.path.join(model_dir, 'model.pth')

    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model state saved at {model_path}")
    except Exception as e:
        print(f"Error saving model state at '{model_path}': {e}")


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    population = checkpoint['population']
    best_individual = checkpoint['best_individual']
    best_fitness = checkpoint['best_fitness']

    return epoch, model, optimizer, population, best_individual, best_fitness


def generate_population(size, target_size):
    # Create a single array of zeros with target_size
    population = np.zeros(target_size, dtype=int)

    # Repeat the population array 'size' times along a new axis
    new_population = np.repeat(population[np.newaxis, :], size, axis=0)

    return new_population


def fitness(individual, fuel):
    metrics = reactor_metrics(individual, fuel)

    if not is_array_valid(individual):
        return 0

    # More nuanced heat penalty
    if metrics['heat_diff'] > 0:
        heat_penalty = -100 * metrics[
            'heat_diff']  # Very high penalty for positive heat diff
    else:
        heat_penalty = metrics[
            'heat_diff'] * 0.1  # Small reward for negative heat diff

    return (1.9 * metrics['energy_gen']) + (
        0.8 *
        metrics['heat_gen']) + heat_penalty + (1.3 *
                                               (metrics['efficiency'] / 100))


# Function to mutate an individual configuration
def mutate(individual, policy_network):
    original_fitness = fitness(individual, CURRENT_FUEL)
    individual = individual.copy()
    state = torch.tensor(individual,
                         dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    action_probs = F.softmax(
        policy_network(state),
        dim=-1).squeeze(0).cpu().detach().numpy()  # Fix softmax dimension
    chosen_action = np.random.choice(2, p=action_probs)
    with torch.no_grad():  # Disable gradient calculation during inference
        action_probs = F.softmax(policy_network(state), dim=-1).squeeze(0).cpu().numpy()

    for i in range(individual.shape[0]):
        for j in range(individual.shape[1]):
            for k in range(individual.shape[2]):
                if random.random() < MUTATION_RATE:
                    chosen_action = np.random.choice(2, p=action_probs)

                    if chosen_action == 0:  # Swap with a valid neighbor
                        neighbors = get_neighbors(individual, i, j, k)
                        direction = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                                              (0, 1, 0), (0, 0, -1), (0, 0, 1)])
                        valid_neighbor_indices = [
                            n for n, neighbor in enumerate(neighbors)
                            if (0 <= i + direction[n][0] < individual.shape[0]
                                and  # Check x-bound
                                0 <= j + direction[n][1] < individual.shape[1]
                                and  # Check y-bound
                                0 <= k + direction[n][2] < individual.shape[2]
                                and  # Check z-bound
                                is_valid(
                                    individual[i, j, k],
                                    np.concatenate((neighbors[:n],
                                                    neighbors[n + 1:])))
                                and is_valid(
                                    neighbor,
                                    np.concatenate((neighbors[:n],
                                                    neighbors[n + 1:],
                                                    [individual[i, j, k]]))))
                        ]
                        if valid_neighbor_indices:
                            neighbor_index = random.choice(
                                valid_neighbor_indices)
                            new_x, new_y, new_z = i + direction[neighbor_index][
                                0], j + direction[neighbor_index][
                                    1], k + direction[neighbor_index][2]
                            individual[i, j, k], individual[
                                new_x, new_y,
                                new_z] = individual[new_x, new_y,
                                                    new_z], individual[i, j, k]

                    elif chosen_action == 1:  # Replace with a valid element
                        neighbors = get_neighbors(individual, i, j, k)
                        valid_elements = [
                            e for e in range(18) if is_valid(e, neighbors)
                        ]
                        if valid_elements:
                            individual[i, j, k] = random.choice(valid_elements)

    new_fitness = fitness(individual, CURRENT_FUEL)
    reward = new_fitness - original_fitness  # Calculate reward
    return individual, reward


# Function to perform crossover between two individual configurations
def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, np.prod(SIZE))
        child = np.concatenate(
            (parent1.flat[:point], parent2.flat[point:])).reshape(SIZE)
        return child
    else:
        return parent1.copy()


# Function to perform tournament selection for parent selection
def tournament_selection(population, fitnesses):
    selected = [
        np.random.randint(0, len(population)) for _ in range(TOURNAMENT_SIZE)
    ]
    best = sorted(selected, key=lambda idx: fitnesses[idx], reverse=True)[0]
    print(best)
    return population[best]


def save_checkpoint(epoch,
                    model,
                    optimizer,
                    population,
                    best_individual,
                    best_fitness,
                    checkpoint_dir='./models/checkpoints'):
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(
            f"Checkpoint directory '{checkpoint_dir}' created or already exists."
        )
    except Exception as e:
        print(f"Error creating checkpoint directory '{checkpoint_dir}': {e}")
        return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'population': population,
        'best_individual': best_individual,
        'best_fitness': best_fitness
    }

    checkpoint_path = os.path.join(checkpoint_dir,
                                   f'checkpoint_epoch_{epoch}.pt')

    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint at '{checkpoint_path}': {e}")


def load_checkpoint(checkpoint_path, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint from '{checkpoint_path}': {e}")
        return None, None, None, None, None, None

    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    population = checkpoint['population']
    best_individual = checkpoint['best_individual']
    best_fitness = checkpoint['best_fitness']

    return epoch, model, optimizer, population, best_individual, best_fitness


def run_drl_optimization(target_size,
                         fuel,
                         population_size=100,
                         generations=50,
                         resume_checkpoint=None):
    epoch = 0
    t1 = time.perf_counter()

    # Ensure target_size is a tuple of length 3
    if not isinstance(target_size, tuple) or len(target_size) != 3:
        raise ValueError("target_size must be a tuple of length 3 (N, M, P)")

    # Initialize policy network and optimizer
    policy_network = PolicyNetwork(target_size).cuda()
    optimizer = optim.Adam(policy_network.parameters(), lr=0.01)

    population = generate_population(population_size, target_size)
    best_individual = None
    best_fitness = -float('inf')

    if resume_checkpoint:
        epoch, policy_network, optimizer, population, best_individual, best_fitness = load_checkpoint(
            resume_checkpoint, policy_network, optimizer)

    for _ in range(epoch, generations):
        epoch += 1

        new_population = []
        fitnesses = []
        total_rewards = []

        for individual in population:
            mutated_individual, reward = mutate(individual.copy(),
                                                policy_network)
            new_population.append(mutated_individual)
            fitness_value = fitness(mutated_individual, fuel)

            # If the array is invalid, set fitness to zero
            if not is_array_valid(mutated_individual):
                fitness_value = 0

            fitnesses.append(fitness_value)
            total_rewards.append(reward)

        population = new_population
        fitness_tensor = torch.tensor(fitnesses,
                                      dtype=torch.float32,
                                      requires_grad=True).cuda()
        rewards_tensor = torch.tensor(total_rewards,
                                      dtype=torch.float32,
                                      requires_grad=True).cuda()

        optimizer.zero_grad()
        loss = -(rewards_tensor * fitness_tensor).mean()
        l2_lambda = 0.001  # Adjust the regularization strength
        l2_reg = torch.tensor(0.).cuda()
        for param in policy_network.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        loss.backward()
        optimizer.step()

        # Check for the best fitness among the current population
        current_best_fitness = fitness_tensor.max().item()
        best_index = fitness_tensor.argmax().item()
        best_candidate = population[best_index]

        # Update best_fitness only if the candidate is valid and has a better fitness value
        if is_array_valid(
                best_candidate) and current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = best_candidate

        # Print current highest fitness alongside the epoch/generation
        color = '\033[92m' if best_fitness > 1000 else '\033[93m' if 0 < best_fitness < 1000 else '\033[91m'
        if epoch % 10 == 0:
            print(
                f"Generation: {epoch} - Current Highest Fitness: {color}{best_fitness}\033[0m"
            )

        # Save checkpoint every 100 epochs
        if epoch % 50 == 0:
            save_checkpoint(epoch, policy_network, optimizer, population,
                            best_individual, best_fitness)
            save_model(policy_network, 'models')
    # After all generations, determine the best individual and its metrics
    if best_individual is None:
        print("No valid solution found.")
        t2 = time.perf_counter()
        print(f"Time taken {t2 - t1} seconds")
        return None

    best_metrics = reactor_metrics(best_individual, fuel)

    # Print and save results based on validity of the best individual
    if is_array_valid(best_individual):
        print(f"\033[92mBest fitness: {int(best_fitness)}\033[0m")
        print(f"Time taken {time.perf_counter() - t1} seconds")
        #print(f"{best_individual} is valid: True")
        print("Metrics:")
        for key, value in best_metrics.items():
            print(f"  {key.upper()}: {value}")
        #
    else:
        print(f"\033[91mBest fitness: {int(best_fitness)}\033[0m")
        print(f"Time taken {time.perf_counter() - t1} seconds")
        print(f"{best_individual} is valid: False")
        print("Metrics:")
        for key, value in best_metrics.items():
            print(f"  {key.upper()}: {value}")
        #imagePath = 'visualization/img'
        #
        return None


#run_drl_optimization(SIZE, CURRENT_FUEL, population_size=POPULATION_SIZE, generations=GENERATIONS)
