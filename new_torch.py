import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from metrics import reactor_metrics
from plot_utils import plot_grid
import time
import random
from nuclear_reactor import is_array_valid, is_valid, get_neighbors
torch.set_default_device('cuda')



def generate_random_size():
    # Generate random dimensions a, b, c
    a = random.randint(3, 5)
    b = random.randint(3, 5)
    c = random.randint(3, 5)
    
    # Return the tuple (a, b, c)
    return (a, b, c)


# Constants
#SIZE = generate_random_size()
SIZE =  (3, 3, 3)
CURRENT_FUEL = "TBU"
POPULATION_SIZE = 100
GENERATIONS = 10
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
    def __init__(self, input_shape):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, padding=1)
        flattened_size = self._get_flattened_size(input_shape)
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.view(-1)
    
    def _get_flattened_size(self, input_shape):
        x = torch.randn(1, 1, *input_shape)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x.view(1, -1).size(1) 


def generate_population(size, target_size):
    population = [np.zeros(target_size, dtype=int) for _ in range(size)]
    return population
def fitness(individual, fuel):
    metrics = reactor_metrics(individual, fuel)

    if not is_array_valid(individual):
        return -1000  # Large negative penalty for invalid arrays

    # More nuanced heat penalty 
    if metrics['heat_diff'] > 0:
        heat_penalty = -100 * metrics['heat_diff'] # Very high penalty for positive heat diff
    else:
        heat_penalty = metrics['heat_diff'] * 0.1  # Small reward for negative heat diff 

    return (1.9 * metrics['energy_gen']) + (0.8 * metrics['heat_gen']) + heat_penalty + (1.3 * (metrics['efficiency'] / 100))

# Function to mutate an individual configuration
def mutate(individual, policy_network):
    state = torch.tensor(individual, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    action_probs = F.softmax(policy_network(state), dim=-1).squeeze(0).cpu().detach().numpy()  # Fix softmax dimension
    chosen_action = np.random.choice(2, p=action_probs)
    original_fitness = fitness(individual, CURRENT_FUEL)

    for i in range(individual.shape[0]):
        for j in range(individual.shape[1]):
            for k in range(individual.shape[2]):
                if random.random() < MUTATION_RATE:
                    chosen_action = np.random.choice(2, p=action_probs)

                    if chosen_action == 0:  # Swap with a valid neighbor
                        neighbors = get_neighbors(individual, i, j, k)
                        direction = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)])
                        valid_neighbor_indices = [
                            n for n, neighbor in enumerate(neighbors)
                            if (
                                0 <= i + direction[n][0] < individual.shape[0] and  # Check x-bound
                                0 <= j + direction[n][1] < individual.shape[1] and  # Check y-bound
                                0 <= k + direction[n][2] < individual.shape[2] and  # Check z-bound
                                is_valid(individual[i, j, k], np.concatenate((neighbors[:n], neighbors[n + 1:]))) and
                                is_valid(neighbor, np.concatenate((neighbors[:n], neighbors[n + 1:], [individual[i, j, k]])))
                            )
                        ]
                        if valid_neighbor_indices:
                            neighbor_index = random.choice(valid_neighbor_indices)
                            new_x, new_y, new_z = i + direction[neighbor_index][0], j + direction[neighbor_index][1], k + direction[neighbor_index][2]
                            individual[i, j, k], individual[new_x, new_y, new_z] = individual[new_x, new_y, new_z], individual[i, j, k]

                    elif chosen_action == 1:  # Replace with a valid element
                        neighbors = get_neighbors(individual, i, j, k)
                        valid_elements = [e for e in range(18) if is_valid(e, neighbors, for_replacement=True)]
                        if valid_elements:
                            individual[i, j, k] = random.choice(valid_elements)

    new_fitness = fitness(individual, CURRENT_FUEL)
    reward = new_fitness - original_fitness  # Calculate reward
    return individual, reward


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
def run_drl_optimization(target_size, fuel, population_size=100, generations=50, progress_function=None):
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

    for _ in range(generations):
        epoch += 1
        print(f"Generation: {epoch}")
        new_population = []
        fitnesses = [] 
        total_rewards = []  # Store rewards for the generation

        for individual in population:
            mutated_individual, reward = mutate(individual.copy(), policy_network)
            new_population.append(mutated_individual)
            fitnesses.append(fitness(mutated_individual, fuel))  # Calculate fitness after mutation
            total_rewards.append(reward)

        population = new_population
        fitness_tensor = torch.tensor(fitnesses, dtype=torch.float32, requires_grad=True).cuda()
        rewards_tensor = torch.tensor(total_rewards, dtype=torch.float32, requires_grad=True).cuda()
        
        optimizer.zero_grad()
        loss = -(rewards_tensor * fitness_tensor).mean()  
        loss.backward()
        optimizer.step()

        best_fitness = fitness_tensor.max().item()

    best_index = np.argmax(fitnesses)
    best_individual = population[best_index]
    best_metrics = reactor_metrics(best_individual, fuel)
    #print("Best solution found:", best_individual)

    color = '\033[92m' if best_fitness > 1000 else '\033[93m' if 0 < best_fitness < 1000 else '\033[91m'
    print(f"{color}Best fitness: {int(best_fitness)}\033[0m")

    t2 = time.perf_counter()
    print(f"Time taken {t2 - t1} seconds")

    print("Metrics:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")

    plot_grid(best_individual, 'new_array_cuda.png')
    print(f"Saved to new_array_cuda.png")
    return best_individual if is_array_valid(best_individual) else None





run_drl_optimization(SIZE, CURRENT_FUEL, population_size=POPULATION_SIZE, generations=50)