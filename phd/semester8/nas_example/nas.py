import random
from deap import base, creator, tools
from model import CandidateCNN
from train_eval import evaluate_model

# Define Fitness and Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# Genetic Operators
def create_individual():
    num_layers = random.randint(3, 5)
    individual = []
    for _ in range(num_layers):
        individual.append(
            {
                "type": "conv",
                "out_channels": random.choice([16, 32, 64]),
                "kernel_size": random.choice([3, 5]),
            }
        )
        individual.append({"type": "relu"})
        if random.random() > 0.5:
            individual.append({"type": "pool"})
    return individual


def mutate(individual):
    layer_idx = random.randint(0, len(individual) - 1)
    if individual[layer_idx]["type"] == "conv":
        individual[layer_idx]["out_channels"] = random.choice([16, 32, 64])
    return individual


def crossover(parent1, parent2):
    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def evaluate(individual):
    model = CandidateCNN(individual)
    fitness = evaluate_model(model)
    return (fitness,)


# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
