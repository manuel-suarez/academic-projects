from nas import toolbox
from deap import tools
import random


# Initialize population
population = toolbox.population(n=10)
NGEN = 5
CXPB, MUTPB = 0.5, 0.2

for gen in range(NGEN):
    print(f"Generation {gen}")
    offspring = list(map(toolbox.clone, population))

    # Apply crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit,)

    # Select next generation
    population[:] = toolbox.select(offspring, len(population))

# Best architecture
best_ind = tools.selBest(population, 1)[0]
print("Best architecture: ", best_ind)
