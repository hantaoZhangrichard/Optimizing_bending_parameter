import numpy as np
from rl_env import bending_env

# Genetic Algorithm Parameters
population_size = 1
mutation_rate = 0.1
num_step = 4

# Initialize population
def initialize_population(population_size, num_step):
    return [np.random.randint(1, 10, size=4) for _ in range(population_size)]

# Select parents for crossover using tournament selection
def select_parents(population, fitness):
    tournament_size = 3
    selected_parents = []
    for _ in range(2):  # Select 2 parents for crossover
        tournament = np.random.randint(0, len(population), size=tournament_size)
        # print(tournament)
        index = fitness.index(max(fitness[i] for i in tournament))
        selected_parents.append(population[index])
    return selected_parents

# Perform crossover to produce offspring
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.hstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Perform mutation on an individual
def mutate(individual, mutation_rate):
    mutated_individual = individual
    for i in range(len(individual)):
        if np.random.uniform(size=1) < mutation_rate:
            noise = np.random.randint(low = -2, high = 2, size=1)
            mutated_individual[i] += noise
    return mutated_individual

if __name__ == "__main__":
    population = initialize_population(population_size, num_step)
    env = bending_env()
    
    num_generation = 1
    best_fitness = np.empty((num_generation, ))
    for i in range(num_generation):
        print("Generation:{}".format(i))
        fitness = np.empty((population_size, ))
        new_population = []
        for j in range(population_size // 2):
            env.reset()
            individual = population[j]
            for k in range(len(individual)):
                action = individual[k]
                _, reward, done = env.step(action)
                if done:
                    break
            fitness[j] = reward
            parents = select_parents(population, fitness)
            # print(parents)
            parent1 = parents[0]
            parent2 = parents[1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            # print(mutate(child1, mutation_rate))
            new_population.extend([child1, child2])
        
        best_fitness[i] = max(fitness)
        print("Best fitness score in this generation:{}".format(max(fitness)))
        best_individual = population[fitness.index(max(fitness))]
        print("Best individual in this generation:{}".format(best_individual))
        population = new_population
    
    print(best_fitness)

