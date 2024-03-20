import numpy as np
from rl_env import bending_env

# Genetic Algorithm Parameters
population_size = 10
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

# Main Genetic Algorithm Loop
def genetic_algorithm(target_string, population_size, mutation_rate):
    population = initialize_population(population_size, target_string)
    generation = 1

    while True:
        # Calculate fitness of each individual
        fitness_scores = [calculate_fitness(individual, target_string) for individual in population]

        # Check for solution
        max_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(max_fitness)]
        if max_fitness == len(target_string):
            return best_individual, generation

        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fit: {max_fitness}/{len(target_string)}")

        # Select parents and perform crossover
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, target_string)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # Apply mutation
        population = [mutate(individual, mutation_rate) for individual in new_population]

        generation += 1

# Run the genetic algorithm
'''best_individual, generations = genetic_algorithm(TARGET_STRING, POPULATION_SIZE, MUTATION_RATE)
print(f"Solution found in {generations} generations: {best_individual}")'''

if __name__ == "__main__":
    population = initialize_population(population_size, num_step)
    fitness = 
    parents = select_parents(population, fitness)
    print(parents)
    parent1 = parents[0]
    parent2 = parents[1]
    child1, child2 = crossover(parent1, parent2)
    print(mutate(child1, mutation_rate))
    num_generation = 5
    for i in range(num_generation):
        fitness = []
        for j in range(population_size):
            env = bending_env(str(i), episode=str(j))
            individual = population[j]
            fitness = 

