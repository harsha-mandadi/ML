from mlrose import mlrose_hiive as mlrose
import numpy as np
from random import randint, seed
import csv
import time
import matplotlib.pyplot as plt


# Create list of city coordinates
N = 100
coords_list = []

# i = randint(0, 100)

# j = randint(0, 100)
# while(len(coords_list) < N):
#     while (i, j) in coords_list:

#         i = randint(0, 100)

#         j = randint(0, 100)
#     coords_list.append((i, j))

# print(coords_list)
# coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
# coords_list = [(61, 34), (42, 9), (65, 32), (81, 9), (8, 35), (91, 41), (91, 18), (96, 49), (50, 31), (91, 88), (78, 16), (75, 67), (59, 45), (24, 46), (40, 34), (78, 47), (67, 83), (74, 34), (64, 63), (27, 67), (31, 89), (26, 70), (73, 9), (43, 44), (54, 32), (20, 34), (43, 11), (92, 53), (57, 19), (82, 51), (9, 8), (7, 74), (90, 93), (66, 52), (63, 35), (37, 45), (21, 75), (47, 86), (12, 36), (44, 74), (2, 93), (41, 38), (36, 50), (36, 90), (13, 17), (64, 8), (45, 30), (38, 16), (59, 91),
#               (36, 73), (54, 97), (88, 58), (43, 63), (69, 27), (4, 71), (21, 34), (65, 12), (76, 56), (87, 48), (34, 65), (18, 52), (65, 31), (0, 46), (20, 63), (8, 19), (36, 53), (18, 99), (24, 5), (98, 88), (17, 27), (55, 96), (9, 98), (65, 28), (66, 72), (89, 57), (19, 12), (75, 77), (48, 95), (54, 85), (54, 54), (35, 89), (78, 75), (51, 56), (80, 25), (53, 73), (95, 18), (9, 49), (54, 16), (76, 72), (66, 67), (1, 9), (65, 90), (52, 34), (96, 32), (86, 10), (68, 45), (96, 19), (58, 42), (74, 63), (7, 56)]
# Initialize fitness function object using coords_list
start_time = time.time()
fitness_coords = mlrose.ContinuousPeaks(t_pct=0.1)

problem_fit = mlrose.DiscreteOpt(N, fitness_coords)

# Solve problem using the genetic algorithm
# best_state1, best_fitness1, fitness_curve1, curve_time = mlrose.genetic_alg(
#     problem_fit, random_state=2, curve='True', max_iters=2500, mutation_prob=0.1, max_attempts=50)
# best_state2, best_fitness2, fitness_curve2, curve_time = mlrose.genetic_alg(
#     problem_fit, random_state=2, curve='True', max_iters=2500, mutation_prob=0.3, max_attempts=50)
# best_state3, best_fitness3, fitness_curve3, curve_time = mlrose.genetic_alg(
#     problem_fit, random_state=2, curve='True', max_iters=2500, mutation_prob=0.5, max_attempts=50)
# best_state4, best_fitness4, fitness_curve4, curve_time = mlrose.genetic_alg(
#     problem_fit, random_state=2, curve='True', max_iters=2500, mutation_prob=0.7, max_attempts=50)


# SA
# best_state1, best_fitness1, fitness_curve = mlrose.simulated_annealing(
#     problem_fit, curve='True',decay = 0.15)
# best_state2, best_fitness, fitness_curve = mlrose.simulated_annealing(
#     problem_fit, curve='True',decay = 0.35)
# best_state3, best_fitness, fitness_curve = mlrose.simulated_annealing(
#     problem_fit, curve='True',decay = 0.55)
# best_state3, best_fitness, fitness_curve = mlrose.simulated_annealing(
#     problem_fit, curve='True',decay = 0.75)

# MIMIC
# best_state, best_fitness, fitness_curve = mlrose.mimic(problem_fit, curve = 'True',keep_pct=0.1)
# best_state, best_fitness, fitness_curve = mlrose.mimic(
#     problem_fit, curve='True', keep_pct=0.3)
# best_state, best_fitness, fitness_curve = mlrose.mimic(
#     problem_fit, curve='True', keep_pct=0.5)
# RHC
best_state1, best_fitness1, fitness_curve1 = mlrose.random_hill_climb(
    problem_fit, curve='True')

#print('The best state found is: ', best_state)

print('The fitness at the best state is: ', best_fitness1)
# print('The fitness at the best state is: ', best_fitness2)
# print('The fitness at the best state is: ', best_fitness3)
# print('The fitness at the best state is: ', best_fitness4)

print(type(fitness_curve1))

# print(len(fitness_curve))
# print(curve_time[52]-curve_time[50])
# with open("tsp_ga.csv", 'wb') as f:
#    csv.writer(f, delimiter=' ').writerows(fitness_curve)
# np.savetxt('data.csv', fitness_curve, delimiter=',')

plt.plot(fitness_curve1, label="HC")
# plt.plot(fitness_curve2, label="0.3")
# plt.plot(fitness_curve3, label="0.5")
# plt.plot(fitness_curve4, label="0.7")
plt.title("TSP-RHC")
plt.xlabel("iterations")
plt.ylabel("fitness")
plt.legend()
plt.show()
