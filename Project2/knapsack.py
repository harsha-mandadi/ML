from mlrose import mlrose_hiive as mlrose
import numpy as np
from random import randint, seed
import csv
import time
import matplotlib.pyplot as plt
from mlrose_hiive.algorithms.decay import GeomDecay


# def queens_cust(state):
#     cnt = 0
#     for i in range(len(state)-1):
#         for j in range(i + 1, len(state)):
#             if (state[j] != state[i]) \
#                     and (state[j] != state[i] + (j - i)) \
#                     and (state[j] != state[i] - (j - i)):

#                 cnt += 1

#     return cnt


# cust_fitness = mlrose.CustomFitness(queens_cust)
n_items = 8
max_val = 5
weights = np.random.choice(range(1, 10), n_items)
values = np.random.choice(range(1, max_val), n_items)

fitness_fn = mlrose.Knapsack(weights, values)

problem_fit = mlrose.DiscreteOpt(
    length=n_items, fitness_fn=fitness_fn, max_val=max_val)

# cust_fitness = mlrose.Knapsack
# problem_fit = mlrose.DiscreteOpt(length=8, fitness_fn=cust_fitness,
#                                 maximize = False, max_val = 8

#                                )

GA_start_time = time.time()
# Solve problem using the genetic algorithm
# best_state1, best_fitness1, fitness_curve1, curve_time = mlrose.genetic_alg(
#     problem_fit, random_state=2, curve='True', max_iters=2500, mutation_prob=0.1, max_attempts=50)
best_state2, best_fitness2, fitness_curve2, curve_time = mlrose.genetic_alg(
    problem_fit, random_state=2, curve='True', mutation_prob=0.5, max_attempts=50)
# best_state3, best_fitness3, fitness_curve3, curve_time = mlrose.genetic_alg(
#     problem_fit, random_state=2, curve='True',  mutation_prob=0.5, max_attempts=20)
# best_state4, best_fitness4, fitness_curve4, curve_time = mlrose.genetic_alg(
#     problem_fit, random_state=2, curve='True', max_iters=2500, mutation_prob=0.7, max_attempts=50)
print(time.time()-GA_start_time)

# SA
# SA_start_time = time.time()
# best_state1, best_fitness1, fitness_curve1 = mlrose.simulated_annealing(
#     problem_fit, curve='True', schedule=GeomDecay(decay=0.15))
# print(time.time()-SA_start_time)
# best_state2, best_fitness2, fitness_curve2 = mlrose.simulated_annealing(
#     problem_fit, curve='True', schedule=GeomDecay(decay=0.35))
best_state3, best_fitness3, fitness_curve3 = mlrose.simulated_annealing(
    problem_fit, curve='True', schedule=GeomDecay(decay=0.15))
# best_state3, best_fitness4, fitness_curve4 = mlrose.simulated_annealing(
#     problem_fit, curve='True', schedule=GeomDecay(decay=0.75))

MIMIC_start_time = time.time()
# MIMIC
best_state1, best_fitness1, fitness_curve1 = mlrose.mimic(
    problem_fit, curve='True', keep_pct=0.3)
# best_state2, best_fitness2, fitness_curve2 = mlrose.mimic(
#     problem_fit, curve='True', keep_pct=0.3)
# best_state3, best_fitness3, fitness_curve3 = mlrose.mimic(
#     problem_fit, curve='True', keep_pct=0.3)
# best_state4, best_fitness4, fitness_curve4 = mlrose.mimic(
#     problem_fit, curve='True', keep_pct=0.7)
print(time.time()-MIMIC_start_time)

RHC_start_time = time.time()
# RHC
best_state4, best_fitness4, fitness_curve4 = mlrose.random_hill_climb(
    problem_fit, curve='True')
print(time.time()-RHC_start_time)
# print('The best state found is: ', best_state)

print('The fitness at the best state is: ', best_fitness1)
print('The fitness at the best state is: ', best_fitness2)
print('The fitness at the best state is: ', best_fitness3)
print('The fitness at the best state is: ', best_fitness4)

# print(type(fitness_curve1))

# print(len(fitness_curve))
# print(curve_time[52]-curve_time[50])
# with open("tsp_ga.csv", 'wb') as f:
#    csv.writer(f, delimiter=' ').writerows(fitness_curve)
# np.savetxt('data.csv', fitness_curve, delimiter=',')

plt.plot(fitness_curve1, label="MIMIC")
plt.plot(fitness_curve2, label="GA")
#plt.plot(fitness_curve3, label="SA")
plt.plot(fitness_curve4, label="RHC")
# plt.plot(fitness_curve1, label="0.1")
# plt.plot(fitness_curve2, label="0.3")
# plt.plot(fitness_curve3, label="0.5")
# plt.plot(fitness_curve4, label="0.7")
plt.title("Knapsack")
plt.xlabel("iterations")
plt.ylabel("fitness")
plt.legend()
plt.show()
