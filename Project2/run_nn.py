#####Thanks to  https://github.com/donaldtf/ml-randomized-optimization for the following flow for RO on NN#######


import numpy as np
from mlrose import mlrose_hiive as mlrose
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.neural_network import MLPClassifier
from utils import get_pulsar_data, run_optimized, compute_stats
import sys
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# Thanks to this source for showing grid search with a neural net
# https://www.kaggle.com/hatone/mlpclassifier-with-gridsearchcv


def run_nn(name, x_train, x_test, y_train, y_test):
    print("Working on BackProp")

    report_name = "{}_nn_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    optimized_clf = MLPClassifier(
        max_iter=1000,
        alpha=0.001,
        hidden_layer_sizes=11,
        random_state=99
    )

    run_optimized(optimized_clf, x_train, y_train, x_test, y_test, name="BP")

    sys.stdout = sys.__stdout__


def run_rhc_nn(name, x_train, x_test, y_train, y_test):
    print("Working on RHC NN...")

    report_name = "{}_nn_rhc_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    rhc_nn = mlrose.NeuralNetwork(
        hidden_nodes=[11], algorithm='random_hill_climb', max_iters=1000, curve=True)

    # print(rhc_nn.fitness_curve[1:5])
    # plt.plot(rhc_nn.fitness_curve)
    # plt.show()
    run_optimized(rhc_nn, x_train, y_train, x_test, y_test, name="RHC")

    sys.stdout = sys.__stdout__


def run_sa_nn(name, x_train, x_test, y_train, y_test):
    print("Working on SA NN...")

    report_name = "{}_nn_sa_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    sa_nn = mlrose.NeuralNetwork(
        hidden_nodes=[11], algorithm='simulated_annealing', max_iters=1000, curve=True)

    # plt.plot(sa_nn.fitness_curve)
    # plt.show()
    run_optimized(sa_nn, x_train, y_train, x_test, y_test, name="SA")

    sys.stdout = sys.__stdout__


def run_ga_nn(name, x_train, x_test, y_train, y_test):
    print("Working on GA NN...")

    report_name = "{}_nn_ga_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    ga_nn = mlrose.NeuralNetwork(
        hidden_nodes=[11],
        algorithm='genetic_alg',
        max_iters=1000,
        early_stopping=True,
        clip_max=10,
        max_attempts=10,
        mutation_prob=0.15,
        curve=True
    )
    plt.plot(ga_nn.fitness_curve)
    # plt.show()
    run_optimized(ga_nn, x_train, y_train, x_test, y_test, name="GA")

    sys.stdout = sys.__stdout__


def run_pulsar_nn():
    name = 'Pulsar'

    x_train, x_test, y_train, y_test = get_pulsar_data()

    run_nn(name, x_train, x_test, y_train, y_test)

    run_rhc_nn(name, x_train, x_test, y_train, y_test)
    run_sa_nn(name, x_train, x_test, y_train, y_test)
    run_ga_nn(name, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    print("Running Neural Net Code, this should take a minute or two")

    np.random.seed(92)

    run_pulsar_nn()
    # plt.title("Neural Network")
    # plt.xlabel("iterations")
    # plt.ylabel("fitness")
    # plt.legend()
    # plt.show()

    print("Finished Running Neural Net")
