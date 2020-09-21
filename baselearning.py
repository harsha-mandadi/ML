import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt


class baselearning:

    def __init__():

    def preprocess_data(dataset):
        if dataset == 1:
            df = pd.read_csv('Dataset/data_banknote_authentication.txt',
                             sep=",", header=None)
        # df = pd.read_csv('Dataset/data_banknote_authentication.txt',
        #                 sep=",", header=None)

    # Checking for missing data
    # NAs = pd.concat([df.isnull().sum()], axis=1, keys=[‘Train’])
    # NAs[NAs.sum(axis=1) > 0]
        df.columns = ["a", "b", "c", "d", "out"]

        X = df.values[:, 0:4]
        y = df.values[:, 4]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, shuffle=True)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=1, shuffle=True)

    def learningcurve(train_scores, test_scores):
        train_mean = np.mean(train_scores, axis=1)

        train_std = np.std(train_scores, axis=1)

        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Draw lines
        plt.plot(train_sizes, train_mean, '--',
                 color="#111111",  label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111",
                 label="Cross-validation score")

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, color="#DDDDDD")

        # Create plot
        curve_type == "Learning Curve":
        plt.title("Learning Curve")
        plt.xlabel("Training Set Size"), plt.ylabel(
            "Accuracy Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def validationcurve(train_scores, test_scores):
        train_mean = np.mean(train_scores, axis=1)

        train_std = np.std(train_scores, axis=1)

        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Draw lines
        plt.plot(train_sizes, train_mean, '--',
                 color="#111111",  label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111",
                 label="Cross-validation score")

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, color="#DDDDDD")
        # plot
        plt.title("Validation Curve")
        plt.xlabel("max_depth"), plt.ylabel("Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
