import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Data processing


def dataset_selection(type):
    if type == 1:
        df = pd.read_csv('Dataset/transfusion.data',
                         sep=",", header=0)
        # df.columns = ["Pregnancies", "Glucose", "BloddPressure", "SkinThickness",
        # "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
        df.columns = ["a", "b", "c", "d", "out"]
        X = df.values[:, 0: 4]
        #print(X[1: 11])
        y = df.values[:, 4]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, shuffle=True)
        print(len(X))

        print(sum(y))
        # X_train, X_val, y_train, y_val = train_test_split(
        # X_train, y_train, test_size=0.25, random_state=1, shuffle=True)
    else:
        # Checking for missing data
        # NAs = pd.concat([df.isnull().sum()], axis=1, keys=[‘Train’])
        # NAs[NAs.sum(axis=1) > 0]
        df = pd.read_csv('Dataset/data_banknote_authentication.txt',
                         sep=",", header=None)
        df.columns = ["a", "b", "c", "d", "out"]

        X = df.values[:, 0:4]
        y = df.values[:, 4]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, shuffle=True)

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=True)  # 0.25 x 0.8 = 0.2
    return X_train, y_train, X_test, y_test
################################learning curve#######################


def learningcurve(Algorithmclass, Xin, yin, scoringmetric):
    X = Xin
    y = yin

    train_sizes, train_scores, test_scores = learning_curve(clf,
                                                            X,
                                                            y,
                                                            # Number of folds in cross-validation
                                                            cv=10,
                                                            # Evaluation metric
                                                            scoring=scoringmetric,
                                                            # Use all computer cores
                                                            n_jobs=-1,
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(
                                                                0.01, 1.0, 25),
                                                            shuffle=True)

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # # Draw lines
    plt.plot(train_sizes, train_mean, '--',
             color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111",
             label="Cross-validation score")

    # # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, color="#DDDDDD")

    # # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel(
        scoringmetric), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
############################ model complexity curve#########################


def modelcomplex(Xin, yin, feature="", paramrange=[]):
    X = Xin
    y = yin
    train_sizes = paramrange
    train_scores, valid_scores = validation_curve(
        clf, X, y, feature, param_range=train_sizes,  cv=5)

    # # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # # Create means and standard deviations of test set scores
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    # # Draw lines
    plt.plot(train_sizes, train_mean, label="Training score")
    plt.plot(train_sizes, valid_mean, label="Cross-validation score")

    # # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, valid_mean - valid_std,
                     valid_mean + valid_std, color="#DDDDDD")

    # # Create plot
    plt.title("Validation Curve")
    plt.xlabel(feature), plt.ylabel(
        "Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# Calling DT classifier


# class DecisionTree(baselearning):

#    def __init__(dataset):
#        self.dataset = dataset

#    def DTlearner():
X_train, y_train, X_test, y_test = dataset_selection(1)
clf = tree.DecisionTreeClassifier(max_depth=6, min_samples_leaf=12)
clf = clf.fit(X_train, y_train)
# tree.plot_tree(clf)

# prediction
y_pred = clf.predict(X_test)

# Evaluation
accuracy = f1_score(y_test, y_pred)
print(accuracy)
# modelcomplex(X_train, y_train, feature="max_depth",
#             paramrange=np.arange(3, 25, 1))
learningcurve(clf, X_train, y_train, "f1")
# f1 score
# f1score = f1_score(y_test, y_pred)
# AUC ROC
# auc = roc_auc_score(y_test, y_pred)
# ROC -
# roc = roc_curve(y_test, y_pred)
# graph
# plt.figure(figsize=(20, 10))
# tree.plot_tree(clf, filled=True, fontsize=10)
# plt.show()
# plt.close()

# learning curve
# Create CV training and test scores for various training set sizes
# print(len(X))
# print(sum(y))
# ROC curve
