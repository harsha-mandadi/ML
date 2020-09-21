import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# visualize data
import seaborn as sns

# visualize the relationship between the features and its response using scatterplots
# sns.pairplot(data, x_vars, y_vars)


def dataset_selection(type):
    if type == 1:
        df = pd.read_csv('Dataset/transfusion.data',
                         sep=",", header=0)
        # df.columns = ["Pregnancies", "Glucose", "BloddPressure", "SkinThickness",
        # "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
        df.columns = ["a", "b", "c", "d", "out"]
        X = df.values[:, 0: 4]
        # print(X[1: 11])
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

    train_sizes, train_scores, test_scores = learning_curve(knn,
                                                            X,
                                                            y,
                                                            # Number of folds in cross-validation
                                                            cv=10,
                                                            # Evaluation metric
                                                            scoring=scoringmetric,
                                                            # Use all computer cores
                                                            n_jobs=-1,
                                                            # 50 different sizes of the training set
                                                            # train_sizes=np.linspace(
                                                            #    0.01, 1.0, 25),
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
        knn, X, y, feature, param_range=train_sizes,  cv=5)

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
# Data processing


# Calling DT classifier
# create an instance Kneighbors classifier class
X_train, y_train, X_test, y_test = dataset_selection(0)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# To find best k
# for k in range(1, 25):
#      # create an instance Kneighbors classifier class
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_val)
#     scores.append(accuracy_score(y_test, y_pred))
# tree.plot_tree(clf)

# prediction
y_pred = knn.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# modelcomplex(X_train, y_train, feature="p",
#             paramrange = np.arange(1, 3, 1))
learningcurve(knn, X_train, y_train, "f1")
