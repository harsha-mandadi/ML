from mlrose import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def modelcomplex(Xin, yin, feature="", paramrange=[]):
    X = Xin
    y = yin
    train_sizes = paramrange
    train_scores, valid_scores = validation_curve(
        nn_model, X, y, feature, param_range=train_sizes,  cv=5)

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

################################learning curve#######################


def learningcurve(Algorithmclass, Xin, yin, scoringmetric):
    X = Xin
    y = yin

    train_sizes, train_scores, test_scores = learning_curve(nn_model,
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


#############NN classifier###########
df = pd.read_csv('Data/data_banknote_authentication.txt',
                 sep=",", header=None)
df.columns = ["a", "b", "c", "d", "out"]

X = df.values[:, 0:4]
y = df.values[:, 4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, shuffle=True)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
                                algorithm='gradient_descent', max_iters=1000,
                                bias=True, is_classifier=True, learning_rate=0.0001,
                                early_stopping=True, clip_max=5, max_attempts=100,
                                random_state=3)

nn_model.fit(X_train, y_train)
# prediction
y_pred = nn_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# modelcomplex(X_train, y_train, feature="learning_rate",
# paramrange = np.arange(0.00001, 0.1, 0.1))
learningcurve(nn_model, X_train, y_train, "accuracy")

############################ model complexity curve#########################
