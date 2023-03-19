import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


def _get_train_test_split():
    np.random.seed(33)
    df = pd.read_csv("./data/featureSelectedAllDataWithY.csv")
    print(df.shape)

    training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25, shuffle=True)

    y_train = training_data['disposition']
    y_test = testing_data['disposition']
    X_train = StandardScaler().fit_transform(training_data.drop("disposition", axis=1))
    X_test = StandardScaler().fit_transform(testing_data.drop("disposition", axis=1))
    return X_train, y_train, X_test, y_test


def _get_performance(clf, X_test, y_test, y_pred):
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    MCCs = []
    auROCs = []
    auPRCs = []

    accuracy_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred))
    f1_scores.append(f1_score(y_true=y_test, y_pred=y_pred))
    recall_scores.append(recall_score(y_true=y_test, y_pred=y_pred))
    precision_scores.append(precision_score(y_true=y_test, y_pred=y_pred))
    MCCs.append(matthews_corrcoef(y_true=y_test, y_pred=y_pred))
    auROCs.append(roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1]))
    auPRCs.append(average_precision_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 0]))

    table = PrettyTable()
    column_names = ['Accuracy', 'auROC', 'auPRC', 'recall', 'precision', 'f1', 'MCC']
    table.add_column(column_names[0], np.round(accuracy_scores, 4))
    table.add_column(column_names[1], np.round(auROCs, 4))
    table.add_column(column_names[2], np.round(auPRCs, 4))
    table.add_column(column_names[3], np.round(recall_scores, 4))
    table.add_column(column_names[4], np.round(precision_scores, 4))
    table.add_column(column_names[5], np.round(f1_scores, 4))
    table.add_column(column_names[6], np.round(MCCs, 4))
    print(confusion_matrix(y_test, y_pred))
    print(table)


def _get_lr():
    c = 0.01
    penalty = 'l2'
    solver = 'saga'
    return LogisticRegression(random_state=0, class_weight="balanced", C=c, penalty=penalty, solver=solver)


def _get_rf():
    bootstrap = True
    max_depth = 80
    max_features = 2
    min_samples_leaf = 3
    min_samples_split = 8
    n_estimators = 1000

    return RandomForestClassifier(random_state=0, class_weight="balanced", bootstrap=bootstrap, max_depth=max_depth,
                                  max_features=max_features, min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  n_estimators=n_estimators)


def _init_fixed_20_samples_lr(X, y, observed_idx, criterion):
    observed_accuracy_scores = []
    data_num = len(X)

    # 20 is the number specified as the start number of observation
    unobserved_idx = list(set(range(data_num)) - set(observed_idx))
    random.shuffle(unobserved_idx)

    X_observed, y_observed = X[observed_idx, :], y[observed_idx]
    X_unobserved, y_unobserved = X[unobserved_idx, :], y[unobserved_idx]

    # Train a logistic regression as the base learner
    model = _get_lr() if criterion == "lr" else _get_rf()
    model.fit(X_observed, y_observed)

    # Get the accuracy for observed and unobserved
    observed_accuracy = np.mean(cross_val_score(model, X_observed, y_observed))
    observed_accuracy_scores.append(observed_accuracy)
    return observed_accuracy_scores, model, X_unobserved, y_unobserved, X_observed, y_observed, unobserved_idx


def _active_learning_simulation(X, y, criterion, observed_idx):
    """
    Your simulation should start with twenty (20) random observations
    Stopping criterion is different, in this case, stop until you have no observation remaining

    base learner: logistic regression

    :param X:
    :param y:
    :return: a plot showing the average and standard deviation of the 5-fold cross-validation accuracy
    as a function of the number of instances in the training data set.

    """
    X = X.values
    y = y.values

    observed_res = []
    observed_accuracy_scores, model, X_unobserved, y_unobserved, X_observed, y_observed, unobserved_idx = _init_fixed_20_samples_lr(
        X, y, observed_idx, criterion)

    # key component in adjusting the criterion used
    most_uncertain_idx = np.random.choice(len(X_unobserved), 1, replace=False)[0]
    new_observed_idx = unobserved_idx[most_uncertain_idx]

    X_observed = np.vstack((X_observed, X[new_observed_idx, :]))
    y_observed = np.append(y_observed, y[new_observed_idx])

    # Remove the selected data point from the unobserved set
    unobserved_idx = np.delete(unobserved_idx, most_uncertain_idx)
    X_unobserved = X[unobserved_idx, :]

    # 100 should be tunable
    while len(X_observed) <= 100:
        # Train a random forest classifier on the observed data
        model = _get_lr() if criterion == "lr" else _get_rf()
        model.fit(X_observed, y_observed)

        if len(X_observed) == 100:
            observed_accuracy_scores.append(np.mean(cross_val_score(model, X_observed, y_observed)))
            break

        # key component in adjusting the criterion used
        # passive learning methods
        most_uncertain_idx = np.random.choice(len(X_unobserved), 1, replace=False)[0]

        new_observed_idx = unobserved_idx[most_uncertain_idx]

        X_observed = np.vstack((X_observed, X[new_observed_idx, :]))
        y_observed = np.append(y_observed, y[new_observed_idx])

        # Remove the selected data point from the unobserved set
        unobserved_idx = np.delete(unobserved_idx, most_uncertain_idx)
        X_unobserved = X[unobserved_idx, :]
        y_unobserved = y[unobserved_idx]

        observed_accuracy_scores.append(np.mean(cross_val_score(model, X_observed, y_observed)))
    print(len(X_unobserved))

    observed_res.append(observed_accuracy_scores)
    # rounds = len(train_accuracy_scores) + 1
    print(len(observed_res))
    print(observed_res)
    print("len observed_accuracy_scores", len(observed_res[0]))
