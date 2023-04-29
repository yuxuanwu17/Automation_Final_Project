import time
from functools import wraps

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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import itertools
import sys
import wandb


method = sys.argv[1]

wandb.init(name=method, project="automation", entity="yanjing")

# data format:
df = pd.read_csv("20000_reduced_featureSelectedAllDataWithY.csv")
df.head()


#Query By committee
def query_by_committe(X_observed, y_observed, X_unobserved, num_per_round):
    RF = RandomForestClassifier(n_estimators=20)
    RF.fit(X_observed, y_observed)

    predictions = np.array([tree.predict(X_unobserved) for tree in RF.estimators_])
    predictions_array = np.array([np.array(x) for x in predictions])
    vote_1 = -(np.multiply(np.sum(predictions_array == 1, axis=0)/predictions_array.shape[0], (np.log(np.sum(predictions_array == 1, axis=0) + 1e-10/predictions_array.shape[0]))))
    vote_2 = -(np.multiply(np.sum(predictions_array == 0, axis=0)/predictions_array.shape[0], (np.log(np.sum(predictions_array == 0, axis=0) + 1e-10/predictions_array.shape[0]))))
    return np.argsort(vote_1 + vote_2)[-num_per_round:]
    # nonzero_indices = np.nonzero(vote_1 + vote_2)[0]
    # return np.random.choice(nonzero_indices, num_per_round, replace=False)


#Density Sampling
def density_based_sampling(X_observed, y_observed, X_unobserved, num_per_round, Beta):
    logre = RandomForestClassifier(n_estimators=20) #Todo: change the model

    logre.fit(X_observed, y_observed)

    preds = np.array(logre.predict_proba(X_unobserved))
    phi_x = 1 - np.amax(preds, axis=1)
    density_ls = []

    for i in range(X_unobserved.shape[0]):
        similarity = 0
        cur_x = X_unobserved[i]
        x_prime_array = np.delete(X_unobserved, i, axis=0)
        cluster_size = num_per_round*2
        if cluster_size > x_prime_array.shape[0]:
          cluster_size = x_prime_array.shape[0]
        if len(x_prime_array) >= 500:
            clusters = KMeans(n_clusters=100).fit_predict(x_prime_array)
            samples = [np.random.choice(np.where(clusters == val)[0], size=1, replace=True).tolist() for val in np.unique(clusters)]
            idxs = list(itertools.chain.from_iterable(samples))
        else:
            idxs = list(range(0, len(x_prime_array)))
        for j in idxs:
            similarity += cosine_similarity([cur_x], [x_prime_array[j, :]])[0][0]
        density_ls.append(((similarity / X_unobserved.shape[0]) ** Beta) * phi_x[i])
    return np.argsort(np.array(density_ls))[-num_per_round:]


#Minimizing the expected risk
def min_expected_risk(X_observed, y_observed, X_unobserved, num_per_round):
    model = RandomForestClassifier(n_estimators=10)

    model.fit(X_observed, y_observed)

    preds = np.array(model.predict_proba(X_unobserved))
    risk_ls = []
    for i in range(preds.shape[0]):
        #subset of u excluding x
        x_prime = np.delete(X_unobserved, i, axis=0)

        #add current x to observed data with label 1, and fit the data to the model
        model.fit(np.append(X_observed, X_unobserved[i].reshape(1, -1), axis=0), np.append(y_observed,1))
        preds_1 = np.array(model.predict_proba(x_prime))

        model.fit(np.append(X_observed, X_unobserved[i].reshape(1, -1), axis=0), np.append(y_observed,2))
        preds_2 = np.array(model.predict_proba(x_prime))

        cur_risk = np.sum(np.multiply(preds[i, :], np.array([sum(1 - np.max(preds_1,axis=1)), sum(1 - np.max(preds_2,axis=1))])))
        risk_ls.append(cur_risk)
    return np.argsort(np.array(risk_ls))[:num_per_round]



#diversity sampling
def diversity_sampling(prob2_data, num_per_round):
  if len(prob2_data) >= num_per_round:
    clusters = KMeans(n_clusters=num_per_round).fit_predict(prob2_data)
    idxs = [np.random.choice(np.where(clusters == val)[0], size=1, replace=False).tolist() for val in np.unique(clusters)]
    return list(itertools.chain.from_iterable(idxs))
  else:
    return list(range(0, len(prob2_data)))


def _get_train_test_split():
    np.random.seed(33)
    df = pd.read_csv("20000_reduced_featureSelectedAllDataWithY.csv")
    #print(df.shape)

    training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25, shuffle=True)

    y_train = training_data['disposition']
    y_test = testing_data['disposition']
    X_train = StandardScaler().fit_transform(training_data.drop("disposition", axis=1))
    X_test = StandardScaler().fit_transform(testing_data.drop("disposition", axis=1))
    return X_train, y_train, X_test, y_test


def _get_passive_index_split(init_observed=2000):
    np.random.seed(33)
    df = pd.read_csv("20000_reduced_featureSelectedAllDataWithY.csv")
    # print(df.shape)
    y = df['disposition'].values
    X = StandardScaler().fit_transform(df.drop("disposition", axis=1))
    # print("X's type", type(X))
    # print("y's type", type(y))
    total_num = len(X)
    #print("total number for init", total_num)
    observed_idx = []
    with open("initial_index.txt") as f:
        lines = f.readlines()
        for i in lines:
            observed_idx.append(int(i.strip("\n")))
        f.close()
    #print("init number for observed index", len(observed_idx))
    return X, y, observed_idx


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


def _append_5value_performance(clf, X_test, y_test, y_pred, accuracy_scores, f1_scores, recall_scores, precision_scores,
                               MCCs, auROCs, auPRCs):
    accuracy_scores.append(np.round(accuracy_score(y_true=y_test, y_pred=y_pred), 4))
    f1_scores.append(np.round(f1_score(y_true=y_test, y_pred=y_pred), 4))
    recall_scores.append(np.round(recall_score(y_true=y_test, y_pred=y_pred), 4))
    precision_scores.append(np.round(precision_score(y_true=y_test, y_pred=y_pred), 4))
    MCCs.append(np.round(matthews_corrcoef(y_true=y_test, y_pred=y_pred), 4))
    auROCs.append(np.round(roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1]), 4))
    auPRCs.append(np.round(average_precision_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 0]), 4))
    return accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs


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
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    MCCs = []
    auROCs = []
    auPRCs = []

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
    # change this to the standard 5 values evaluation
    y_pred = model.predict(X_unobserved)
    accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs = _append_5value_performance(
        model, X_unobserved, y_unobserved, y_pred, accuracy_scores, f1_scores,
        recall_scores, precision_scores,
        MCCs, auROCs, auPRCs)
    return accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs, model, X_unobserved, \
        y_unobserved, X_observed, y_observed, unobserved_idx


def _active_learning_simulation(X, y, criterion, observed_idx, num_per_round=int(sys.argv[2]), method="random sampling", Beta=1):
    """
    Your simulation should start with twenty (20) random observations
    Stopping criterion is different, in this case, stop until you have no observation remaining

    base learner: logistic regression

    :param X:
    :param y:
    :return: a plot showing the average and standard deviation of the 5-fold cross-validation accuracy
    as a function of the number of instances in the training data set.

    """
    print("current chosen query method is %s" % method)
    accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs, model, X_unobserved, y_unobserved, X_observed, y_observed, unobserved_idx = _init_fixed_20_samples_lr(
        X, y, observed_idx, criterion)

    learning_round = [len(observed_idx)]
    # key component in adjusting the criterion used
    most_uncertain_idxs = np.random.choice(len(X_unobserved), num_per_round, replace=False)
    new_observed_idxs = [unobserved_idx[i] for i in most_uncertain_idxs]

    X_observed = np.vstack((X_observed, X[new_observed_idxs, :]))
    y_observed = np.append(y_observed, y[new_observed_idxs])

    # Remove the selected data point from the unobserved set
    unobserved_idx = np.setdiff1d(unobserved_idx, new_observed_idxs)
    X_unobserved = X[unobserved_idx, :]
    y_unobserved = y[unobserved_idx]
    # 100 should be tunable
    num = 0
    while len(X_observed) <= len(X):
        wandb.log({"round": num})
        idx = len(learning_round) - 1
        # print(accuracy_scores)
        print(
            f"learning_round:{len(X_observed)}, acc: {accuracy_scores[idx]}, f1: {f1_scores[idx]},"
            f"recall: {recall_scores[idx]}, precision: {precision_scores[idx]}, mcc: {MCCs[idx]}, auROC: {auROCs[idx]}, "
            f"auPRC: {auPRCs[idx]}")
        learning_round.append(len(X_observed))
        # Train a random forest classifier on the observed data
        model = _get_lr() if criterion == "lr" else _get_rf()
        model.fit(X_observed, y_observed)

        if len(X_observed) + num_per_round >= len(X):
            y_pred = model.predict(X_unobserved)
            accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs = _append_5value_performance(
                model, X_unobserved, y_unobserved, y_pred, accuracy_scores, f1_scores,
                recall_scores, precision_scores,
                MCCs, auROCs, auPRCs)
            break

        # key component in adjusting the criterion used
        # passive learning methods
        ################ active learning query based methods inserted HERE!##########################
        if method == "random_sampling":
          most_uncertain_idxs = np.random.choice(len(X_unobserved), num_per_round, replace=False) # passive learning
        elif method == "query_by_committee":
          most_uncertain_idxs = query_by_committe(X_observed, y_observed, X_unobserved, num_per_round)
        elif method == "density_based_sampling":
          most_uncertain_idxs = density_based_sampling(X_observed, y_observed, X_unobserved, num_per_round, Beta)
        elif method == "min_expected_risk":
          most_uncertain_idxs = min_expected_risk(X_observed, y_observed, X_unobserved, num_per_round)
        else:
          most_uncertain_idxs = diversity_sampling(X_unobserved, num_per_round)
        ################ active learning query based methods inserted HERE!##########################

        new_observed_idxs = [unobserved_idx[i] for i in most_uncertain_idxs]

        X_observed = np.vstack((X_observed, X[new_observed_idxs, :]))
        y_observed = np.append(y_observed, y[new_observed_idxs])

        # Remove the selected data point from the unobserved set
        unobserved_idx = np.setdiff1d(unobserved_idx, new_observed_idxs)
        X_unobserved = X[unobserved_idx, :]
        y_unobserved = y[unobserved_idx]
        # unobserved_accuracy_scores.append(np.mean(cross_val_score(model, X_unobserved, y_unobserved)))
        y_pred = model.predict(X_unobserved)
        accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs = _append_5value_performance(
            model, X_unobserved, y_unobserved, y_pred, accuracy_scores, f1_scores,
            recall_scores, precision_scores,
            MCCs, auROCs, auPRCs)
        
        num += 1
    return accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs, learning_round


def result_logging(val, name):
    with open(name, 'w') as f:
        for item in val:
            f.write("%s\n" % item)


def timeit(func):
    """
    decorator used to evaluate the time consumed in each simulation
    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # start the timer
        result = func(*args, **kwargs)
        end_time = time.time()  # stop the timer
        elapsed_minutes = (end_time - start_time) / 60  # calculate the elapsed time in minutes
        print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
        return result

    return wrapper





import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# from utils import _get_performance, _get_train_test_split, _get_passive_index_split, _active_learning_simulation, \
#     result_logging, timeit


def rf_machine_learning():
    X_train, y_train, X_test, y_test = _get_train_test_split()
    bootstrap = True
    max_depth = 80
    max_features = 2
    min_samples_leaf = 3
    min_samples_split = 8
    n_estimators = 1000

    clf = RandomForestClassifier(random_state=0, class_weight="balanced", bootstrap=bootstrap, max_depth=max_depth,
                                 max_features=max_features, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split,
                                 n_estimators=n_estimators).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    _get_performance(clf, X_test, y_test, y_pred)


@timeit
def rf_passive_learning(method="random sampling", Beta=1):
    X, y, observed_idx = _get_passive_index_split()
    accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs, learning_round = _active_learning_simulation(
        X,
        y,
        "rf",
        observed_idx,
        method=method,
        Beta=Beta)

    result_logging(accuracy_scores, f"{method}_{sys.argv[2]}_rf_acc_res.txt")
    result_logging(f1_scores, f"{method}_{sys.argv[2]}_rf_f1_res.txt")
    result_logging(recall_scores, f"{method}_{sys.argv[2]}_rf_recall_res.txt")
    result_logging(precision_scores, f"{method}_{sys.argv[2]}_rf_precision_res.txt")
    result_logging(MCCs, f"{method}_{sys.argv[2]}_rf_mcc_res.txt")
    result_logging(auROCs, f"{method}_{sys.argv[2]}_rf_auROC_res.txt")
    result_logging(auPRCs, f"{method}_{sys.argv[2]}_rf_auPRC_res.txt")
    result_logging(learning_round, f"{method}_{sys.argv[2]}_rf_learning_round_res.txt")


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# from utils import _get_performance, _get_train_test_split, _get_passive_index_split, _active_learning_simulation, \
#     result_logging, timeit


def lr_machine_learning():
    X_train, y_train, X_test, y_test = _get_train_test_split()
    c = 0.01
    penalty = 'l2'
    solver = 'saga'

    clf = LogisticRegression(random_state=0, class_weight="balanced", C=c, penalty=penalty, solver=solver).fit(X_train,
                                                                                                               y_train)

    y_pred = clf.predict(X_test)

    _get_performance(clf, X_test, y_test, y_pred)


@timeit
def lr_passive_learning(method="random sampling"):
    X, y, observed_idx = _get_passive_index_split()
    accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs, learning_round = _active_learning_simulation(
        X,
        y,
        "lr",
        observed_idx,
        method=method)

    result_logging(accuracy_scores, f"{method}_{sys.argv[2]}_lr_acc_res.txt")
    result_logging(f1_scores, f"{method}_{sys.argv[2]}_lr_f1_res.txt")
    result_logging(recall_scores, f"{method}_{sys.argv[2]}_lr_recall_res.txt")
    result_logging(precision_scores, f"{method}_{sys.argv[2]}_lr_precision_res.txt")
    result_logging(MCCs, f"{method}_{sys.argv[2]}_lr_mcc_res.txt")
    result_logging(auROCs, f"{method}_{sys.argv[2]}_lr_auROC_res.txt")
    result_logging(auPRCs, f"{method}_{sys.argv[2]}_lr_auPRC_res.txt")
    result_logging(learning_round, f"{method}_{sys.argv[2]}_lr_learning_round_res.txt")




if method == "rf_committee":
    rf_passive_learning(method="query_by_committee")

if method == "rf_diversity":
    rf_passive_learning(method="diversity")

if method == "rf_min_expected_risk":
    rf_passive_learning(method="min_expected_risk")

if method == "rf_density_based_sampling":
    rf_passive_learning(method="density_based_sampling")

if method == "lr_committee":
    lr_passive_learning(method="query_by_committee")

if method == "lr_diversity":
    lr_passive_learning(method="diversity")

if method == "lr_min_expected_risk":
    lr_passive_learning(method="min_expected_risk")

if method == "lr_density_based_sampling":
    lr_passive_learning(method="density_based_sampling")