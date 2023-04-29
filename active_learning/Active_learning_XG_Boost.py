import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score
from prettytable import PrettyTable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
import random
import warnings
import itertools
from functools import wraps
import argparse
from collections import Counter
warnings.filterwarnings("ignore")

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

"""
active learning"""

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

#Density Sampling with no diversity
def density_based_sampling_only(X_observed, y_observed, X_unobserved, num_per_round, Beta):
    logre = RandomForestClassifier(n_estimators=20) #Todo: change the model

    logre.fit(X_observed, y_observed)

    preds = np.array(logre.predict_proba(X_unobserved))
    phi_x = 1 - np.amax(preds, axis=1)
    density_ls = []

    for i in range(X_unobserved.shape[0]):
        similarity = 0
        cur_x = X_unobserved[i]
        x_prime_array = np.delete(X_unobserved, i, axis=0)
        for j in range(x_prime_array.shape[0]):
            similarity += cosine_similarity([cur_x], [x_prime_array[j, :]])[0][0]
        density_ls.append(((similarity / X_unobserved.shape[0]) ** Beta) * phi_x[i])
    return np.argsort(np.array(density_ls))[-num_per_round:]

#Density Sampling with diversity
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
        # cluster_size = num_per_round*2
        # if cluster_size > x_prime_array.shape[0]:
        #   cluster_size = x_prime_array.shape[0]
        idxs = diversity_sampling(x_prime_array, 1000)
        for j in idxs:
            similarity += cosine_similarity([cur_x], [x_prime_array[j, :]])[0][0]
        density_ls.append(((similarity / X_unobserved.shape[0]) ** Beta) * phi_x[i])
    return np.argsort(np.array(density_ls))[-num_per_round:]

#diversity sampling
def diversity_sampling(prob2_data, num_per_round):
  if len(prob2_data) >= num_per_round:
    clusters = KMeans(n_clusters=num_per_round).fit_predict(prob2_data)
    idxs = [np.random.choice(np.where(clusters == val)[0], size=1, replace=False).tolist() for val in np.unique(clusters)]
    return list(itertools.chain.from_iterable(idxs))
  else:
    return list(range(0, len(prob2_data)))

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
    observed_idx = []
    with open("initial_index.txt") as f:
        lines = f.readlines()
        for i in lines:
            observed_idx.append(int(i.strip("\n")))
        f.close()
    #print("init number for observed index", len(observed_idx))
    return observed_idx

"""machine learning model"""


# construct the XGBoost model
@timeit
def XGBoostModel(data, method="one-time", initial=2000, once_add=1000, query_method="random", Beta=1):
    # split the data into training and testing set with 2:8 ratio
    # and evaluate its performance
    dis = data["disposition"]
    counter = Counter(dis)
    estimate = counter[1]/counter[2]
    if method == "one-time":
        X_train, y_train, X_test, y_test = _get_train_test_split()
        # jump through grid search
        clf = XGBClassifier(n_estimators=10, max_depth=8,
                            learning_rate=0.3, n_jobs=-1, random_state=1, scale_pos_weight=estimate,
                            use_label_encoder=True).fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # evaluate the model performance
        accuracy_scores = []
        f1_scores = []
        recall_scores = []
        precision_scores = []
        MCCs = []
        auROCs = []
        auPRCs = []

        # calculate the metrices
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
        return table
    elif method == "iterative":
        print("current method", query_method)
        # initialization, randomly select
        initial_index = _get_passive_index_split()
        remain_index = [x for x in range(0, len(data)) if x not in initial_index]
        current_dat = data.iloc[initial_index]
        remain_dat = data.iloc[remain_index]

        # set training and testing data
        y_train = current_dat['disposition']
        y_test = remain_dat['disposition']
        X_train = StandardScaler().fit_transform(current_dat.drop("disposition", axis=1))
        X_test = StandardScaler().fit_transform(remain_dat.drop("disposition", axis=1))

        # evaluate the model performance
        sampleNum = []
        accuracy_scores = []
        f1_scores = []
        recall_scores = []
        precision_scores = []
        MCCs = []
        auROCs = []
        auPRCs = []

        # construct model
        ## jump through grid search
        model = XGBClassifier(n_estimators=10, max_depth=8,
                              learning_rate=0.3, n_jobs=-1, random_state=1, scale_pos_weight=estimate,
                              use_label_encoder=True)
        clf = model.fit(X_train, y_train)
        # add 10000 new samples each time
        # calculate the number of round we need to run in total
        numRound = ((len(data) - initial) // once_add) + 1
        for i in range(numRound + 1):
            print("current round", i)
            # only continue training when there is unobserved sample
            # if len(remain_index) > 0:
            # obtain performance
            y_pred = clf.predict(X_test)
            sampleNum.append(initial + once_add * i)
            accuracy_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred))
            f1_scores.append(f1_score(y_true=y_test, y_pred=y_pred))
            recall_scores.append(recall_score(y_true=y_test, y_pred=y_pred))
            precision_scores.append(precision_score(y_true=y_test, y_pred=y_pred))
            MCCs.append(matthews_corrcoef(y_true=y_test, y_pred=y_pred))
            auROCs.append(roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1]))
            auPRCs.append(average_precision_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 0]))
            # add certain number of samples each time
            if query_method == "random":
                if once_add <= len(remain_index):
                    new_index = random.sample(remain_index, once_add)
                # if the number of remaining sample is less than once_add, end the for loop
                else:
                    break
            #################### active learning ####################
            elif query_method == "query_by_committee":
                if once_add <= len(remain_index):
                    new_index = query_by_committe(X_train, y_train, X_test, once_add)
                # if the number of remaining sample is less than once_add, end the for loop
                else:
                    break
            elif query_method == "density_based_sampling":
                if once_add <= len(remain_index):
                    new_index = density_based_sampling(X_train, y_train, X_test, once_add, Beta)
                # if the number of remaining sample is less than once_add, end the for loop
                else:
                    break
            elif query_method == "min_expected_risk":
                if once_add <= len(remain_index):
                    new_index = min_expected_risk(X_train, y_train, X_test, once_add)
                # if the number of remaining sample is less than once_add, end the for loop
                else:
                    break
            else:
                if once_add <= len(remain_index):
                    new_index = diversity_sampling(X_test, once_add)
                # if the number of remaining sample is less than once_add, end the for loop
                else:
                    break
            #################### active learning ####################
            remain_index = [i for i in remain_index if i not in new_index]
            new_dat = data.iloc[new_index]
            current_dat = pd.concat([current_dat, new_dat])
            remain_dat = data.iloc[remain_index]

            # set training and testing data
            y_train = current_dat['disposition']
            y_test = remain_dat['disposition']
            X_train = StandardScaler().fit_transform(current_dat.drop("disposition", axis=1))
            X_test = StandardScaler().fit_transform(remain_dat.drop("disposition", axis=1))

            clf = model.fit(X_train, y_train)
        # print performance table
        table = PrettyTable()
        column_names = ['sampleNum', 'Accuracy', 'auROC', 'auPRC', 'recall', 'precision', 'f1', 'MCC']
        table.add_column(column_names[0], sampleNum)
        table.add_column(column_names[1], np.round(accuracy_scores, 4))
        table.add_column(column_names[2], np.round(auROCs, 4))
        table.add_column(column_names[3], np.round(auPRCs, 4))
        table.add_column(column_names[4], np.round(recall_scores, 4))
        table.add_column(column_names[5], np.round(precision_scores, 4))
        table.add_column(column_names[6], np.round(f1_scores, 4))
        table.add_column(column_names[7], np.round(MCCs, 4))
        print(table)
        with open(f'results/XG_Boost_{query_method}_initial_{initial}_batchsize_{once_add}_Beta{Beta}_estimator10_maxdepth8.csv', 'w',
                  newline='') as f_output:
            f_output.write(table.get_csv_string())
        return table

def main():
    parser = argparse.ArgumentParser(description='This is a test argument parser.')
    parser.add_argument('--once_add', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--query_method', type=str, default='random',
                        help='query_method')
    parser.add_argument('--method', type=str, default="iterative",
                        help='method')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    results = main()
    data = pd.read_csv("20000_reduced_featureSelectedAllDataWithY.csv")
    XGBoostModel(data=data, method=results.method, query_method=results.query_method, once_add=results.once_add)