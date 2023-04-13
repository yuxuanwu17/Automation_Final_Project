import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from utils import _get_performance, _get_train_test_split, _get_passive_index_split, _active_learning_simulation, \
    result_logging, timeit


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
def rf_passive_learning(num_per_round):
    X, y, observed_idx = _get_passive_index_split()
    accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs, learning_round = _active_learning_simulation(
        X,
        y,
        "rf",
        observed_idx, num_per_round)

    result_logging(accuracy_scores, f"res/rf/rf_acc_res_{num_per_round}.txt")
    result_logging(f1_scores, f"res/rf/rf_f1_res_{num_per_round}.txt")
    result_logging(recall_scores, f"res/rf/rf_recall_res_{num_per_round}.txt")
    result_logging(precision_scores, f"res/rf/rf_precision_res_{num_per_round}.txt")
    result_logging(MCCs, f"res/rf/rf_mcc_res_{num_per_round}.txt")
    result_logging(auROCs, f"res/rf/rf_auROC_res_{num_per_round}.txt")
    result_logging(auPRCs, f"res/rf/rf_auPRC_res_{num_per_round}.txt")
    result_logging(learning_round, f"res/rf/rf_learning_round_res_{num_per_round}.txt")


if __name__ == '__main__':
    rf_machine_learning()
    rf_passive_learning(250)
    rf_passive_learning(500)
    rf_passive_learning(1000)
