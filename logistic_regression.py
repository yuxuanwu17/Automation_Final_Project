from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

from utils import _get_performance, _get_train_test_split, _get_passive_index_split, _active_learning_simulation, \
    result_logging, timeit


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
def lr_passive_learning():
    X, y, observed_idx = _get_passive_index_split()
    accuracy_scores, f1_scores, recall_scores, precision_scores, MCCs, auROCs, auPRCs, learning_round = _active_learning_simulation(
        X,
        y,
        "lr",
        observed_idx)

    result_logging(accuracy_scores, "res/lr/lr_acc_res.txt")
    result_logging(f1_scores, "res/lr/lr_f1_res.txt")
    result_logging(recall_scores, "res/lr/lr_recall_res.txt")
    result_logging(precision_scores, "res/lr/lr_precision_res.txt")
    result_logging(MCCs, "res/lr/lr_mcc_res.txt")
    result_logging(auROCs, "res/lr/lr_auROC_res.txt")
    result_logging(auPRCs, "res/lr/lr_auPRC_res.txt")
    result_logging(learning_round, "res/lr/lr_learning_round_res.txt")


if __name__ == '__main__':
    # lr_machine_learning()
    lr_passive_learning()
