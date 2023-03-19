from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings

from utils import _get_performance, _get_train_test_split

warnings.filterwarnings("ignore")

X_train, y_train, X_test, y_test = _get_train_test_split()
c = 0.01
penalty = 'l2'
solver = 'saga'

clf = LogisticRegression(random_state=0, class_weight="balanced", C=c, penalty=penalty, solver=solver).fit(X_train,
                                                                                                           y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

_get_performance(clf, X_test, y_test, y_pred)

print(cm)
print(ac)
