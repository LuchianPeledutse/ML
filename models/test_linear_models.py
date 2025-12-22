import sys
sys.path.append("C:\\main\\GitHub\\ML\\losses")


import pytest

import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

from linear_models import LinearRegression, LogisticRegression

NUM_TESTS = 10
NUM_SAMPLES = 1_000
NUM_FEATURES = 2

X_ = [np.random.uniform(-10, 10, 1000) for _ in range(NUM_TESTS)]
y_ = [np.random.uniform(-10, 10, 1)*x_arr+np.random.uniform(-10, 10, 1) for x_arr in X_]

@pytest.mark.parametrize("X, y", zip(X_, y_))
def test_dataset_linear_learning(X: np.ndarray, y: np.ndarray, eps=1e-2) -> None:
    """
    Tests whether implemented model solution is similar to sklearn solution

    Args
    ----
    X: np.ndarray
        Feature matrix X of shape Nx1 where N is number of objects, 1 is number of features
    
    y: np.ndarray
        Target vector column
    """
    lib_mod = LinearRegression(epoch=1000, alpha=0.005)
    sk_mod = linear_model.LinearRegression()

    lib_mod.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    sk_mod.fit(X.reshape(-1, 1), y.reshape(-1, 1))

    lib_weight = lib_mod.weight # shape is 3x1 where first row is bias
    sk_weight = np.vstack((sk_mod.intercept_.reshape(-1, 1), sk_mod.coef_))

    # Check wether the difference is less than epsilon
    diff = abs(lib_weight - sk_weight)
    assert (diff < eps).any(), "Some of coefficients difference is larger than epsilon"


log_reg_testing_set = [make_classification(
    n_samples=1000,          
    n_features=2,           
    n_informative=2,         
    n_redundant=0,           
    n_clusters_per_class=1 
) for _ in range(NUM_TESTS)]


@pytest.mark.parametrize("X,y", log_reg_testing_set)
def test_dataset_linear_logistic_learning(X: np.ndarray, y: np.ndarray, eps: float = 7e-2) -> None:
    """
    Tests whether implemented model solution is similar to sklearn solution

    Args
    ----
    X: np.ndarray
        Feature matrix X of shape NxD where N is number of objects, D is number of features
    
    y: np.ndarray
        Target vector column (binary values 0 and 1)
    
    eps: float
        Epsilon up to which the accuracies have to be the same
    """
    # Preparing models
    my_logreg = LogisticRegression(epoch=100_000, alpha=7e-1)
    sk_logreg = linear_model.LogisticRegression()
    # Fitting them
    my_logreg.fit(X, y.reshape(-1, 1))
    sk_logreg.fit(X, y)
    # Comparing accuracies
    my_acc = accuracy_score(y_pred=my_logreg.predict(X).reshape(-1), y_true=y)
    sk_acc = accuracy_score(y_pred=sk_logreg.predict(X), y_true=y)
    assert abs(my_acc-sk_acc) < eps, "The differences between accuracies is larger than epsilon"


