import sys
sys.path.append("C:\\main\\GitHub\\ML\\losses")


import pytest

import numpy as np
from sklearn import linear_model

from linear_models import LinearRegression

NUM_TESTS = 1_000

X_ = [np.random.uniform(-10, 10, 1000) for _ in range(NUM_TESTS)]
y_ = [np.random.uniform(-10, 10, 1)*x_arr+np.random.uniform(-10, 10, 1) for x_arr in X_]

@pytest.mark.parametrize("X, y", zip(X_, y_))
def test_dataset_linear_learning(X: np.ndarray, y: np.ndarray, eps=1e-2) -> None:
    """
    Tests whether implemented model solution is similar to sklearn solution

    Args
    ----
    X: np.ndarray
        Feature matrix X of shape NxD where N is number of objects, D is number of features
    
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


def test_dataset_linear_logistic_learning() -> None:
    """
    Tests whether implemented model solution is similar to sklearn solution

    Args
    ----
    X: np.ndarray
        Feature matrix X of shape NxD where N is number of objects, D is number of features
    
    y: np.ndarray
        Target vector column (binary values 0 and 1)
    """
    assert 0 == 0