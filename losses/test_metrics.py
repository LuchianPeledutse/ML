import math
import torch
import numpy as np
import random as rd


import pytest
from linear_mse import MSE
from linear_log_loss import LogLoss

VECTOR_SIZE = 53
NUM_TESTS = 1000
NUM_FEATURES = 10
NUM_SAMPLES = 1000
rng = np.random.default_rng()

# Testing mse function output
@pytest.mark.parametrize(
        "x,y",
        [(rng.random((VECTOR_SIZE, 1)), rng.random((VECTOR_SIZE, 1))) for _ in range(NUM_TESTS)])
def test_mse_func(x: np.ndarray, y: np.ndarray) -> None:
    """
    Tests mse implemented function (formula) on two vectors

    Args
    ----
    x: np.ndarray
        first vector of shape Nx1
    
    y: np.ndarray
        second vector of shape Nx1
    """
    # Prepare mse function
    lib_mse = MSE()
    torch_mse = torch.nn.MSELoss()
    # Compare values of calculations rounded to 5
    lib_value = round(lib_mse(x, y), 5)
    torch_value = round(torch_mse(torch.from_numpy(x), torch.from_numpy(y)).item(), 5)
    assert lib_value == torch_value, "Implemented MSE values does not equal to torch MSE value"


# Testing mse gradient output
@pytest.mark.parametrize(
        "w,X,y",
        [(rng.random((NUM_FEATURES, 1)),
          rng.random((NUM_SAMPLES, NUM_FEATURES)), 
          rng.random((NUM_SAMPLES, 1))) for _ in range(NUM_TESTS)]
)
def test_mse_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> None:
    """
    Tests mse gradient implementation for linear function
    
    Args
    ----
    w: np.ndarray
        weight vector of shape Dx1 where D is number of features
    
    X: np.ndarray
        Feature matrix of shape NxD where D is number of features; N is number of samples

    
    y: np.ndarray
        Target vector of shape Nx1 
    """
    # Prepare mse functions
    lib_mse = MSE()
    torch_mse = torch.nn.MSELoss()
    # Setting torch tensors with gradients for comparison
    w_torch = torch.tensor(w, requires_grad = True)
    X_torch = torch.tensor(X, requires_grad = True)
    y_torch = torch.tensor(y, requires_grad = True)
    # Calculating loss and grad of linear regression
    y_pred = X_torch@w_torch
    loss = torch_mse(y_pred, y_torch) 
    loss.backward()
    # Comparing results
    torch_grad = np.round(w_torch.grad.detach().numpy(), 4)
    lib_grad = np.round(lib_mse.grad(w, X, y), 4)
    assert (torch_grad-lib_grad).sum().item() == 0, "Implemented gradient does not equal to torch calculated gradient"


# Testing log loss function value 
@pytest.mark.parametrize(
        'y_pred,y',
        [(np.random.uniform(1e-20, 1, NUM_SAMPLES), np.random.randint(0, 2, NUM_SAMPLES)) for _ in range(NUM_TESTS)])
def test_log_function(y_pred: np.ndarray, y: np.ndarray, eps=1e-4) -> None:
    """
    Test log_loss outputs for probability vector and label vector

    Args
    ----
    y_pred: np.ndarray
        Numpy array (probabilities) of shape (N,) where N is number of samples
    
    y: np.ndarray
        Numpy array (binary labels 0 and 1) of shape (N, ) where N is number of samples 
    """
    # Python calculation
    list_to_sum = []
    for prob, label in zip(y_pred, y):
        prob = prob
        list_to_sum.append(label*math.log(prob)+(1-label)*math.log(1-prob))
    iter_value = -sum(list_to_sum)/len(list_to_sum)
    # Library calculation
    lib_loss = LogLoss()
    vector_value = lib_loss(y_pred.reshape(-1, 1), y.reshape(-1, 1))
    assert abs(iter_value-vector_value) < eps, """The difference between values is greater than epsilon"""



# Test log loss gradient value
@pytest.mark.parametrize(
        "w,X,y",
        [(rng.random((NUM_FEATURES, 1)),
          rng.random((NUM_SAMPLES, NUM_FEATURES)), 
          np.random.randint(0, 2, (NUM_SAMPLES, 1)).astype(np.float64)) for _ in range(NUM_TESTS)]
)
def test_log_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> None:
    """
    Tests log loss gradient implementation for linear function
    
    Args
    ----
    w: np.ndarray
        weight vector of shape (D+1)x1 where D is number of features
    
    X: np.ndarray
        Feature matrix of shape Nx(D+1) where D is number of features; N is number of samples

    y: np.ndarray
        Target vector (binary labels 0 and 1) of shape Nx1; N is number of features
    """
    # Prepare log functions
    lib_log_loss = LogLoss()
    torch_sigm = torch.nn.Sigmoid()
    # Setting torch tensors with gradients for comparison
    w_torch = torch.tensor(w, requires_grad = True)
    X_torch = torch.tensor(X, requires_grad = True)
    y_torch = torch.tensor(y, requires_grad = True)
    # Calculating loss and grad of linear regression
    y_pred = torch_sigm(X_torch@w_torch) # Shape Nx1
    loss = -(y_torch*torch.log(y_pred) + (1-y_torch)*torch.log(1-y_pred)).mean()
    loss.backward()
    # Comparing results
    torch_grad = np.round(w_torch.grad.detach().numpy(), 4)
    lib_grad = np.round(lib_log_loss.grad(w, X, y), 4)
    assert (torch_grad-lib_grad).sum().item() == 0, "Implemented gradient does not equal to torch calculated gradient"




