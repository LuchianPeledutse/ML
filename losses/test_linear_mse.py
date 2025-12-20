import torch
import numpy as np
import random as rd


import pytest
from linear_mse import MSE

VECTOR_SIZE = 53
NUM_TESTS = 1000
NUM_FEATURES = 10
NUM_SAMPLES = 100
rng = np.random.default_rng()


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

'--------------------------------------------------------------------------------'
