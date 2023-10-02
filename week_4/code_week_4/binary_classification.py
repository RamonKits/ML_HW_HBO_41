import numpy as np

def sigmoid(z):
    """
    Calculate the sigmoid function

    Returns the sigmoid of z

    Parameters
    ----------
    z : numpy array

    Returns
    -------
    numpy array containing the sigmoid of z
    """
    return 1 / (1 + np.exp(-z))

def cost(X, y, theta):
    """
    Calculate the cost function

    Returns the cost of the current theta

    Parameters
    ----------
    X : numpy array
    y : numpy array
    theta : numpy array

    Returns
    -------
    float containing the cost of the current theta
    """
    h = sigmoid(np.dot(X, theta))
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient(X, y, theta):
    """
    Calculate the gradient

    Returns the gradient of the current theta

    Parameters
    ----------
    X : numpy array
    y : numpy array
    theta : numpy array

    Returns
    -------
    numpy array containing the gradient of the current theta
    """
    h = sigmoid(np.dot(X, theta))
    return np.dot(X.T, (h - y)) / y.size