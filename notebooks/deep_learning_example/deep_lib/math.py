import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))

    assert (A.shape == Z.shape)

    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z

    return A, cache



def compute_cost(YHat, Y, layer_dims, parameters, lambd):
    """
    Computes the cost function as the sum of the cross entropy cost and the regularization cost.

    Arguments:
    YHat -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    layer_dims -- layer dimensions for L2 regularization cost calculation
    parameters -- parameters for L2 regularization cost calculation
    lambd -- lamda parameter for the L2 regularization cost calculation

    Returns:
    cost -- total cost as the sum of the cross-entropy cost + the L2 regularization cost
    """
    m = Y.shape[1]

    cross_entropy_cost = log_loss_cost(YHat, Y)

    # L2 Regularization cost calculation. If lamdda = 0, the L2_regularization_cost is zero.
    # So, no addition for the total cost is made.
    W_sum = 0
    for l in range(1, len(layer_dims)):
        W = parameters['W' + str(l)]
        W_sum += np.nansum(np.square(W))
    L2_regularization_cost = (lambd / (2 * m)) * W_sum

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def log_loss_cost(YHat, Y):
    """
    Compute the cross-entropy cost.

    Arguments:
    YHat -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(YHat).T) - np.dot(1 - Y, np.log(1 - YHat).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost
