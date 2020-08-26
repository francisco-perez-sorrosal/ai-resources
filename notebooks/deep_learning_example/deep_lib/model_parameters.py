import numpy as np
import logging


class Hyperparameters(object):
    """
        Arguments:
        iterations -- number of iterations of the optimization loop
        layers_dimensions -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        param_initializer --
        lambd --
        keep_prob --
    """


    def __init__(self, iterations, layer_dimensions, learning_rate, param_initializer, lambd=0., keep_prob=1., minibatch_size=16):
        self.iterations = iterations
        self.layer_dimensions = layer_dimensions
        self.learning_rate = learning_rate
        self.param_initializer = param_initializer
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.minibatch_size = minibatch_size

        self.__class_description = """Hyperparameters:
        Iterations: %d
        Layer dims: %s
        Learning_rate: %f
        Parameter initializer: %s
        Lambd (L2 Reg param): %.2f
        Keep Prob (Dropout): %.2f
        Minibatch size: %d
        """

    def __str__(self):
        return self.__class_description % (self.iterations,
                                           self.layer_dimensions,
                                           self.learning_rate,
                                           self.param_initializer.__name__,
                                           self.lambd,
                                           self.keep_prob,
                                           self.minibatch_size)


def zero_initializer(rows, cols):
    return np.zeros((rows, cols))


def random_initializer(rows, cols):
    return np.random.randn(rows, cols) * 10


def xavier_initializer(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(1. / cols)


def he_initializer(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2. / cols)


def initialize_parameters(layer_dims, initializer_f):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    logger = logging.getLogger(__name__)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = initializer_f(layer_dims[l], layer_dims[l - 1])
        parameters['b' + str(l)] = zero_initializer(layer_dims[l], 1)

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
        logger.info("W%d shape: %s" % (l, parameters['W' + str(l)].shape))
        logger.info("b%d shape: %s" % (l, parameters['b' + str(l)].shape))

    return parameters


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
