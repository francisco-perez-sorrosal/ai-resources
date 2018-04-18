import time
import datetime
from deep_lib.model_parameters import initialize_parameters, update_parameters
from deep_lib.forward_prop import L_model_forward
from deep_lib.backward_prop import L_model_backward
from deep_lib.math import compute_cost
from deep_lib.eval import evaluate
from deep_lib.plot_utils import plot_eval, CostDiagram

import logging
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

from ipywidgets import FloatProgress
from IPython.display import display, clear_output


class LLayerModel:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.logger.info("LLayer model created...")

    def execute(self, X, Y, hp, save_cost=False):  # lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        save_cost -- if True, it prints the cost every 100 iterations

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        progress_bar = FloatProgress(min=0, max=hp.iterations, description='Iterations:')
        costs_diagram = CostDiagram(hp.learning_rate)
        display(progress_bar, costs_diagram.get_fig())
        self.logger.info('Starting model execution...')
        iterations = []
        costs = []  # keep track of costs in iterations
        elapsed_times = []

        # Parameters initialization.
        parameters = initialize_parameters(hp.layer_dimensions, hp.param_initializer)

        # Loop (gradient descent)
        start_time = time.time()
        for i in range(0, hp.iterations):

            start_loop_time = datetime.datetime.now()
            progress_bar.value += 1
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(X, parameters, hp.keep_prob)

            # Compute cost.
            cost = compute_cost(AL, Y, hp.layer_dimensions, parameters, hp.lambd)

            # Backward propagation.
            grads = L_model_backward(AL, Y, caches, parameters, hp.lambd, hp.keep_prob)

            # Update parameters.
            parameters = update_parameters(parameters, grads, hp.learning_rate)

            elapsed_loop_time = datetime.datetime.now() - start_loop_time

            # Print the cost every 100 training example
            if save_cost and i % 100 == 0:
                costs.append((i, np.squeeze(cost), elapsed_loop_time.microseconds / 1000))
                elapsed_times.append(elapsed_loop_time.microseconds / 1000)
                clear_output(wait=True)
                costs_diagram.update_data(costs)
                display(progress_bar, costs_diagram.get_fig())
                self.logger.info("Cost after iteration %i: %f (elapsed time: %sms)" % (
                i, cost, elapsed_loop_time.microseconds / 1000))

        elapsed_time = time.time() - start_time
        self.logger.info('Model execution finished. Elapsed time %s' %
                         (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        return parameters, costs

    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(parameters) // 2  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = L_model_forward(X, parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        print("Accuracy: " + str(np.sum((p == y) / m)))

        return p

    def evaluate(self, y, yhat):
        cm, p, r, f1 = evaluate(y, yhat)
        self.logger.info("Precision/Recall/F1: %f/%f/%f" % (p, r, f1))
        plot_eval(p, r, f1)
        return cm, p, r, f1
