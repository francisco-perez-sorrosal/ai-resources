import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


class CostDiagram:
    def __init__(self, learning_rate):
        self.fig, ax = plt.subplots()
        plt.title("Learning rate =" + str(learning_rate))
        self.ax1, self.ax2 = create_two_scales(ax, 'r', 'b')

    def update_data(self, costs_data):
        iterations = [x[0] for x in costs_data]
        costs = [x[1] for x in costs_data]
        elapsed_times = [x[2] for x in costs_data]
        self.ax1.plot(iterations, costs, color='r')
        self.ax2.plot(iterations, elapsed_times, color='b')

    def get_fig(self):
        return self.fig


def create_two_scales(ax1, c1, c2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax1 : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('cost')
    color_y_axis(ax1, c1)

    ax2.set_ylabel('elapsed time (ms)')
    color_y_axis(ax2, c2)
    return ax1, ax2


def color_y_axis(ax, color):
    """
    Color your axes.
    """
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None


def load_image(url):
    print("Loading image from url %s" % url)
    return imread(url)


def plot_eval(precision, recall, f1):
    fig, ax = plt.subplots()
    ind = np.arange(1, 4)

    pm, pc, pn = plt.bar(ind, np.array([precision, recall, f1]))
    pm.set_facecolor('r')
    pc.set_facecolor('g')
    pn.set_facecolor('b')
    ax.set_xticks(ind)
    ax.set_xticklabels(['Precision', 'Recall', 'F1'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Value')
    ax.set_title('Model Evaluation')

    plt.show()
