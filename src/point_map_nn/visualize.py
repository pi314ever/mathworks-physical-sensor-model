
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from utils import get_point_map_data
from model import create_model

def plot_scatter(Y_test, Y_pred, title='Neural Network Predicted Sample Points', filename='scatter_sample.png'):
    fig = plt.figure()
    plt.scatter(Y_test[:50, 0], Y_test[:50, 1], label='Ground Truth')
    plt.scatter(Y_pred[:50, 0], Y_pred[:50, 1], label='Prediction')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    fig.savefig(filename)

def plot_errors(Y_test, Y_pred, title='Neural Network Predicted Errors', filename='errors_histogram_full.png'):
    # Calculate radial distances
    R = np.sqrt((Y_test[:, 0] - Y_pred[:,0]) ** 2 + (Y_test[:, 1] - Y_pred[:, 1]) ** 2)
    fig = plt.figure()
    plt.hist(R[R < 0.1], 200, label=f'{len(R[R < 0.1])} of {len(R)} shown')
    plt.xlabel('Radial distance')
    plt.ylabel('Count')
    plt.legend()
    plt.title(title)
    fig.savefig(filename)

if __name__ == '__main__':
    X_test, Y_test = get_point_map_data('test')
    model = create_model()
    model.load_weights('point_map_nn.h5')
    Y_pred = model.predict(X_test, batch_size = 99999999999999)
    plot_scatter(Y_test, Y_pred)
    plot_errors(Y_test, Y_pred)