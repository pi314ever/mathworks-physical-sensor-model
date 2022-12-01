### Model class for point maps neural network
# Inputs: (K, P, X_distorted, Y_distorted)
# Outputs: (X, Y)

# Script examples:
# Train with MAE loss and saving plots
#   python model.py -t -n "mae_point_map" -s --loss "mae"

import tensorflow as tf

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import get_point_map_data

Sequential = tf.keras.models.Sequential


# ------------------------------- CONFIGURATION ------------------------------ #

import argparse

class ModelParserNamespace(argparse.Namespace):
    train: bool
    load: bool
    model_name: str
    save_plots: bool
    loss: str

parser = argparse.ArgumentParser(description='Train a neural network to predict point maps')
parser.add_argument('-t', '--train', action='store_true', help='Train the model')
parser.add_argument('-l', '--load', action='store_true', help='Load a model from a file')
parser.add_argument('-n', '--model-name', help='Model name (alphanumeric + _)', type=str, default='point_map_nn')
parser.add_argument('-s', '--save-plots', help='Saves plots into corresponding pngs', action='store_true')
parser.add_argument('--loss', help='Loss function to be applied. "mse" or "mae".', default='mse')

args: ModelParserNamespace = parser.parse_args()  # type: ignore # Only adding type annotations

LAYER_SIZES = [16, 16, 16, 16]

LOAD_MODEL = args.load
TRAIN_MODEL = args.train

LOSS = args.loss

MODEL_FILE = f'{args.model_name}.h5'

# ----------------------------- END CONFIGURATION ---------------------------- #

def create_model(layer_sizes=LAYER_SIZES, activation='relu', optimizer='adam', loss=LOSS, metrics=['mse']):
    layers = [tf.keras.layers.Dense(layer_sizes[0], activation=activation, input_shape=(8,), activity_regularizer=tf.keras.regularizers.l2(0.01))]
    layers += [tf.keras.layers.Dense(layer_size, activation=activation, activity_regularizer=tf.keras.regularizers.L2(0.01)) for layer_size in layer_sizes[1:]]
    model = tf.keras.Sequential(
        layers=layers + [tf.keras.layers.Dense(2, activation='linear')]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def train_model(model: Sequential, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=None):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_val, Y_val))
    return model

def test_model(model: Sequential, X_test, Y_test):
    err = model.evaluate(X_test, Y_test, batch_size = 99999999999999)
    return err


if __name__ == '__main__':
    import visualize
    print(tf.config.list_physical_devices('GPU'))
    model = create_model()
    model.summary()
    if LOAD_MODEL:
        model.load_weights(MODEL_FILE)
    if TRAIN_MODEL:
        print('Gathering data...')
        print('Gathered training data...')
        X_train, Y_train = get_point_map_data('train')
        print(f'Training data of size {X_train.shape}, {Y_train.shape} obtained')
        print('Gathering validation data...')
        X_val, Y_val = get_point_map_data('val')
        print(f'Validation data of size {X_val.shape}, {Y_val.shape} obtained')
        print('Finished gathering data')
        model = train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, batch_size = 10000)
    model.save(MODEL_FILE)
    X_test, Y_test = get_point_map_data('test')
    print(test_model(model, X_test, Y_test))
    Y_pred = model.predict(X_test, batch_size=999999999999)
    if args.save_plots:
        visualize.plot_errors(Y_test, Y_pred, title=f'Neural Network Predicted Errors with {args.loss.upper()} Loss', filename=f'errors_histogram_full_{args.model_name}.png')
        visualize.plot_scatter(Y_test, Y_pred, title=f'Neural Network Predicted Sample with {args.loss.upper()} Loss', filename=f'scatter_sample_{args.model_name}.png')