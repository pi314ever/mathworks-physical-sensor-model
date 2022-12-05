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

args = None  # type: ignore
LAYER_SIZES = [16, 16, 16, 16]

if __name__ == '__main__':
    import argparse

    class ModelParserNamespace(argparse.Namespace):
        train: bool
        load: bool
        model_name: str
        save_plots: bool
        loss: str
        regularization: float

    parser = argparse.ArgumentParser(description='Train a neural network to predict point maps')
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-l', '--load', action='store_true', help='Load a model from a file')
    parser.add_argument('-n', '--model-name', help='Model name (alphanumeric + _)', type=str, default='point_map_nn')
    parser.add_argument('-s', '--save-plots', help='Saves plots into corresponding pngs', action='store_true')
    parser.add_argument('--loss', help='Loss function to be applied. "mse" or "mae".', default='mse')
    parser.add_argument('-r', '--regularization', help='L2 regularization constant', type=float, default=0.01)

    args: ModelParserNamespace = parser.parse_args()  # type: ignore # Only adding type annotations


    LOAD_MODEL = args.load
    TRAIN_MODEL = args.train

    LOSS = args.loss
    REGULARIZATION_CONSTANT = args.regularization

    MODEL_FILE = f'model_weights/{args.model_name}.h5'
else:
    LOSS = 'mse'
    LOAD_MODEL = False
    MODEL_FILE = 'model_weights/point_map_nn.h5'
    TRAIN_MODEL = True
    REGULARIZATION_CONSTANT = 0.01

# ----------------------------- END CONFIGURATION ---------------------------- #

def create_model(layer_sizes=LAYER_SIZES, activation='relu', optimizer='adam', loss=LOSS, metrics=['mse'], reg=REGULARIZATION_CONSTANT):
    layers = [tf.keras.layers.Dense(layer_sizes[0], activation=activation, input_shape=(8,), activity_regularizer=tf.keras.regularizers.l2(reg))]
    layers += [tf.keras.layers.Dense(layer_size, activation=activation, activity_regularizer=tf.keras.regularizers.L2(reg)) for layer_size in layer_sizes[1:]]
    model = tf.keras.Sequential(
        layers=layers + [tf.keras.layers.Dense(2, activation='linear')]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def train_model(model: Sequential, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=None):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_val, Y_val), validation_batch_size=999999999999)
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
        model.fit(X_train, Y_train, epochs=100, batch_size=10000,validation_data=(X_val, Y_val), validation_batch_size=999999999999)
        model.save(MODEL_FILE)

    if args and args.save_plots:
        if not TRAIN_MODEL:
            X_val, Y_val = get_point_map_data('val')
        Y_pred = model.predict(X_val, batch_size=999999999999)                              # type: ignore # Caught by `if not TRAIN_MODEL` block
        visualize.plot_errors(Y_val, Y_pred,                                                         # type: ignore # Caught by `if not TRAIN_MODEL` block
            title=f'Predicted Errors with {args.loss.upper()} Loss ({args.regularization} Reg)',
            filename=f'errors_histogram_full_{args.model_name}_{str(args.regularization).split(".")[-1]}.png')
        visualize.plot_scatter(Y_val, Y_pred,                                                        # type: ignore # Caught by `if not TRAIN_MODEL` block
            title=f'Scatterplot Sample with {args.loss.upper()} Loss ({args.regularization} Reg)',
            filename=f'scatter_sample_{args.model_name}_{str(args.regularization).split(".")[-1]}.png')