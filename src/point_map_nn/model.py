### Model class for point maps neural network
# Inputs: (K, P, X_distorted, Y_distorted)
# Outputs: (X, Y)

import tensorflow as tf

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data.data_util import get_point_map_data

Sequential = tf.keras.models.Sequential


# ------------------------------- CONFIGURATION ------------------------------ #

import argparse

parser = argparse.ArgumentParser(description='Train a neural network to predict point maps')
parser.add_argument('-t', '--train', action='store_true', help='Train the model')
parser.add_argument('-l', '--load', default=False, action='store_true', help='Load a model from a file')
parser.add_argument('-n', '--model-name', help='Model name of ')

args = parser.parse_args()
print(args, args.train)

LAYER_SIZES = [16, 16, 16, 16]

LOAD_MODEL = False
TRAIN_MODEL = True

MODEL_FILE = 'point_map_nn.h5'

# ----------------------------- END CONFIGURATION ---------------------------- #

def create_model(layer_sizes=LAYER_SIZES, activation='relu', optimizer='adam', loss='mse', metrics=['mse'], reg=0.01):
    layers = [tf.keras.layers.Dense(layer_sizes[0], activation=activation, input_shape=(8,), activity_regularizer=tf.keras.regularizers.l2(reg))]
    layers += [tf.keras.layers.Dense(layer_size, activation=activation, activity_regularizer=tf.keras.regularizers.L2(reg)) for layer_size in layer_sizes[1:]]
    model = tf.keras.Sequential(
        layers=layers + [tf.keras.layers.Dense(2, activation='linear')]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def train_model(model: Sequential, epochs=10, batch_size=None):
    print('Gathering data...')
    print('Gathered training data...')
    X_train, Y_train = get_point_map_data('train')
    print(f'Training data of size {X_train.shape}, {Y_train.shape} obtained')
    print('Gathering validation data...')
    X_val, Y_val = get_point_map_data('val')
    print(f'Validation data of size {X_val.shape}, {Y_val.shape} obtained')
    print('Finished gathering data')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_val, Y_val))
    return model

def test_model(model: Sequential, X_test, Y_test):
    err = model.evaluate(X_test, Y_test, batch_size = 99999999999999)
    return err


# if __name__ == '__main__':
#     print(tf.config.list_physical_devices('GPU'))
#     model = create_model()
#     model.summary()
#     if LOAD_MODEL:
#         model.load_weights(MODEL_FILE)
#     if TRAIN_MODEL:
#         model = train_model(model, epochs=100, batch_size = 10000)
#     model.save(MODEL_FILE)
#     X_test, Y_test = get_point_map_data('test')
#     print(test_model(model, X_test, Y_test))