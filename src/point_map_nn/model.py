### Model class for point maps neural network
# Inputs: (K, P, X_distorted, Y_distorted)
# Outputs: (X, Y)

import tensorflow as tf

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import get_point_map_data

Sequential = tf.keras.models.Sequential

def create_model(layer_sizes=[16, 16], activation='relu', optimizer='adam', loss='mse', metrics=['mse']):
    layers = [tf.keras.layers.Dense(layer_sizes[0], activation=activation, input_shape=(7,), activity_regularizer=tf.keras.regularizers.l2(0.01))]
    layers += [tf.keras.layers.Dense(layer_size, activation=activation, activity_regularizer=tf.keras.regularizers.L2(0.01)) for layer_size in layer_sizes[1:]]
    model = tf.keras.Sequential(
        layers=layers + [tf.keras.layers.Dense(2, activation='linear')]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def train_model(model: Sequential, epochs=10, batch_size=None):
    X_train, Y_train = get_point_map_data('train')
    X_val, Y_val = get_point_map_data('val')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_val, Y_val))
    return model

def test_model(model: Sequential, X_test, Y_test):
    err = model.evaluate(X_test, Y_test)
    return err


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    model = create_model(layer_sizes=[16, 16, 16, 16])
    # model.load_weights('point_map_nn.h5')
    model = train_model(model, epochs=100)
    print(model.summary())
    model.save('point_map_nn.h5')
    X_test, Y_test = get_point_map_data('test')
    for x, y in zip(X_test, Y_test):
        print(x)
        print(model.predict(x.reshape((1, 7))))
        print(y)
        break