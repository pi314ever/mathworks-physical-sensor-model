#!/bin/bash
# Train models and compare plots

# No regularization
python model.py combined -r 0 -L
python model.py radial -r 0 -L
python model.py tangential -r 0 -L

# Low regularization constant (0.005)
python model.py combined -r 0.005 -L
python model.py radial -r 0.005 -L
python model.py tangential -r 0.005 -L

# Normal regularization constant (0.01)
python model.py combined -r 0.01 -L
python model.py radial -r 0.01 -L
python model.py tangential -r 0.01 -L

# High regularization constant (0.03)
python model.py combined -r 0.03 -L
python model.py radial -r 0.03 -L
python model.py tangential -r 0.03 -L
