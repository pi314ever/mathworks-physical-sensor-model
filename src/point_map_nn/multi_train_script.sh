# Train models and compare plots

# Normal regularization constant (0.01)
python model.py combined -r 0.01
python model.py radial -r 0.01
python model.py tangential -r 0.01

# High regularization constant (0.03)
python model.py combined -r 0.03
python model.py radial -r 0.03
python model.py tangential -r 0.03

# Low regularization constant (0.005)
python model.py combined -r 0.005
python model.py radial -r 0.005
python model.py tangential -r 0.005

# No regularization
python model.py combined -r 0
python model.py radial -r 0
python model.py tangential -r 0
