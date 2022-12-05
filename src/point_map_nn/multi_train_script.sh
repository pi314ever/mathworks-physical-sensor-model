# Train models and compare plots

# Normal regularization constant (0.01)
python model.py -l -n "mae_point_map_med_reg" -s --loss "mae"
python model.py -l -n "mse_point_map_med_reg" -s --loss "mse"

# High regularization constant (0.03)
python model.py -l -n "mae_point_map_high_reg" -s --loss "mae" -r 0.03
python model.py -l -n "mse_point_map_high_reg" -s --loss "mse" -r 0.03

# Low regularization constant (0.005)
python model.py -l -n "mae_point_map_low_reg" -s --loss "mae" -r 0.005
python model.py -l -n "mse_point_map_low_reg" -s --loss "mse" -r 0.005

# No regularization
python model.py -t -n "mae_point_map_no_reg" -s --loss "mae" -r 0
python model.py -t -n "mse_point_map_no_reg" -s --loss "mse" -r 0