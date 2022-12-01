# Train two models and compare plots
python model.py -t -n "mae_point_map" -s --loss "mae"
python model.py -t -n "mse_point_map" -s --loss "mse"