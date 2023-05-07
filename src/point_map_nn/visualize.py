import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(
    Y_test,
    Y_pred,
    title="Neural Network Predicted Sample Points",
    filename="scatter_sample.png",
):
    fig = plt.figure()
    plt.scatter(Y_test[:50, 0], Y_test[:50, 1], label="Ground Truth")
    plt.scatter(Y_pred[:50, 0], Y_pred[:50, 1], label="Prediction")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    fig.savefig(f"images/{filename}")


def plot_errors(
    Y_test,
    Y_pred,
    cutoff=0.1,
    title="Neural Network Predicted Errors",
    filename="errors_histogram_full.png",
):
    # Calculate radial distances
    R = np.sqrt((Y_test[:, 0] - Y_pred[:, 0]) ** 2 + (Y_test[:, 1] - Y_pred[:, 1]) ** 2)
    fig = plt.figure()
    plt.hist(R[R < cutoff], 200, label=f"{len(R[R < cutoff])} of {len(R)} shown")
    plt.xlabel("Radial distance")
    plt.ylabel("Count")
    plt.legend()
    plt.title(title)
    fig.savefig(f"images/{filename}")


if __name__ == "__main__":
    print("Loading data...")
    X_val, Y_val = get_point_map_data("val")
    print("Loading MAE models...")
    model_mae_01 = create_model(loss="mae", reg=0.01)
    model_mae_01.load_weights("model_weights/mae_point_map_med_reg.h5")
    model_mae_03 = create_model(loss="mae", reg=0.03)
    model_mae_03.load_weights("model_weights/mae_point_map_high_reg.h5")
    model_mae_005 = create_model(loss="mae", reg=0.005)
    model_mae_005.load_weights("model_weights/mae_point_map_low_reg.h5")
    model_mae_0 = create_model(loss="mae", reg=0)
    model_mae_0.load_weights("model_weights/mae_point_map_no_reg.h5")
    print("Predicting MAE models...")
    fig = plt.figure()
    cutoff = 0.05
    print("Reg 0")
    Y_pred_mae_0 = model_mae_0.predict(X_val, batch_size=99999999999999)
    R_mae_0 = np.sqrt(
        (Y_val[:, 0] - Y_pred_mae_0[:, 0]) ** 2
        + (Y_val[:, 1] - Y_pred_mae_0[:, 1]) ** 2
    )
    plt.hist(
        R_mae_0[R_mae_0 < cutoff],
        200,
        label=f"MAE 0: {len(R_mae_0[R_mae_0 < cutoff])} of {len(R_mae_0)} shown",
    )
    del model_mae_0, Y_pred_mae_0, R_mae_0
    print("Reg 0.005")
    Y_pred_mae_005 = model_mae_005.predict(X_val, batch_size=99999999999999)
    R_mae_005 = np.sqrt(
        (Y_val[:, 0] - Y_pred_mae_005[:, 0]) ** 2
        + (Y_val[:, 1] - Y_pred_mae_005[:, 1]) ** 2
    )
    plt.hist(
        R_mae_005[R_mae_005 < cutoff],
        200,
        label=f"MAE 0.005: {len(R_mae_005[R_mae_005 < cutoff])} of {len(R_mae_005)} shown",
        alpha=0.5,
    )
    del model_mae_005, Y_pred_mae_005, R_mae_005
    print("Reg 0.01")
    Y_pred_mae_01 = model_mae_01.predict(X_val, batch_size=99999999999999)
    R_mae_01 = np.sqrt(
        (Y_val[:, 0] - Y_pred_mae_01[:, 0]) ** 2
        + (Y_val[:, 1] - Y_pred_mae_01[:, 1]) ** 2
    )
    plt.hist(
        R_mae_01[R_mae_01 < cutoff],
        200,
        label=f"MAE 0.01: {len(R_mae_01[R_mae_01 < cutoff])} of {len(R_mae_01)} shown",
        alpha=0.25,
    )
    del model_mae_01, Y_pred_mae_01, R_mae_01
    print("Reg 0.03")
    Y_pred_mae_03 = model_mae_03.predict(X_val, batch_size=99999999999999)
    R_mae_03 = np.sqrt(
        (Y_val[:, 0] - Y_pred_mae_03[:, 0]) ** 2
        + (Y_val[:, 1] - Y_pred_mae_03[:, 1]) ** 2
    )
    plt.hist(
        R_mae_03[R_mae_03 < cutoff],
        200,
        label=f"MAE 0.03: {len(R_mae_03[R_mae_03 < cutoff])} of {len(R_mae_03)} shown",
        alpha=0.25,
    )
    del model_mae_03, Y_pred_mae_03, R_mae_03
    print("Done")
    plt.xlabel("Radial distance")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Error Comparison MAE different regularization")
    fig.savefig("images/errors_histogram_mae_reg_all.png")
    # print('Predicting MAE...')
    # Y_pred_mae = model_mae.predict(X_val, batch_size = 99999999999999)
    # print('Calculating Radius MAE...')
    # R_mae = np.sqrt((Y_val[:, 0] - Y_pred_mae[:,0]) ** 2 + (Y_val[:, 1] - Y_pred_mae[:, 1]) ** 2)
    # del model_mae, Y_pred_mae
    # print('Loading MSE model...')
    # model_mse = create_model(loss='mse', reg=0.01)
    # model_mse.load_weights('model_weights/mse_point_map_med_reg.h5')
    # print('Predicting MSE...')
    # Y_pred_mse = model_mse.predict(X_val, batch_size = 99999999999999)
    # print('Calculating Radius MSE...')
    # R_mse = np.sqrt((Y_val[:, 0] - Y_pred_mse[:,0]) ** 2 + (Y_val[:, 1] - Y_pred_mse[:, 1]) ** 2)
    # del model_mse, Y_pred_mse
    # print('Plotting...')
    # fig = plt.figure()
    # plt.hist(R_mae[R_mae < 0.1], 200, label=f'MAE {len(R_mae[R_mae < 0.1])} of {len(R_mae)} shown')
    # plt.hist(R_mse[R_mse < 0.1], 200, label=f'MSE {len(R_mse[R_mse < 0.1])} of {len(R_mse)} shown', alpha=0.5)
    # plt.xlabel('Radial distance')
    # plt.ylabel('Count')
    # plt.legend()
    # plt.title('Error Comparison MSE vs MAE (0.01 Reg)')
    # fig.savefig('images/errors_histogram_mse_mae_med_reg_01.png')
