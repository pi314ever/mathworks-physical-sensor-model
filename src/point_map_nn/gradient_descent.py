# Gradient descent algorithm to solve distortion problem
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def gradient_descent(x0, f, lr=0.01, eps=1e-5, max_iter=1000):
    x1 = x0
    for _ in tqdm(range(max_iter), desc="Gradient Descent"):
        x0 = tf.Variable(x1)
        with tf.GradientTape() as tape:
            tape.watch(x0)
            y = f(x0)
        # x1 = x0 - f(x0) / df(x0) * lr
        x1 = x0 - tape.gradient(y, x0).numpy() * lr
        if np.linalg.norm(x1 - x0) < eps:
            print("Converged")
            break
    print(f"Final loss: {np.linalg.norm(x1 - x0)}")
    return x1


if __name__ == "__main__":
    import os, sys
    import matplotlib.pyplot as plt

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

    from utils.distortions import get_distorted_location
    from model import load_model_from_dirs

    Xd, Yd = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    Xd, Yd = Xd.flatten(), Yd.flatten()
    K = np.array([0.1, 0.03, 0.005])
    P = np.array([0.0, 0.0])

    # def f(XY):
    #     xd, yd = get_distorted_location(XY[0], XY[1], K, P)
    #     return tf.reduce_sum((Xd - xd) ** 2 + (Yd - yd) ** 2)

    # gd_guess = np.array(gradient_descent([Xd, Yd], f, max_iter=10000))
    # save_array = np.stack([Xd, Yd, gd_guess[0], gd_guess[1]], axis=-1)
    # gd_guess = gd_guess.T
    # np.savetxt("test_gd.txt", save_array)
    save_array = np.loadtxt("test_gd.txt")
    gd_guess = save_array[:, 2:]

    # model = load_model_from_dirs(["./models/combined_l4_s16_r0.0"])
    # model = load_model_from_dirs(["./models/combined_l4_s16_r0.0"])
    # model_name = "combined small"
    model = load_model_from_dirs(
        ["./models/radial_l4_s16_r0.0", "./models/tangential_l4_s16_r0.0"]
    )
    model_name = "separate small"
    # model = load_model_from_dirs(["./big_models/combined_l6_s128_r0.0"])
    # model_name = "combined large"

    XY = model.predict(np.array([Xd, Yd]).T, K, P)
    X, Y = XY[:, 0], XY[:, 1]

    diff = np.linalg.norm(gd_guess - XY, axis=-1)
    mean = np.mean(diff)
    std = np.std(diff)
    plt.figure()
    plt.imshow(diff.reshape(100, 100))
    plt.title(f"Errors in locations ({model_name}): {mean:.3f} +/- {std:.3f}")
    plt.xlabel("$X_d$")
    plt.ylabel("$Y_d$")
    plt.colorbar(label="Error in XY")
    plt.show()

    plt.figure()
    plt.scatter(gd_guess[:, 0], gd_guess[:, 1], s=1, label="SGD")
    plt.scatter(X, Y, s=1, label="PMNN")
    plt.title(f"Gradient Descent vs. PMNN ({model_name})")
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.legend()
    plt.show()
