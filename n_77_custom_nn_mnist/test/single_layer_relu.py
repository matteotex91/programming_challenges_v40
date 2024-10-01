import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ortho_group
from random import shuffle

""" Build a single layer nn from scratch.
Activation function : sigmoid
Works not so bad for orthogonal inputs
100% accuracy
"""


def apply(W, D, X):
    return activation(np.dot(W, X) + D)


def activation(v: np.ndarray) -> np.ndarray:  # relu analytical
    return np.log(1 + np.exp(v))


def activation_1(v: np.ndarray) -> np.ndarray:
    return np.divide(1, 1 + np.exp(-v))


def cost(W, D, x_calc_arr, y_exp_arr) -> float:
    cost = 0
    for x_calc, y_exp in zip(x_calc_arr, y_exp_arr):
        y_delta = apply(W, D, x_calc) - y_exp
        cost += 0.5 * np.dot(y_delta, y_delta)
    return cost


def cost_gradient_W_D(W, D, x_calc_arr, y_exp_arr):
    gradient_D = np.zeros_like(D)
    gradient_W = np.zeros_like(W)
    for x_calc, y_exp in zip(x_calc_arr, y_exp_arr):
        sigm_arg = np.dot(W, x_calc) + D
        y_calc = activation(sigm_arg)
        gradient_D += (y_calc - y_exp) * activation_1(sigm_arg)
        gradient_W += np.tensordot(gradient_D, x_calc, axes=0)
    return gradient_W, gradient_D


if __name__ == "__main__":
    size = 100
    iterations = 60000
    learning_rage = 0.01
    flag_train_together = False

    W = np.random.random((size, size))  # generate random weights
    D = np.random.random(size)  # generate random biases

    # X = [np.random.random(size) for i in range(size)]
    # Y = [row for row in np.eye(size)]
    X = ortho_group.rvs(dim=size)  # orthogonal inputs
    Y = np.eye(size)  # canonical vectors as output
    costs = []

    if flag_train_together:
        for i in tqdm(range(iterations)):
            g_W, g_D = cost_gradient_W_D(W, D, X, Y)
            W -= g_W * learning_rage
            D -= g_D * learning_rage
            costs.append(cost(W, D, X, Y))
    else:
        epochs = 6
        for i in range(epochs):
            print(f"epoch {i+1} over {epochs}")
            shuffle_data = list(zip(X, Y))
            shuffle(shuffle_data)
            for x, y in tqdm(shuffle_data):
                for i in range(iterations // epochs):
                    g_W, g_D = cost_gradient_W_D(W, D, [x], [y])
                    W -= g_W * learning_rage
                    D -= g_D * learning_rage
            costs.append(cost(W, D, X, Y))

    confusion_matrix = []
    accuracy = 0.0
    for i, x in enumerate(X):
        y = apply(W, D, x)
        confusion_matrix.append(y)
        if np.argmax(y) == i:
            accuracy += 1
        print(y)
    accuracy *= 100 / size
    print(f"accuracy : {accuracy}%")

    plt.plot(costs)
    plt.show()

    plt.pcolormesh(confusion_matrix)
    plt.show()
