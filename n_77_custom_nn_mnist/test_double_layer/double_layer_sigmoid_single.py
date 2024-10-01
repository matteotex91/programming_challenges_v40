import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ortho_group

""" Build a double layer nn from scratch.
Activation function : sigmoid
Still todo : train over single samples in different epochs, not collectively
"""


def transform(W1, D1, W2, D2, X):
    return activation_function(np.dot(W2, activation_function(np.dot(W1, X) + D1)) + D2)


def activation_function(v: np.ndarray) -> np.ndarray:
    return np.sinh(v) / np.cosh(v)


def activation_function_derivative(v: np.ndarray) -> np.ndarray:
    return 1 / np.power(np.cosh(v), 2)


def cost(W1, D1, W2, D2, x_calc_arr, y_exp_arr) -> float:
    cost = 0
    for x_calc, y_exp in zip(x_calc_arr, y_exp_arr):
        y_delta = transform(W1, D1, W2, D2, x_calc) - y_exp
        cost += 0.5 * np.dot(y_delta, y_delta)
    return cost


def cost_gradient_W_D(W1, D1, W2, D2, x_calc_arr, y_exp_arr):
    gradient_W1 = np.zeros_like(W1)
    gradient_D1 = np.zeros_like(D1)
    gradient_W2 = np.zeros_like(W2)
    gradient_D2 = np.zeros_like(D2)
    for x_calc, y_exp in zip(x_calc_arr, y_exp_arr):
        y_calc = transform(W1, D1, W2, D2, x_calc)
        y_delta = y_calc - y_exp
        y_calc_1 = activation_function(np.dot(W1, x_calc) + D1)
        y_1_der = activation_function_derivative(np.dot(W1, x_calc) + D1)
        y_2_der = activation_function_derivative(
            np.dot(W2, activation_function(np.dot(W1, x_calc) + D1)) + D2
        )
        gradient_W1 += np.tensordot(
            np.dot(y_delta * y_2_der, W2) * y_1_der, x_calc, axes=0
        )
        gradient_D1 += np.dot(y_delta * y_2_der, W2) * y_1_der
        gradient_W2 += np.tensordot(y_delta * y_2_der, y_calc_1, axes=0)
        gradient_D2 += y_delta * y_2_der

    return gradient_W1, gradient_D1, gradient_W2, gradient_D2


size = 10
epochs = 5
iterations = 100
learning_rate = 0.001

W1 = np.random.random((size, size))  # generate random weights
D1 = np.random.random(size)  # generate random biases
W2 = np.random.random((size, size))  # generate random weights
D2 = np.random.random(size)  # generate random biases

# X = [np.random.random(size) for i in range(size)]
# Y = [row for row in np.eye(size)]
X = ortho_group.rvs(dim=size)  # orthogonal inputs
Y = np.eye(size)  # canonical vectors as output
costs = []

for i in tqdm(range(iterations)):
    (
        g_W1,
        g_D1,
        g_W2,
        g_D2,
    ) = cost_gradient_W_D(W1, D1, W2, D2, X, Y)
    W1 -= g_W1 * learning_rate
    D1 -= g_D1 * learning_rate
    W2 -= g_W2 * learning_rate
    D2 -= g_D2 * learning_rate
    costs.append(cost(W1, D1, W2, D2, X, Y))

confusion_matrix = []
accuracy = 0.0
for i, x in enumerate(X):
    y = transform(W1, D1, W2, D2, x)
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
