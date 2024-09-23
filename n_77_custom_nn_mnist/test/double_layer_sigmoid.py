import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ortho_group

""" Build a double layer nn from scratch.
Activation function : sigmoid



"""


def transform(W1, D1, W2, D2, X):
    X1 = activation_function(np.dot(W1, X) + D1)
    return activation_function(np.dot(W2, X1) + D2)


def activation_function(v: np.ndarray) -> np.ndarray:
    return np.sinh(v) / np.cosh(v)


def activation_function_derivative(v: np.ndarray) -> np.ndarray:
    return 1 / np.power(np.cosh(v), 2)


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
        y_calc = sigmoid(sigm_arg)
        gradient_D += (y_calc - y_exp) * activation_function_derivative(sigm_arg)
        gradient_W += np.tensordot(gradient_D, x_calc, axes=0)
    return gradient_W, gradient_D


size = 100
iterations = 10000
learning_rage = 0.001

W = np.random.random((size, size))  # generate random weights
D = np.random.random(size)  # generate random biases

# X = [np.random.random(size) for i in range(size)]
# Y = [row for row in np.eye(size)]
X = ortho_group.rvs(dim=size)  # orthogonal inputs
Y = np.eye(size)  # canonical vectors as output
costs = []

for i in tqdm(range(iterations)):
    g_W, g_D = cost_gradient_W_D(W, D, X, Y)
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
