import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle
import tensorflow as tf

""" Build a double layer nn from scratch.
Activation function : sigmoid
Not converging, almost all results produced are (1,1,1...1)
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


size_1 = 784
size_2 = 128
size_3 = 10

shape1 = (size_2, size_1)
shape2 = (size_3, size_2)

epochs = 3
iterations_per_epoch = 1
learning_rate = 0.01
iteration_per_point = 3

W1 = np.random.random(shape1)  # generate random weights
D1 = np.random.random(size_2)  # generate random biases
W2 = np.random.random(shape2)  # generate random weights
D2 = np.random.random(size_3)  # generate random biases


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


X = np.reshape(x_train, (x_train.shape[0], 784)) / np.max(x_train)
Y = np.array([np.eye(10)[y] for y in y_train])  # canonical vectors as output
costs = []
xy_data = list(zip(X, Y))
for epoch in range(epochs):
    print(f"epoch {epoch+1} over {epochs}")
    for i in range(iterations_per_epoch):
        shuffle(xy_data)
        for x, y in tqdm(xy_data):
            for j in range(iteration_per_point):
                (
                    g_W1,
                    g_D1,
                    g_W2,
                    g_D2,
                ) = cost_gradient_W_D(W1, D1, W2, D2, [x], [y])
                W1 -= g_W1 * learning_rate
                D1 -= g_D1 * learning_rate
                W2 -= g_W2 * learning_rate
                D2 -= g_D2 * learning_rate
        costs.append(cost(W1, D1, W2, D2, X, Y))


X_T = np.reshape(x_test, (x_test.shape[0], 784)) / np.max(x_test)
Y_T = y_test
confusion_matrix = np.zeros((10, 10))

for x, y in zip(X_T, Y_T):
    y_calc = np.argmax(transform(W1, D1, W2, D2, x))
    confusion_matrix[y, y_calc] += 1
accuracy = (
    100 * np.sum([confusion_matrix[i, i] for i in range(10)]) / np.sum(confusion_matrix)
)
print(f"accuracy : {accuracy}%")


plt.plot(costs)
plt.show()

plt.pcolormesh(confusion_matrix)
plt.show()
