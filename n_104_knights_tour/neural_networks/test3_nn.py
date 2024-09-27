import numpy as np
from random import randint
import matplotlib.pyplot as plt

""" Attempt 1 with neural network
"""

MOVES = np.array(
    [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [-2, 1], [2, -1], [-2, -1]]
)
STABILIZATION_COUNT = 20

if __name__ == "__main__":
    map_shape = np.array([3, 3])
    map_area = map_shape[0] * map_shape[1]
    neurons_list_A = list()
    neurons_list_B = list()
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            for p in np.repeat([[i, j]], axis=0, repeats=8) + MOVES:
                if (
                    p[0] >= 0
                    and p[1] >= 0
                    and p[0] < map_shape[0]
                    and p[1] < map_shape[1]
                ):
                    lin_A = i * map_shape[0] + j
                    lin_B = p[0] * map_shape[0] + p[1]
                    # if (
                    #     not any(
                    #         np.array_equal([lin_A, lin_B], row)
                    #         for row in np.array([neurons_list_A, neurons_list_B]).T
                    #     )
                    # ) and (
                    #     not any(
                    #         np.array_equal([lin_B, lin_A], row)
                    #         for row in np.array([neurons_list_A, neurons_list_B]).T
                    #     )
                    # ):
                    neurons_list_A.append(lin_A)
                    neurons_list_B.append(lin_B)

    neurons_list_A = np.array(neurons_list_A)
    neurons_list_B = np.array(neurons_list_B)
    print(np.array([neurons_list_A, neurons_list_B]).T)

    linearized_state_function = np.zeros_like(neurons_list_A)
    linearized_output_function = np.random.randint(0, 2, neurons_list_A.shape)

    running = True
    stabilization = 0
    while running:
        new_state = np.zeros_like(linearized_state_function)
        new_output = np.zeros_like(linearized_output_function)
        for i, (A, B) in enumerate(zip(neurons_list_A, neurons_list_B)):
            new_state[i] = linearized_state_function[i] + (
                2
                - np.sum(
                    linearized_output_function[
                        np.where(
                            np.logical_and(neurons_list_A == A, neurons_list_B != B)
                        )
                    ]
                )
                - np.sum(
                    linearized_output_function[
                        np.where(
                            np.logical_and(neurons_list_A != A, neurons_list_B == B)
                        )
                    ]
                )
                - np.sum(
                    linearized_output_function[
                        np.where(
                            np.logical_or(neurons_list_A == B, neurons_list_B == A)
                        )
                    ]
                )
            )
            if new_state[i] > 3:
                new_output[i] = 1
            elif new_state[i] < 0:
                new_output[i] = 0
            else:
                new_output[i] = linearized_output_function[i]
            # new_output[i] = (
            #     1
            #     if new_state[i] > 3
            #     else (0 if new_state[i] < 0 else linearized_output_function[i])
            # )
        if np.array_equal(new_output, linearized_output_function):
            linearized_state_function = np.copy(new_state)
            linearized_output_function = np.copy(new_output)
            stabilization += 1
        else:
            linearized_state_function = np.copy(new_state)
            linearized_output_function = np.copy(new_output)
            stabilization = 0
        if stabilization == STABILIZATION_COUNT:
            running = False

    network_A_linearized = neurons_list_A[np.where(linearized_output_function == 1)]
    network_B_linearized = neurons_list_B[np.where(linearized_output_function == 1)]

    network_A = np.array(
        [
            network_A_linearized // map_shape[0],
            network_A_linearized % map_shape[0],
        ]
    ).T
    network_B = np.array(
        [
            network_B_linearized // map_shape[0],
            network_B_linearized % map_shape[0],
        ]
    ).T

    for a, b in zip(network_A, network_B):
        plt.plot([a[0], b[0]], [a[1], b[1]])
        # plt.plot([a[1], b[1]], [a[0], b[0]])

    plt.show()
    print(network_A)
    print(network_B)

    print("stop here")
