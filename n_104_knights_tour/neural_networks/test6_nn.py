import numpy as np
import matplotlib.pyplot as plt

""" Attempt 3 with neural network
"""

MOVES = np.array(
    [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [-2, 1], [2, -1], [-2, -1]]
)


class Neuron:
    def __init__(self, pos1: np.ndarray, pos2: np.ndarray):
        self.pos1 = pos1
        self.pos2 = pos2
        self.neighbors = list()
        self.state = 0
        self.new_state = 0
        self.output = np.random.randint(0, 2)
        self.new_output = 0

    def equals(self, n):
        return (
            np.array_equal(self.pos1, n.pos1) and np.array_equal(self.pos2, n.pos2)
        ) or (np.array_equal(self.pos2, n.pos1) and np.array_equal(self.pos1, n.pos2))

    def is_neighbor(self, n):
        return (
            np.array_equal(self.pos1, n.pos1)
            or np.array_equal(self.pos2, n.pos2)
            or np.array_equal(self.pos1, n.pos2)
            or np.array_equal(self.pos2, n.pos1)
        )

    def try_add_neighbor(self, n):
        if self.is_neighbor(n) and not self.equals(n):
            self.neighbors.append(n)

    def compute_new_state(self):
        self.new_state = max(
            -8, min(self.state + 2 - np.sum([n.output for n in self.neighbors]), 8)
        )
        if self.new_state > 3:
            self.new_output = 1
        elif self.new_state < 0:
            self.new_output = 0
        else:
            self.new_output = self.output

    def update_state(self):
        state_changed = self.new_state != self.state
        output_changed = self.new_output != self.output
        self.output = self.new_output
        self.state = self.new_state
        return state_changed, output_changed


def simulate(map_shape):
    neurons = list()
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            pos1 = np.array([i, j])
            for pos2 in np.repeat([pos1], axis=0, repeats=8) + MOVES:
                if (
                    pos2[0] >= 0
                    and pos2[1] >= 0
                    and pos2[0] < map_shape[0]
                    and pos2[1] < map_shape[1]
                ):
                    candidate = Neuron(pos1, pos2)
                    if not any(n.equals(candidate) for n in neurons):
                        neurons.append(candidate)

    for n1 in neurons:
        for n2 in neurons:
            n1.try_add_neighbor(n2)

    flag_neighbors_plot = False
    if flag_neighbors_plot:
        for n in neurons:
            plt.plot([n.pos1[0], n.pos2[0]], [n.pos1[1], n.pos2[1]], color="blue")
            for n1 in n.neighbors:
                plt.plot(
                    [n1.pos1[0], n1.pos2[0]], [n1.pos1[1], n1.pos2[1]], color="red"
                )
            plt.show()

    running = True
    converge = False
    converge_count = 0
    diverge_count = 0
    while running:
        for n in neurons:
            n.compute_new_state()
        update_report = np.array([n.update_state() for n in neurons])

        state_changed = any(update_report[:, 0])
        output_changed = any(update_report[:, 1])
        if (not output_changed) and state_changed:
            diverge_count += 1
            converge_count = 0
        elif not (output_changed and state_changed):
            converge_count += 1
            diverge_count = 0
        if converge_count == 2:
            running = False
            converge = True
        elif diverge_count == 2:
            running = False
            converge = False
    return n, neurons, converge


def get_converging_simulation(map_shape):
    count = 0
    while True:
        n, neurons, converge = simulate(map_shape)
        if converge:
            return n, neurons
        else:
            print(count)
            count += 1


if __name__ == "__main__":
    map_shape = np.array([5, 5])

    n, neurons = get_converging_simulation(map_shape)

    print([n.output for n in neurons])
    print([n.state for n in neurons])
    print("stop here")
    for n in neurons:
        if n.output == 1:
            plt.plot([n.pos1[0], n.pos2[0]], [n.pos1[1], n.pos2[1]])
    plt.show()
