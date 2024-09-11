import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, init_patterns: np.ndarray):
        self.init_patterns = init_patterns
        self.N = init_patterns.shape[1]
        self.W = np.tensordot(init_patterns, init_patterns, axes=0) / self.N

    def relax_state(self, state: np.ndarray, n_iter: int):
        energy_arr = []
        state_arr = []
        for i in range(n_iter):
            energy_arr.append(-0.5 * np.dot(np.dot(self.W, state), state))
            state_arr.append(np.copy(state))
            state = np.sign(np.dot(self.W, state))
        return state_arr, energy_arr


if __name__ == "__main__":
    h = HopfieldNetwork(np.array([[1, -1, -1, -1], [-1, -1, 1, -1]]))
    s, e = h.relax_state(np.array([]))
