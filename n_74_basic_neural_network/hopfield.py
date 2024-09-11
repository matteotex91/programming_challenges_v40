import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, init_patterns: np.ndarray):
        self.init_patterns = init_patterns
        self.N = init_patterns.shape[1]
        self.W = np.zeros((self.N, self.N))
        for pattern in init_patterns:
            self.W += np.tensordot(pattern, pattern, axes=0) / self.N
        self.W *= np.ones((self.N, self.N)) - np.eye(self.N)

    def relax_state_synchro(self, state: np.ndarray, n_iter: int):
        energy_arr = []
        state_arr = []
        overlaps = []
        for i in range(n_iter):
            energy_arr.append(-0.5 * np.dot(np.dot(self.W, state), state))
            state_arr.append(np.copy(state))
            overlaps.append(
                np.array([np.dot(state, pattern) for pattern in self.init_patterns])
                / self.N
            )
            state = np.sign(np.dot(self.W, state))
        return np.array(state_arr), np.array(energy_arr), np.array(overlaps)

    def relax_state_synchro_autodetect_convergence(self, state: np.ndarray):
        energy_arr = []
        state_arr = []
        overlaps = []
        running = True
        while running:
            newstate = np.sign(np.dot(self.W, state))
            running = not np.array_equal(newstate, state)
            state = newstate
            overlaps.append(
                np.array([np.dot(state, pattern) for pattern in self.init_patterns])
                / self.N
            )
            energy_arr.append(-0.5 * np.dot(np.dot(self.W, state), state))
            state_arr.append(np.copy(state))
        return np.array(state_arr), np.array(energy_arr), np.array(overlaps)


if __name__ == "__main__":
    h = HopfieldNetwork(
        np.array([[1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, 1, -1, -1, -1, -1, -1]])
    )
    s, e, o = h.relax_state_synchro_autodetect_convergence(
        np.array([1, 1, -1, -1, -1, -1, -1, 1])
    )
    print("stop here")
