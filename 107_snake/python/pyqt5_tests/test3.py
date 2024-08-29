import numpy as np


class testclass:
    def __init__(self, var: np.ndarray):
        self._var = var


v = np.array([1, 2, 3])
t = testclass(v)
print(t._var)
v[0] = 0
print(t._var)
