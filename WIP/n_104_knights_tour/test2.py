import numpy as np
from tqdm import tqdm


class Knight:
    def __init__(self, path):
        self.path = np.copy(path)

    def can_move(self, pos):
        return not any(np.array_equal(row, pos) for row in self.path[:-1])

    def closed(self, pos):
        return self.path.shape[0] and np.array_equal(self.path[-1], pos)


MOVES = np.array(
    [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [-2, 1], [2, -1], [-2, -1]]
)

if __name__ == "__main__":
    map_shape = np.array([5, 3])
    init_pos = np.array([[0, 0]])
    knights = list()
    knights.append(Knight(np.array(init_pos)))  # initial knight
    solutions = list()
    for index in tqdm(range(map_shape[0] * map_shape[1] - 1)):
        new_knights = list()
        for k in knights:
            new_positions = np.repeat([k.path[-1]], axis=0, repeats=8) + MOVES
            for p in new_positions:
                if (
                    p[0] >= 0
                    and p[1] >= 0
                    and p[0] < map_shape[0]
                    and p[1] < map_shape[1]
                    and k.can_move(p)
                ):
                    new_knights.append(Knight(np.vstack((k.path, p))))
                if index == map_shape[0] * map_shape[1] - 2 and np.array_equal(
                    k.path[0], p
                ):
                    solutions.append(np.vstack((k.path, p)))

        knights = new_knights.copy()

    for s in solutions:
        print(s)
    print("stop here")
