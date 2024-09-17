import numpy as np


class Knight:
    def __init__(self, path):
        self.path = np.copy(path)

    def can_move(self, pos):
        return not any(np.array_equal(row, pos) for row in self.path)

    def move(self, pos):
        self.path = np.append(self.path, pos)


MOVES = np.array(
    [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [-2, 1], [2, -1], [-2, -1]]
)

if __name__ == "__main__":
    map_shape = np.array([5, 5])
    init_pos = np.array([0, 0])
    knights = list()
    knights.append(Knight(np.array(init_pos)))
    for index in range(map_shape[0] * map_shape[1]):
       for k in knights:
            new_positions = np.repeat([init_pos], axis=0, repeats=8)+MOVES
            for p in new_positions:
                if p[0]>=0 and p[1]>=0 and p[0]<map_shape[0] and p[1]<map_shape[1]:
                    
