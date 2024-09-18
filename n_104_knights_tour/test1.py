import numpy as np
from tqdm import tqdm


class Path:
    def __init__(self, p=None):
        self.path = list()
        if p is not None:
            for pos in p:
                self.path.append(np.copy(pos))

    def contains(self, pos: np.ndarray):
        return any(np.array_equal(pos, arr) for arr in self.path)

    def append_pos(self, pos: np.ndarray):
        self.path.append(pos)


class Knight:
    def __init__(self, pos: np.ndarray):
        self.path_list = list()
        self.position = pos

    def receive_paths(self, paths):
        for path in paths:
            if not path.contains(self.position):
                path.append_pos(self.position)
                self.path_list.append(path)

    def copy_knight(self):
        cp = Knight(self.position)
        for p in self.path_list:
            cp.path_list.append(Path(p.path))
        return cp


def build_path(shape: np.ndarray, pos0: np.ndarray):
    knight_map = [
        [Knight(np.array([i + 2, j + 2])) for i in range(shape[0] + 4)]
        for j in range(shape[1] + 4)
    ]
    path0 = Path()
    path0.path.append(np.array(pos0))
    knight_map[pos0[0]][pos0[1]].receive_paths([path0])
    for iteration in tqdm(range(shape[0] * shape[1])):
        copy_map = [
            [knight_map[i][j].copy_knight() for i in range(shape[0] + 4)]
            for j in range(shape[1] + 4)
        ]
        for i in range(shape[0]):
            for j in range(shape[1]):
                copy_map[i + 1][j].receive_paths(knight_map[i + 2][j + 2].path_list)
                copy_map[i][j + 1].receive_paths(knight_map[i + 2][j + 2].path_list)
                copy_map[i][j + 3].receive_paths(knight_map[i + 2][j + 2].path_list)
                copy_map[i + 3][j].receive_paths(knight_map[i + 2][j + 2].path_list)
                copy_map[i + 4][j + 1].receive_paths(knight_map[i + 2][j + 2].path_list)
                copy_map[i + 1][j + 4].receive_paths(knight_map[i + 2][j + 2].path_list)
                copy_map[i + 4][j + 3].receive_paths(knight_map[i + 2][j + 2].path_list)
                copy_map[i + 3][j + 4].receive_paths(knight_map[i + 2][j + 2].path_list)
        knight_map = copy_map

    array = np.array(knight_map[pos0[0]][pos0[1]].path_list[0].path)
    print(array)


if __name__ == "__main__":
    build_path(np.array([4, 4]), np.array([2, 2]))
