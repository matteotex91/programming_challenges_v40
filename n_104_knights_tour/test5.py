import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

MOVES = np.array(
    [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [-2, 1], [2, -1], [-2, -1]]
)

# hard research over the tree


def find_path(map_shape, path1, path2):
    if np.array_equal(path1[-1], path2[-1]):
        if len(path1) * 2 == map_shape[0] * map_shape[1] + 1:
            return True  # solution found
        elif len(path1) > 1:
            return False
    moves1 = list(np.repeat([path1[-1]], axis=0, repeats=8) + MOVES)
    moves2 = list(np.repeat([path2[-1]], axis=0, repeats=8) + MOVES)
    shuffle(moves1)
    shuffle(moves2)
    for newpos1 in moves1:
        if (
            newpos1[0] < 0
            or newpos1[0] >= map_shape[0]
            or newpos1[1] < 0
            or newpos1[1] >= map_shape[1]
            or any(np.array_equal(newpos1, pos) for pos in path1 + path2)
        ):
            continue
        for newpos2 in moves2:
            if (
                newpos2[0] < 0
                or newpos2[0] >= map_shape[0]
                or newpos2[1] < 0
                or newpos2[1] >= map_shape[1]
                or any(np.array_equal(newpos2, pos) for pos in path1 + path2)
            ):
                continue
            path1.append(newpos1)
            path2.append(newpos2)
            print(len(path1))
            if find_path(map_shape, path1, path2):
                return True
            else:
                path1.pop()
                path2.pop()
    return False


if __name__ == "__main__":
    map_shape = [6, 6]
    path1 = list()
    path2 = list()
    path1.append(np.array([0, 0]))
    path2.append(np.array([0, 0]))

    find_path(map_shape, path1, path2)
    path1 = np.array(path1)
    path2 = np.array(path2)
    plt.plot(path1[:, 0], path1[:, 1], color="blue")
    plt.plot(path2[:, 0], path2[:, 1], color="red")
    plt.xticks(np.arange(0, map_shape[1], 1))
    plt.yticks(np.arange(0, map_shape[0], 1))
    plt.grid(True)
    plt.show()
    print("stop here")
