import numpy as np


class Knight:
    def __init__(self, pos: np.ndarray):
        self.messages = np.array(
            [], dtype=np.int_
        )  # should be a list of arrays, each one is a single message that is enriched with current position and resent to the neighbors
        self.position = pos

    def receive_message(self, m):
        if not any(np.array_equal(m, x) for x in self.messages):
            self.messages = np.append(self.messages, m)
            print("added")
        else:
            print("discarded")


def build_path(shape: np.ndarray, pos0: np.ndarray):
    knight_map = [
        [Knight(np.array([i, j])) for i in range(shape[0] + 4)]
        for j in range(shape[1] + 4)
    ]
    knight_map[pos0[0]][pos0[1]].receive_message(pos0)
    for iteration in range(shape[0] * shape[1]):
        new_map = knight_map.copy()
        for i in range(shape[0]):
            for j in range(shape[1]):
                knight_map[i + 1][j].receive_message(knight_map[i + 2][j + 2].messages)
                knight_map[i][j + 1].receive_message(knight_map[i + 2][j + 2].messages)
                knight_map[i][j + 3].receive_message(knight_map[i + 2][j + 2].messages)
                knight_map[i + 3][j].receive_message(knight_map[i + 2][j + 2].messages)
                knight_map[i + 4][j + 1].receive_message(
                    knight_map[i + 2][j + 2].messages
                )
                knight_map[i + 1][j + 4].receive_message(
                    knight_map[i + 2][j + 2].messages
                )
                knight_map[i + 4][j + 3].receive_message(
                    knight_map[i + 2][j + 2].messages
                )
                knight_map[i + 3][j + 4].receive_message(
                    knight_map[i + 2][j + 2].messages
                )
        knight_map = new_map

    print("stop here")


if __name__ == "__main__":
    build_path(np.array([10, 10]))
