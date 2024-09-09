import numpy as np


class Sudoku:
    def __init__(self, N: int = 3):
        self.N = N
        self.data = np.zeros((N**2, N**2))

    """This function computes the list of available numbers at some position.
    """

    def available_number_list(self, pos):
        return [
            i
            for i in range(1, 10)
            if (
                np.sum(self.data[:, pos[1]] == i) == 0
                and np.sum(self.data[pos[0], :] == i) == 0
                and np.sum(
                    self.data[
                        (pos[0] // self.N) * self.N : (pos[0] // self.N) * self.N
                        + self.N,
                        (pos[1] // self.N) * self.N : (pos[1] // self.N) * self.N
                        + self.N,
                    ]
                    == i
                )
                == 0
            )
        ]

    """ This functions computes a soft solution of the sudoku, without guessing.
    It returns:
        - fill_list : list of the filled positions
        - consistent : bool, True if the sudoku is solvable or undetermined, False if errors are detected (same numbers in the same row, column or quadrant while computing the soft solution).

    """

    def fill_determined_numbers(self):
        fill_list = []
        still_running = True
        while still_running:
            still_running = False
            for empty_coord in np.transpose(np.where(self.data == 0)):
                avl_num_list = self.available_number_list(empty_coord)
                if len(avl_num_list) == 1:
                    self.data[*empty_coord] = avl_num_list[0]
                    fill_list.append(empty_coord)
                    still_running = True
                    break
                elif len(avl_num_list) == 0:
                    return fill_list, False
        return fill_list, True

    """ Computes the complete solution, possibly by guessing.
    Returns:
    True -> solved
    False -> unsolvable
    """

    def solve(self):
        fill_list, consist = self.fill_determined_numbers()
        if not consist:
            for pos in fill_list:  # rollback
                self.data[*pos] = 0
            return False
        if np.sum(self.data == 0) == 0:  # found solution
            return True
        pos = np.transpose(np.where(self.data == 0))[0]
        for num in self.available_number_list(pos):
            print(str(pos) + " -> " + str(num))
            self.data[*pos] = num
            solved = self.solve()
            if solved:
                return True
            else:
                self.data[*pos] = 0
                print(str(pos) + " -> " + str(0))
        return False


if __name__ == "__main__":
    s = Sudoku()
    s.data = np.array(
        [
            [0, 4, 5, 0, 0, 0, 0, 0, 0],
            [8, 3, 0, 0, 0, 7, 4, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 5, 0],
            [0, 8, 4, 6, 0, 0, 5, 0, 1],
            [2, 0, 0, 8, 3, 0, 0, 4, 0],
            [0, 0, 0, 5, 0, 0, 0, 0, 7],
            [3, 7, 0, 0, 0, 5, 0, 6, 0],
            [0, 2, 0, 0, 0, 0, 0, 8, 0],
            [5, 6, 1, 9, 0, 0, 0, 7, 0],
        ]
    )
    print(s.solve())
    print(s.data)
