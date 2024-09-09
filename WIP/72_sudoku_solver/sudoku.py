import numpy as np


class Sudoku:
    def __init__(self, N: int = 3):
        self.N = N
        self.data = np.zeros((N**2, N**2))

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

    """ this functions fills the map with the only numbers which are univocally determined.
    It returns:
        - fill_list : list of the filled positions
        - consistend : bool, True if the sudoku is consistent, False if not.

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

    """ looks for the complete solution
    1- look for univocally determined numbers
    2- if not completed, looks for the most determined cell (the one with the least of available numbers)
    3- cycle over all the available numbers. If inconsistency detected, or it's impossible to determine any solution, roll back
    return values:
    True -> solved
    False -> inconsistent
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
        num_list_at_pos = self.available_number_list(pos)
        for num in num_list_at_pos:
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
    # s.data = np.array(
    #    [
    #        [0, 7, 0, 5, 8, 3, 0, 2, 0],
    #        [0, 5, 9, 2, 0, 0, 3, 0, 0],
    #        [3, 4, 0, 0, 0, 6, 5, 0, 7],
    #        [7, 9, 5, 0, 0, 0, 6, 3, 2],
    #        [0, 0, 3, 6, 9, 7, 1, 0, 0],
    #        [6, 8, 0, 0, 0, 2, 7, 0, 0],
    #        [9, 1, 4, 8, 3, 5, 0, 7, 6],
    #        [0, 3, 0, 7, 0, 1, 4, 9, 5],
    #        [5, 6, 7, 4, 2, 9, 0, 1, 3],
    #    ]
    # )
    # fill_list, consist = s.fill_determined_numbers()
    # print(consist)
    s.data = np.array(
        [
            [0, 0, 0, 0, 0, 3, 0, 2, 0],
            [0, 0, 9, 2, 0, 0, 3, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 7],
            [0, 0, 5, 0, 0, 0, 0, 3, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    print(s.solve())
    print(s.data)
