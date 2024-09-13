import numpy as np
from time import time
from random import randint


class Sudoku:
    def __init__(
        self,
        N: int = 3,
        random_fill: bool = False,
        random_erase: bool = False,
        guessing_depth: int = 0,
    ):
        self.N = N
        self.data = np.zeros((N**2, N**2))
        if random_fill:
            self.data[0, 0] = randint(0, 9)
            self.solve_randomized()
            if random_erase:
                depth = 0
                erasing = True
                while erasing:
                    erasing = False
                    full_positions = np.transpose(np.where(self.data != 0))
                    pos = full_positions[randint(0, N**3) % len(full_positions)]
                    symm_pos = np.array([8, 8]) - pos
                    cp_s = self.copy()
                    cp_s.data[*pos] = 0
                    cp_s.data[*symm_pos] = 0
                    depth, _ = cp_s.guess_count()
                    if depth <= guessing_depth:
                        erasing = True
                        self.data[*pos] = 0
                        self.data[*symm_pos] = 0

    def copy(self):
        s = Sudoku()
        s.data = np.copy(self.data)
        return s

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
        still_running = True  # This flag is needed to iterate the soft solve until no more numbers are found
        while still_running:
            still_running = False
            for empty_coord in np.transpose(np.where(self.data == 0)):
                avl_num_list = self.available_number_list(empty_coord)
                if len(avl_num_list) == 1:  # if only one number available, insert it
                    self.data[*empty_coord] = avl_num_list[0]
                    fill_list.append(empty_coord)
                    still_running = True  # trigger soft solution again and restart
                    break
                elif len(avl_num_list) == 0:  # error detection for rollback
                    return fill_list, False
        return fill_list, True

    """ Computes the complete solution, possibly by guessing.
    Returns:
    True -> solved
    False -> unsolvable
    """

    def solve_sequentially(self):
        fill_list, consist = self.fill_determined_numbers()  # try soft solution
        if not consist:  # error detected, rollback erasing cells and return False
            for pos in fill_list:
                self.data[*pos] = 0
            return False
        if np.sum(self.data == 0) == 0:  # solution found, return True
            return True
        pos = np.transpose(np.where(self.data == 0))[0]  # pick the first empty cell
        for num in self.available_number_list(pos):  # cycle over the available numbers
            print(str(pos) + " -> " + str(num))
            self.data[*pos] = num  # set the number in the cell
            solved = self.solve_sequentially()  # go recursively
            if solved:  # if solution found, go back with True
                return True
            else:  # otherwise, rollback and continue with the cycle
                self.data[*pos] = 0
                print(str(pos) + " -> " + str(0))
        return False  # if none of the available numbers is ok, then it's unsolvable

    def guess_count(self):
        fill_list, consist = self.fill_determined_numbers()  # try soft solution
        if not consist:  # error detected, rollback erasing cells and return False
            for pos in fill_list:
                self.data[*pos] = 0
            return 0, False
        if np.sum(self.data == 0) == 0:  # solution found, return True
            return 0, True
        undetermine_positions = np.transpose(np.where(self.data == 0))
        available_numbers_count = [
            len(self.available_number_list(pos)) for pos in undetermine_positions
        ]
        pos = [
            x
            for _, x in sorted(
                zip(available_numbers_count, undetermine_positions.tolist())
            )
        ][
            0
        ]  # pos with least available numbers
        for num in self.available_number_list(pos):  # cycle over the available numbers
            print(str(pos) + " -> " + str(num))
            self.data[*pos] = num  # set the number in the cell
            depth, solved = self.guess_count()  # go recursively
            if solved:  # if solution found, go back with True
                return depth + 1, True
            else:  # otherwise, rollback and continue with the cycle
                self.data[*pos] = 0
                print(str(pos) + " -> " + str(0))
        return 0, False  # if none of the available numbers is ok, then it's unsolvable

    def solve_randomized(self):
        fill_list, consist = self.fill_determined_numbers()  # try soft solution
        if not consist:  # error detected, rollback erasing cells and return False
            for pos in fill_list:
                self.data[*pos] = 0
            return False
        if np.sum(self.data == 0) == 0:  # solution found, return True
            return True
        free_positions = np.transpose(np.where(self.data == 0))

        pos = free_positions[
            randint(0, len(free_positions) - 1)
        ]  # pick the first empty cell
        for num in self.available_number_list(pos):  # cycle over the available numbers
            print(str(pos) + " -> " + str(num))
            self.data[*pos] = num  # set the number in the cell
            solved = self.solve_randomized()  # go recursively
            if solved:  # if solution found, go back with True
                return True
            else:  # otherwise, rollback and continue with the cycle
                self.data[*pos] = 0
                print(str(pos) + " -> " + str(0))
        return False  # if none of the available numbers is ok, then it's unsolvable


if __name__ == "__main__":

    # s = Sudoku()
    # s.data = np.array(
    #     [
    #         [0, 4, 5, 0, 0, 0, 0, 0, 0],
    #         [8, 3, 0, 0, 0, 7, 4, 0, 0],
    #         [0, 0, 0, 2, 0, 0, 0, 5, 0],
    #         [0, 8, 4, 6, 0, 0, 5, 0, 1],
    #         [2, 0, 0, 8, 3, 0, 0, 4, 0],
    #         [0, 0, 0, 5, 0, 0, 0, 0, 7],
    #         [3, 7, 0, 0, 0, 5, 0, 6, 0],
    #         [0, 2, 0, 0, 0, 0, 0, 8, 0],
    #         [5, 6, 1, 9, 0, 0, 0, 7, 0],
    #     ]
    # )
    # print(s.guess_count())
    # print(s.data)

    print(Sudoku(random_fill=True, random_erase=True, guessing_depth=10).data)
