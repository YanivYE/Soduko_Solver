from Classifier import classify_board
import numpy as np


class Solver:
    solutions = []
    puzzle = []

    def __init__(self, path):
        self.path = path
        self.puzzle = classify_board(path)

    def is_possible(self, row, col, num):
        for i in range(9):
            if self.puzzle[row][i] == num:
                return False

        for i in range(9):
            if self.puzzle[i][col] == num:
                return False

        return self.not_in_square(row, col, num)

    def not_in_square(self, row, col, num):
        x0 = (col // 3) * 3
        y0 = (row // 3) * 3
        for i in range(3):
            for j in range(3):
                if self.puzzle[y0 + i][x0 + j] == num:
                    return False
        return True

    def solve(self):
        for row in range(9):
            for col in range(9):
                if self.puzzle[row][col] == 0:
                    for num in range(1, 10):
                        if self.is_possible(row, col, num):
                            self.puzzle[row][col] = num
                            self.solve()
                            self.puzzle[row][col] = 0
                    return
        self.solutions.append(np.matrix(self.puzzle))
