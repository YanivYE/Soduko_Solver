from Classifier import classify_board
import numpy as np

puzzle = classify_board()

def is_possible(row, col, num):
    for i in range(9):
        if puzzle[row][i] == num:
            return False

    for i in range(9):
        if puzzle[i][col] == num:
            return False

    return not_in_square(row, col, num)


def not_in_square(row, col, num):
    x0 = (col // 3) * 3
    y0 = (row // 3) * 3
    for i in range(3):
        for j in range(3):
            if puzzle[y0 + i][x0 + j] == num:
                return False
    return True



def main():


if __name__ == '__main__':
    main()
