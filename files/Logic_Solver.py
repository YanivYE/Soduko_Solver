from Classifier import classify_board
import numpy as np

BOARD_IMG_PATH = "../boards/board2.png"

puzzle = classify_board(BOARD_IMG_PATH)
solutions = []


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


def solve():
    for row in range(9):
        for col in range(9):
            if puzzle[row][col] == 0:
                for num in range(1, 10):
                    if is_possible(row, col, num):
                        puzzle[row][col] = num
                        solve()
                        puzzle[row][col] = 0
                return

    solutions.append(np.matrix(puzzle))


def main():
    solve()
    if len(solutions) > 1:
        print("More than one possible solution!\n")
    print(solutions)


if __name__ == '__main__':
    main()
