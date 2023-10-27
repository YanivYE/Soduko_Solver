from Logic_Solver import Solver

BOARD_IMG_PATH = "../boards/board2.png"


def main():
    my_solver = Solver(BOARD_IMG_PATH)
    my_solver.solve()
    puzzle_solutions = my_solver.solutions
    if len(puzzle_solutions) > 1:
        print("More than one possible solution!\n")
    print(puzzle_solutions)


if __name__ == '__main__':
    main()
