# import the necessary packages
import Extract_Puzzle
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2


def initialize_cell_locations(board, cell_locs, warped, model, stepX, stepY):
    # loop over the grid locations
    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))

            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            digit = Extract_Puzzle.extract_digit(cell)

            # verify that the digit is not empty
            if digit is not None:
                roi = prepare_cell(digit)
                # classify the digit and update the Sudoku board with the
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]
                # print(pred)
                board[y, x] = pred

        # add the row to our cell locations
        cell_locs.append(row)
    return board, cell_locs


def prepare_cell(cell):
    # resize the cell to 28x28 pixels and then prepare the
    # cell for classification
    roi = cv2.resize(cell, (28, 28))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi


def classify_board(img_path):
    # load the digit classifier from disk
    print("[INFO] loading digit classifier...")
    model = load_model('my_model.keras')

    # load the input image from disk and resize it
    print("[INFO] processing image...")
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=600)

    # find the puzzle in the image and then
    (puzzleImage, warped) = Extract_Puzzle.define_board(image)
    # initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int")

    # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    step_X = warped.shape[1] // 9
    step_Y = warped.shape[0] // 9
    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cell_locs = []

    # cell locs for version 3.0.0
    board, cell_locs = initialize_cell_locations(board, cell_locs, warped, model, step_X, step_Y)

    board = np.matrix(board)
    board = board.tolist()
    return board
