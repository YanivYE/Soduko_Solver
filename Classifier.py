# import the necessary packages
import Extract_Puzzle
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2

BOARD_IMG_PATH = "boards/board2.png"

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model('my_model.keras')

# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(BOARD_IMG_PATH)
image = imutils.resize(image, width=600)

# find the puzzle in the image and then
(puzzleImage, warped) = Extract_Puzzle.define_board(image)
# initialize our 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int")
# a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped image
# into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9
# initialize a list to store the (x, y)-coordinates of each cell
# location
cell_locs = []
