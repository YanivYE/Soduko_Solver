# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

DEBUG = True


def define_board(image):
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # check to see if we are visualizing each step of the image
    # processing pipeline (in this case, thresholding)
    if DEBUG:
        cv2.imshow("Board Thresh", thresh)
        cv2.waitKey(0)

    board_contour = find_board_contour(thresh, image)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    board = four_point_transform(image, board_contour.reshape(4, 2))
    warped = four_point_transform(gray, board_contour.reshape(4, 2))

    # check to see if we are visualizing the perspective transform
    if DEBUG:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Board Transform", board)
        cv2.waitKey(0)
    # return a 2-tuple of puzzle in both RGB and grayscale
    return board, warped


def find_board_contour(thresh_img, image):
    # initialize a contour that corresponds to the puzzle outline
    board_cnt = None

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            board_cnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if board_cnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    if DEBUG:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [board_cnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    return board_cnt


def extract_digit(cell):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    # check to see if we are visualizing the cell thresholding step
    if DEBUG:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
