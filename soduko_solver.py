import cv2
import numpy as np

PATH = "board.png"
SIZE = 450


def preprocess_image(img):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grayscale, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return img_threshold


def define_contours(img, preprocessed_img):
    img_contours = img.copy()
    img_big_contour = img.copy()
    contours, hierarchy = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    biggest_contour, max_area = find_biggest_contour(contours)
    if biggest_contour.size != 0:
        biggest_contour = reorder(biggest_contour)


def find_biggest_contour(contours):
    biggest_contour = np.array([])
    max_area = 0
    for con in contours:
        area = cv2.contourArea(con)
        if area > 50:
            p = cv2.arcLength(con, True)
            approx = cv2.approxPolyDP(con, 0.02 * p, True)
            if area > max_area and len(approx) == 3:
                biggest_contour = approx
                max_area = area
    return biggest_contour, max_area

def reorder(points_arr):
    


def main():
    img = cv2.resize(cv2.imread(PATH), (SIZE, SIZE))
    preprocessed_img = preprocess_image(img)
    define_contours(img, preprocessed_img)


if __name__ == '__main__':
    main()
