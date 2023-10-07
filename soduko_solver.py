print('Setting UP')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from keras import models

PATH = "board.png"
SIZE = 450
RESOLUTION = 28


def initialize_cnn_model():
    loaded_model = models.load_model('my_model.keras')
    print("Model loaded.")
    return loaded_model


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
        cv2.drawContours(img_big_contour, biggest_contour, -1, (0, 255, 0), 10)
        pts1 = np.float32(biggest_contour)
        pts2 = np.float32([[0, 0], [SIZE, 0], [0, SIZE], [SIZE, SIZE]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp_colored = cv2.warpPerspective(img, matrix, (SIZE, SIZE))
        img_warp_colored = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
        return img_warp_colored
    return None


def find_biggest_contour(contours):
    biggest_contour = np.array([])
    max_area = 0
    for con in contours:
        area = cv2.contourArea(con)
        if area > 50:
            p = cv2.arcLength(con, True)
            approx = cv2.approxPolyDP(con, 0.02 * p, True)
            if area > max_area and len(approx) == 4:
                biggest_contour = approx
                max_area = area
    return biggest_contour, max_area


def reorder(points_arr):
    points_arr = points_arr.reshape((4, 2))

    new_points_arr = np.zeros((4, 1, 2), dtype=np.int32)
    add = points_arr.sum(1)
    new_points_arr[0] = points_arr[np.argmin(add)]
    new_points_arr[3] = points_arr[np.argmax(add)]
    diff = np.diff(points_arr, axis=1)
    new_points_arr[1] = points_arr[np.argmin(diff)]
    new_points_arr[2] = points_arr[np.argmax(diff)]
    return new_points_arr


def classify_digits(board, cnn_model):
    boxes = split_boxes(board)
    # digits = predict(boxes, cnn_model)


def split_boxes(board):
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def predict(boxes, model):
    result = []
    for box in boxes:
        img = prepare_image_box(box)
        prediction = model.predict(img)
        class_index = np.argmax(prediction, axis=1)
        prob_value = np.amax(prediction)
        print(class_index, prob_value)
        result = generate_result(result, prob_value, class_index)
    return result


def prepare_image_box(box):
    img = np.asarray(box)
    img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
    img = cv2.resize(img, (RESOLUTION, RESOLUTION)).astype(np.float32) / 255.0
    img = img.reshape(1, RESOLUTION, RESOLUTION, 1)
    return img



def generate_result(result, prob_value, class_index):
    if prob_value > 0.8:
        result.append(class_index[0])
    else:
        result.append(0)
    return result


def main():
    model = initialize_cnn_model()
    img = cv2.resize(cv2.imread(PATH), (SIZE, SIZE))
    preprocessed_img = preprocess_image(img)
    board = define_contours(img, preprocessed_img)
    classify_digits(board, model)


if __name__ == '__main__':
    main()
