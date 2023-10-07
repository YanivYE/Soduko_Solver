import cv2

PATH = "board.png"
SIZE = 450


def prepare_image():
    img = cv2.imread(PATH)
    img = cv2.resize(img, (SIZE, SIZE))
    img_threshold = preprocessing(img)


def preprocessing(img):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grayscale, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 2, 2, 11, 2)
    return img_threshold


