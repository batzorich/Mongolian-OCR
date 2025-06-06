import cv2
import numpy as np
from imutils import rotate

def horizontal_projection_skew(sobel_img):
    sum_cols = []
    rows, cols = sobel_img.shape
    for row in range(rows-1):
        sum_cols.append(np.sum(sobel_img[row,:]))
    return sum_cols

def skew_correct(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel_img = np.sqrt(gx * gx + gy * gy)
    sobel_img_inv = 255 - sobel_img
    correction_angle = 0
    highest_hp = 0
    for angle in np.arange(-5, 5.1, 0.1):
        hp = horizontal_projection_skew(rotate(sobel_img_inv, angle))
        median_hp = np.median(hp)
        if highest_hp < median_hp:
            correction_angle = angle
            highest_hp = median_hp
    rows = img.shape[0]
    cols = img.shape[1]
    img_center = (cols / 2, rows / 2)
    matrix = cv2.getRotationMatrix2D(img_center, correction_angle, 1)
    rotated_img = cv2.warpAffine(img, matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
    return rotated_img