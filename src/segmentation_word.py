import cv2
import numpy as np

def thresholding(image, inv=True):
    if inv:
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return thresh

def crop_line(line_image):
    thresh = thresholding(line_image)
    binary = line_image > thresh
    vertical_projection = np.sum(binary, axis=0)
    height = line_image.shape[0]
    index = 0
    while vertical_projection[index] == height:
        index += 1
    if index > 2:
        index -= 2
    line_image = line_image[:, index:]
    
    thresh = thresholding(line_image)
    binary = line_image > thresh
    vertical_projection = np.sum(binary, axis=0)
    
    index = line_image.shape[1] - 1
    while vertical_projection[index] == height:
        index -= 1
    if index < line_image.shape[1] - 1:
        index += 2
    
    line_image = line_image[:, :index]
    return line_image

def segment_words(line_image):
    line = crop_line(line_image)
    dst = cv2.fastNlMeansDenoising(line, None, 12, 7, 21)
    thresh = thresholding(dst)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1) 
    binary = line > dilated
    vertical_projection = np.sum(binary, axis=0)
    
    height = line.shape[0]
    whitespace_lengths = []
    current_whitespace = 0
    for vp in vertical_projection:
        if vp == height:
            current_whitespace += 1
        elif current_whitespace:
            whitespace_lengths.append(current_whitespace)
            current_whitespace = 0

    if current_whitespace:
        whitespace_lengths.append(current_whitespace)

    avg_white_space_length = min(np.mean(whitespace_lengths)*2, 5) if whitespace_lengths else 0
    divider_indexes = [0]
    current_whitespace = 0
    for index, vp in enumerate(vertical_projection):
        if vp == height:
            current_whitespace += 1
        else:
            if current_whitespace > avg_white_space_length:
                divider_indexes.append(index - current_whitespace // 2)
            current_whitespace = 0
    
    divider_indexes.append(len(vertical_projection))

    dividers = np.column_stack((divider_indexes[:-1], divider_indexes[1:]))
    
    words = [line[:, window[0]:window[1]] for window in dividers]

    return words