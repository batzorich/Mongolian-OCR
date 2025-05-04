import cv2
import numpy as np
from collections import Counter

def character_segmentation_light(image):
    
    _, binary = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    merged_contours = []
    bboxes = []
    skip_indices = set()
    for i, c1 in enumerate(contours):
        if i in skip_indices:
            continue
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        merged = False
        
        for j, c2 in enumerate(contours[i+1:], start=i+1):
            if j in skip_indices:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(c2)
            if (x1 <= x2 and x1 + w1 >= x2 + w2): # and mean_pixel>160
                merged_contour = np.vstack((c1, c2))
                merged_contours.append(merged_contour)
                skip_indices.add(j)
                merged = True
                break
        
        if not merged:
            merged_contours.append(c1)
    return merged_contours

def find_letter_ratio(image):
    contours = character_segmentation_light(image)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    height_width_list = [(cv2.boundingRect(contour)[2], cv2.boundingRect(contour)[3]) for contour in contours]
    
    return height_width_list

def get_width_height_ratio(img):

    list_ratio = find_letter_ratio(img)
    
    counter_lower = Counter(list_ratio)
    median_lower = counter_lower.most_common(1)[0]
    median_h_lower = median_lower[0][1]
    median_w_lower = median_lower[0][0]

    counter_upper = Counter((x, y) for x, y in list_ratio if x > median_w_lower + 2 and y > median_h_lower + 2)
    median_upper = counter_upper.most_common(1)[0]
    median_h_upper = median_upper[0][1]
    median_w_upper = median_upper[0][0]

    rescale_idx = (median_w_lower/median_h_lower)/(8/10)
    mean_w_lower = int(median_w_lower/ rescale_idx)
    mean_w_upper = int(median_w_upper/ rescale_idx)
    
    return (median_h_lower, median_w_lower, median_h_upper, mean_w_lower, rescale_idx)

def adjust_height_width(img):
    median_h_lower_raw, median_w_lower_raw, _, _, _ = get_width_height_ratio(img)
    
    new_height = int(img.shape[0] / (median_h_lower_raw/12))
    new_width = int(img.shape[1] / (median_w_lower_raw/10)) 
    resized_img = cv2.resize(img, (new_width, new_height))

    return resized_img, get_width_height_ratio(resized_img)