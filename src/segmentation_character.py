import numpy as np
import cv2

def process_and_compare_images(image_array):
    height, width = image_array.shape
    
    cut_width = width // 4

    top_portion = image_array[:cut_width,:]
    bottom_portion = image_array[height-cut_width:,:]
    mid_point = width // 2
    if width % 2 == 1:
        top_left_half = top_portion[:, :mid_point+1].copy()
        top_right_half = top_portion[:, mid_point:].copy()
    else:
        top_left_half = top_portion[:, :mid_point].copy()
        top_right_half = top_portion[:, mid_point:].copy()
    
    rotated_top_left = np.fliplr(top_left_half)
    white_pixels_left = (rotated_top_left == 255)
    white_pixels_right = (top_right_half == 255)
    
    matching_white_top = np.sum(np.logical_and(white_pixels_left, white_pixels_right))
    total_top_pixels = rotated_top_left.size
    top_white_percentage = (matching_white_top / total_top_pixels) * 100
    
    mid_point = width // 2
    if width % 2 == 1:
        bottom_left_half = bottom_portion[:, :mid_point+1].copy()
        bottom_right_half = bottom_portion[:, mid_point:].copy()
    else:
        bottom_left_half = bottom_portion[:, :mid_point].copy()
        bottom_right_half = bottom_portion[:, mid_point:].copy()
    
    rotated_bottom_left = np.fliplr(bottom_left_half)
    
    white_pixels_left = (rotated_bottom_left == 255)
    white_pixels_right = (bottom_right_half == 255)
    
    matching_white_bottom = np.sum(np.logical_and(white_pixels_left, white_pixels_right))
    total_bottom_pixels = rotated_bottom_left.size
    bottom_white_percentage = (matching_white_bottom / total_bottom_pixels) * 100
    
    def hmean(a, b):
        return 0 if a == 0 or b == 0 else 2 / (1/a + 1/b)
    
    return hmean(top_white_percentage, bottom_white_percentage)

def process_and_compare_images_horizontal(image_array):
    height, width = image_array.shape

    cut_width = width // 4

    left_portion = image_array[:, :cut_width]
    right_portion = image_array[:, width-cut_width:]

    mid_h = height // 2
    if height % 2 == 1:
        left_top = left_portion[:mid_h+1, :].copy()
        left_bottom = left_portion[mid_h:, :].copy()
    else:
        left_top = left_portion[:mid_h, :].copy()
        left_bottom = left_portion[mid_h:, :].copy()

    rotated_left_top = np.flipud(left_top)
    match_left = np.sum(np.logical_and(rotated_left_top == 255, left_bottom == 255))
    left_percent = (match_left / rotated_left_top.size) * 100

    if height % 2 == 1:
        right_top = right_portion[:mid_h+1, :].copy()
        right_bottom = right_portion[mid_h:, :].copy()
    else:
        right_top = right_portion[:mid_h, :].copy()
        right_bottom = right_portion[mid_h:, :].copy()

    rotated_right_top = np.flipud(right_top)

    match_right = np.sum(np.logical_and(rotated_right_top == 255, right_bottom == 255))
    right_percent = (match_right / rotated_right_top.size) * 100

    def hmean(a, b):
        return 0 if a == 0 or b == 0 else 2 / (1/a + 1/b)
    return hmean(left_percent, right_percent)

def trim_black_rows(single_char_image, x, y):
    rows_with_white = np.where(np.any(single_char_image == 255, axis=1))[0]
    img_height, img_width = single_char_image.shape
    if len(rows_with_white) > 0:
        top, bottom = rows_with_white[0], rows_with_white[-1] + 1
        trimmed_img = single_char_image[top:bottom, :]
        bbox = (x, img_width+x-1, top + y, bottom + y)
    else:
        trimmed_img = single_char_image
        bbox = (x, img_width+x-1, y, img_height-1 + y)
    return trimmed_img, bbox

def find_fit_x(binary, coord_x):
    strip = binary[:, coord_x-5:coord_x+4]
    v_sum = np.sum(strip, axis=0)
    smooth_sum = np.convolve(v_sum, np.ones(3), mode='valid')
    min_idx = np.argmin(smooth_sum)
    return coord_x - 4 + min_idx + 1

def character_segmentation(image, mean_w):

    from src.recognition import recognize_char
    
    _, binary = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    merged_contours = []
    skip_indices = set()

    for i in range(len(contours)-1):
        if i in skip_indices:
            continue
        x1, y1, w1, h1 = cv2.boundingRect(contours[i])
        merged = contours[i].copy()

        for j in range(i+1, len(contours)):
            x2, y2, w2, h2 = cv2.boundingRect(contours[j])
            if (x1 <= x2 and x1 + w1 >= x2 + w2) or \
                (abs(x1+w1-x2)<3 and abs(y1-y2)< 3 and w2<w1/2 and 0.5<h1/h2<3 and mean_w/1.5 > w2) or \
                    (w1<mean_w-3 and w2<mean_w-3) or\
                        (x1+w1>x2 and h1==h2): # and mean_pixel>160
                merged = np.vstack((merged, contours[j]))
                skip_indices.add(j)
                x1, y1, w1, h1 = cv2.boundingRect(merged)
            else:
                break
        merged_contours.append(merged)
    if (len(contours)-1) not in skip_indices:
        merged_contours.append(contours[-1])
    
    median_lower_y = np.median([cv2.boundingRect(ctr)[1] for ctr in merged_contours])
    result_chars = []
    bboxes = []
    long_flag = []
    for i in range(len(merged_contours)):
        x, y, w, h = cv2.boundingRect(merged_contours[i])
        # Check if character height is much smaller than the width (indicating possible multi-letter segmentation)
        if w >= 2 and h >= 2:  # Ensure the character is large enough 10
            #if h * 1.2 < w and h * 2 > w:  # If the width is much larger than the height, check the middle column
            
            if mean_w * 1.8 < w < mean_w * 2.8:
                char_img = binary[y:y + h, x:x + w]
                wh_ratio = w / h
                w_ratio = w / mean_w
                mid_x = x + w // 2

                hmean_v = process_and_compare_images(char_img)
                hmean_h = process_and_compare_images_horizontal(char_img)
                min_hmean, max_hmean = min(hmean_v, hmean_h), max(hmean_v, hmean_h)
                
                should_add = (
                    (min_hmean > 40 and max_hmean > 60 and w_ratio < 1.9) or
                    (min_hmean > 30 and max_hmean > 50 and w_ratio < 1.8) or
                    (min_hmean > 20 and max_hmean > 40 and w_ratio < 1.7) or
                    (w_ratio < 1.6) or
                    (median_lower_y - y > 3 and wh_ratio < 1.1 and hmean_v >0)
                )

                if not should_add and wh_ratio < 1.3:
                    pred = recognize_char(char_img, [x,x+w,y,y+h])
                    should_add = (
                        (pred in ['Ф', 'М'] and min_hmean > 20) or
                        (pred == 'Ю' and max_hmean > 50)
                    )

                if should_add and np.sum(char_img[0]) < 2200:
                    char, bbox_y = trim_black_rows(char_img, x, y)
                    result_chars.append(char)
                    bboxes.append(bbox_y)
                    long_flag.append(False)
                else:
                    mid_x = max(find_fit_x(binary, mid_x), x + 7) if min_hmean <= 40 or max_hmean <= 40 else mid_x
                    left_char, left_bbox_y = trim_black_rows(binary[y:y + h, x:mid_x], x, y)
                    right_char, right_bbox_y = trim_black_rows(binary[y:y + h, mid_x:x + w], mid_x, y)
                    result_chars += [left_char, right_char]
                    bboxes += [left_bbox_y, right_bbox_y]
                    long_flag += [False, False]
            else:
                # If the character is not too wide, add the entire character
                char, bbox_y = trim_black_rows(binary[y:y + h, x:x + w], x, y)
                
                if w>= mean_w*2.8:
                    result_chars.append(image[y:y + h, x:x + w])
                    bboxes.append(bbox_y)
                    long_flag.append(True)
                else:
                    result_chars.append(char)
                    bboxes.append(bbox_y)
                    long_flag.append(False)
    return result_chars, bboxes, long_flag
