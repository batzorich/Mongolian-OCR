import numpy as np
import cv2

def character_segmentation(image, visual):
    kernel = np.ones((1, 3), np.uint8)
    dilated = cv2.erode(image, kernel, iterations=2)
    
    denoised = cv2.fastNlMeansDenoising(dilated, None, h=30, templateWindowSize=7, searchWindowSize=21)
    
    _, binary = cv2.threshold(denoised, 170, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    result_chars = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Check if character height is much smaller than the width (indicating possible multi-letter segmentation)
        if w > 10 and h > 10:  # Ensure the character is large enough
            if h * 1.2 < w and h * 2 > w:  # If the width is much larger than the height, check the middle column
                mid_x = x + w // 2
                
                middle_column_sum = np.sum(binary[y:y + h, mid_x])
                # If the sum of pixels in the middle column is less than the threshold, split the character
                if middle_column_sum < h*255*0.3:
                    left_char = binary[y:y + h, x:mid_x]
                    right_char = binary[y:y + h, mid_x:x + w]

                    result_chars.append(left_char)
                    result_chars.append(right_char)
                else:
                    # If the middle column has enough pixels, keep the entire character
                    char = binary[y:y + h, x:x + w]
                    result_chars.append(char)
            elif h * 2 < w:  # If width is much larger than height, split into 3 parts
                third_x1 = x + w // 3
                third_x2 = x + 2 * w // 3

                left_char = binary[y:y + h, x:third_x1]
                middle_char = binary[y:y + h, third_x1:third_x2]
                right_char = binary[y:y + h, third_x2:x + w]

                result_chars.append(left_char)
                result_chars.append(middle_char)
                result_chars.append(right_char)
            else:
                # If the character is not too wide, add the entire character
                char = binary[y:y + h, x:x + w]
                result_chars.append(char)
    return result_chars
