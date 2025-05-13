import numpy as np

from src.segmentation_contour import segment_contours
from src.segmentation_line import segment_lines
from src.segmentation_word import segment_words
from src.segmentation_character import character_segmentation
from src.recognition import preprocess_tf
from src.utils import reshape_image, count_consecutive

def segment(resized_img, rescale_idx, mean_w, median_h_lower):
    contour_images = segment_contours(resized_img)
    list_length_words_full = []
    list_height_threshold = []
    list_lower_threshold = []
    list_median_lower_y = []
    list_bboxes_normal = []
    list_bboxes_long = []
    stack_chars_normal = []
    stack_chars_long = []
    for c in contour_images:
        line_images = segment_lines(c)
        if line_images:
            for line in line_images:
                if line.shape[0] > 10:
                    line_height, _ = line.shape
                    height_treshold = line_height / 1.6
                    lower_threshold = line_height / 8  # New condition
                    words = segment_words(line)

                    lower_letter_top_y = []
                    list_length_words = []
                    for word in words:
                        characters, bboxes, long_flag = character_segmentation(reshape_image(word, rescale_idx), mean_w) 
                        list_length_words.append(count_consecutive(long_flag))
                        for box, flag in zip(bboxes,long_flag):
                            if abs(box[3] - box[2] - median_h_lower)< 2 and not flag:
                                lower_letter_top_y.append(box[2])
                        median_lower_y = np.median(lower_letter_top_y) if lower_letter_top_y else 0

                        for character, bbox, flag in zip(characters, bboxes, long_flag):
                            if flag == False:
                                preprocessed_img = preprocess_tf(character, bbox)
                                stack_chars_normal.append((preprocessed_img, bbox[3]-bbox[2]))
                                list_height_threshold.append(height_treshold)
                                list_lower_threshold.append(lower_threshold)
                                list_median_lower_y.append(median_lower_y)
                                list_bboxes_normal.append(bbox)
                            else:
                                stack_chars_long.append(np.stack([character]*3, axis=-1))
                                list_bboxes_long.append(bbox)
                    list_length_words_full.append(list_length_words)
    return stack_chars_normal, stack_chars_long, list_bboxes_normal, list_height_threshold, list_lower_threshold, list_median_lower_y, \
        list_bboxes_long, list_length_words_full