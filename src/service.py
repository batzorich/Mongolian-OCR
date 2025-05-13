import logging
import time

from src.skew_correction import skew_correct
from src.adjust_height_width import adjust_height_width
from src.segmentation_combined import segment
from src.recognition import recognize_both
from src.post_process import post_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text(img):
    start_time = time.time()
    skew_corrected_img = skew_correct(img)
    resized_img, (median_h_lower, _, median_h_upper, mean_w, rescale_idx) = adjust_height_width(skew_corrected_img)
    
    stack_chars_normal, stack_chars_long, list_bboxes_normal, list_height_threshold, list_lower_threshold, list_median_lower_y, \
        list_bboxes_long, list_length_words_full = segment(resized_img, rescale_idx, mean_w, median_h_lower)

    recognized_text = recognize_both(stack_chars_normal, stack_chars_long, list_bboxes_normal, list_height_threshold, list_lower_threshold, \
                    list_median_lower_y, median_h_lower, median_h_upper, list_bboxes_long, list_length_words_full)
    postprocessed_text = post_process(recognized_text)
    
    logger.info(f"Extraction succeeded in {time.time() - start_time:.2f}s")
    return postprocessed_text
    