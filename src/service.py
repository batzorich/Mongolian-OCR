import logging
import time
from src.skew_correction import skew_correct
from src.adjust_height_width import adjust_height_width
from src.segmentation_contour import segment_contours
from src.segmentation_line import segment_lines
from src.recognition import recognize_line
from src.post_process import post_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text(img):
    start_time = time.time()
    try:
        skew_corrected_img = skew_correct(img)
        resized_img, (median_h_lower, _, median_h_upper, mean_w, rescale_idx) = adjust_height_width(skew_corrected_img)
        
        contour_images = segment_contours(resized_img)
        line_images_all = []
        for c in contour_images:
            line_images = segment_lines(c)
            if line_images:
                for line in line_images:
                    if line.shape[0] > 10:
                        line_images_all.append(line)

        result_full_text = ""
        for line in line_images_all:
            result_line_text = recognize_line(line, rescale_idx, mean_w, median_h_upper, median_h_lower)
            result_full_text += result_line_text + "\n"

        final_result = post_process(result_full_text)
        logger.info(f"Extraction succeeded in {time.time() - start_time:.2f}s")
        return final_result

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.info(f"Execution time before failure: {time.time() - start_time:.2f}s")
        return None
