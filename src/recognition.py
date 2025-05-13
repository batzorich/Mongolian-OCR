import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from src import recognition_word

MODEL_PATH = 'model/recognition_model.h5'
MODEL_CHAR = tf.keras.models.load_model(MODEL_PATH)

ENCODER_PATH = 'model/encoder.npy'
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.classes_ = np.load(ENCODER_PATH, allow_pickle=True)

MODEL_WORD = recognition_word.Recognition('model/recognition_model_word.onnx')

def preprocess_char_image(img: np.ndarray, bbox, size=28):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orig_height, orig_width = img.shape
    if (orig_height > 7 or orig_width > 7) or (orig_width / orig_height>1.7):
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        char_crop = img[y:y+h, x:x+w]
        
        scale = min(size / max(w, h, 0.001),2)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        resized = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        padded = np.full((size, size), 0, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    else:
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        char_crop = img[y:y+h, x:x+w]
        
        scale = min(size / max(w, h, 0.001),2)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2 + 8
        if pad_h + new_h >28:
            pad_h -= 1
        padded = np.full((size, size), 0, dtype=np.uint8)
        if bbox[2]<3:
            padded[:new_h, pad_w:pad_w + new_w] = resized
        else:
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return padded

def preprocess_tf(character, single_bbox):
    resized_image = preprocess_char_image(character, single_bbox)
    resized_image = 255 - resized_image
    resized_image = resized_image.astype('float32') / 255.0
    resized_image = np.expand_dims(resized_image, axis=-1)
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image

def recognize_char(character, bbox, model=MODEL_CHAR, label_encoder=LABEL_ENCODER):

    stack_chars = []
    char_height, _ = character.shape
    resized_image = preprocess_char_image(character, bbox)
    resized_image = 255 - resized_image
    resized_image = resized_image.astype('float32') / 255.0
    resized_image = np.expand_dims(resized_image, axis=-1)
    resized_image = np.expand_dims(resized_image, axis=0)
    stack_chars.append((resized_image, char_height))

    word_stack = np.stack([char[0].squeeze(axis=0) for char in stack_chars], axis=0)
    predictions = model.predict(word_stack)
    predicted_indices = np.argmax(predictions, axis=1)

    word_decoded = label_encoder.inverse_transform(predicted_indices)
    return word_decoded[0]

def postprocess_by_height_v1(char, bbox, most_common_h_lower, height_treshold, lower_threshold, most_common_h_upper, median_lower_y):
    if char == "0" and abs(bbox[3]-bbox[2] - most_common_h_lower) < 3:
        char = "о"
    elif char == "3" and abs(bbox[3]-bbox[2] - most_common_h_lower) < 3:
        char = "з"
    elif char == "4" and abs(bbox[3]-bbox[2] - most_common_h_lower) < 3:
        char = "ч"
    elif char == "9" and abs(bbox[3]-bbox[2] - most_common_h_lower) < 3:
        char = "э"
    elif char == "8" and abs(bbox[3]-bbox[2] - most_common_h_lower) < 3:
        char = "н"
    elif char == "6" and abs(bbox[3]-bbox[2] - most_common_h_lower) < 3:
        char = "ө"
    elif (bbox[3]-bbox[2] < height_treshold or bbox[2] > lower_threshold or 
        (char == "Й" and bbox[3]-bbox[2] < most_common_h_upper+3)):
        char = char.lower()
    elif abs(bbox[2]-median_lower_y)<2:
        char = char.lower()
    elif bbox[3]-bbox[2] > most_common_h_lower+1 and bbox[2] <= 1:
        char = char
    else:
        char = char.lower()
    return char

def postprocess_by_height_v2(char, bbox, most_common_h_lower):
    
    if bbox[3]-bbox[2] > most_common_h_lower+1 and bbox[2] <= 1:
        char = char
    else:
        char = char.lower()
    return char

def recognize_both(stack_chars_normal, stack_chars_long, list_bboxes_normal, list_height_threshold, list_lower_threshold, \
                    list_median_lower_y, median_h_lower, median_h_upper, list_bboxes_long, list_length_words_full):
    
    word_stack = np.stack([char[0].squeeze(axis=0) for char in stack_chars_normal], axis=0)
    predictions = MODEL_CHAR.predict(word_stack)
    predicted_indices = np.argmax(predictions, axis=1)
    recognized_chars_normal = LABEL_ENCODER.inverse_transform(predicted_indices)
    
    recognized_chars_long = MODEL_WORD(stack_chars_long)[0]

    case_corrected_chars_normal = []
    case_corrected_chars_long = []

    for char, bbox, height_treshold, lower_threshold, median_lower_y in zip(recognized_chars_normal, list_bboxes_normal, list_height_threshold, list_lower_threshold, list_median_lower_y):
        corrected_char = postprocess_by_height_v1(char, bbox, median_h_lower, height_treshold, lower_threshold, median_h_upper, median_lower_y)
        case_corrected_chars_normal.append(corrected_char)
    for char, bbox in zip(recognized_chars_long, list_bboxes_long):
        corrected_char = postprocess_by_height_v2(char, bbox, median_h_lower)
        case_corrected_chars_long.append(corrected_char)

    word_final = ""
    for line_length in list_length_words_full:
        for word_length in line_length:
            for itr in word_length:
                word_final += ''.join(case_corrected_chars_normal[:itr[0]])
                case_corrected_chars_normal = case_corrected_chars_normal[itr[0]:]
                word_final += ''.join(case_corrected_chars_long[:itr[1]])
                case_corrected_chars_long = case_corrected_chars_long[itr[1]:]
            word_final += ' '
        word_final += '\n'

    return word_final