import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from src.segmentation_word import segment_words
from src.segmentation_character import character_segmentation
from src.utils import reshape_image

MODEL_PATH = 'model/recognition_model.h5'
MODEL = tf.keras.models.load_model(MODEL_PATH)

ENCODER_PATH = 'model/encoder.npy'
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.classes_ = np.load(ENCODER_PATH, allow_pickle=True)

def preprocess_char_image(img: np.ndarray, bbox, size=28):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orig_height, orig_width = img.shape
    if (orig_height > 7 or orig_width > 7) or (orig_width / orig_height>1.7):
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        char_crop = img[y:y+h, x:x+w]
        
        scale = min(size / max(w, h, 0.001),2)
        new_w, new_h = int(w * scale), int(h * scale)
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

def recognize_char(character, bbox, model=MODEL, label_encoder=LABEL_ENCODER):

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

def recognize_line(line_image, rescale_idx, mean_w, most_common_h_upper, most_common_h_lower, model=MODEL, label_encoder=LABEL_ENCODER):

    line_height, _ = line_image.shape
    height_treshold = line_height / 1.6
    lower_threshold = line_height / 8  # New condition
    words = segment_words(line_image)
    list_length_words = []
    stack_chars = []
    bboxes_all = []
    lower_letter_top_y = []

    for word in words:
        characters, bboxes = character_segmentation(reshape_image(word, rescale_idx), mean_w)
        for box in bboxes:
            if abs(box[3] - box[2] - most_common_h_lower)< 2:
                lower_letter_top_y.append(box[2])
        median_lower_y = np.median(lower_letter_top_y) if lower_letter_top_y else 0
        list_length_words.append(len(characters))
        bboxes_all.extend(bboxes)
        for character, single_bbox in zip(characters, bboxes):
            char_height, char_width = character.shape
            resized_image = preprocess_char_image(character, single_bbox)
            resized_image = 255 - resized_image
            resized_image = resized_image.astype('float32') / 255.0
            resized_image = np.expand_dims(resized_image, axis=-1)
            resized_image = np.expand_dims(resized_image, axis=0)
            stack_chars.append((resized_image, char_height))
    
    word_stack = np.stack([char[0].squeeze(axis=0) for char in stack_chars], axis=0)
    predictions = model.predict(word_stack)
    predicted_indices = np.argmax(predictions, axis=1)
    
    word_decoded = label_encoder.inverse_transform(predicted_indices)
    
    word_corrected = []
    for i, (char, height, bbox) in enumerate(zip(word_decoded, [char[1] for char in stack_chars], bboxes_all)):
        if char == "0" and abs(height - most_common_h_lower) < 2:
            corrected_char = "о"
        elif char == "3" and abs(height - most_common_h_lower) < 2:
            corrected_char = "з"
        elif char == "4" and abs(height - most_common_h_lower) < 2:
            corrected_char = "ч"
        elif char == "9" and abs(height - most_common_h_lower) < 2:
            corrected_char = "э"
        elif char == "8" and abs(height - most_common_h_lower) < 2:
            corrected_char = "н"
        elif char == "6" and abs(height - most_common_h_lower) < 2:
            corrected_char = "ө"
        elif (height < height_treshold or bbox[2] > lower_threshold or 
            (char == "Й" and height < most_common_h_upper+3)):
            corrected_char = char.lower()
        elif abs(bbox[2]-median_lower_y)<2:
            corrected_char = char.lower()
        elif height > most_common_h_lower+1 and bbox[2] <= 1:
            corrected_char = char
        else:
            corrected_char = char.lower()
    
        word_corrected.append(corrected_char)
    
    word_no_space = ''.join(word_corrected)
    
    for num in list_length_words:
        last_space = word_no_space.rfind(" ")
        word_no_space = word_no_space[:last_space+num+1] + " " + word_no_space[last_space+num+1:]
    
    return word_no_space.rstrip()