from contour_detection import detect_cntr
from line_segmentation import line_segmentation_horizontal_projection
from word_segmentation import word_segmentation
from char_segmentation import character_segmentation
from char_recognition import predict_char

def postprocess_text(input_string):
    replacements_dict = {
        "ь[": "Ы",
        "[[": "Н",
        "[": "Н",
        "][": "Н",
        "]]": "Н",
    }
    for key, value in replacements_dict.items():
        input_string = input_string.replace(key, value)
    return input_string

def predict(img_path):
    contour_images = detect_cntr(img_path)
    line_images_all = []
    for i in range(len(contour_images)):
        line_images = line_segmentation_horizontal_projection(contour_images[i])
        if line_images != None:
            for single_line in line_images:
                if single_line.shape[0] > 10: #change
                    line_images_all.append(single_line)

    words_images = []
    for i in range(len(line_images_all)):
        words = word_segmentation(line_images_all[i], False)
        words_images.append(words)

    result_full_text = ""
    for ind1, words in enumerate(words_images):
        result_line_text = ""
        for ind2, word in enumerate(words):
            result_word_text = ""
            characters = character_segmentation(word, False)
            for character in characters:
                predicted_char = predict_char(character, model, label_encoder)
                result_word_text = result_word_text + predicted_char
            result_line_text = result_line_text + result_word_text + " "
        result_full_text = result_full_text + result_line_text + "\n"

    output_string = postprocess_text(result_full_text)

    return output_string
    