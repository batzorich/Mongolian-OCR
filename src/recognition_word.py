import math
import os

import cv2
import numpy as np
from onnxruntime import InferenceSession

def sort_polygon(points):
    points.sort(key=lambda x: (x[0][1], x[0][0]))
    for i in range(len(points) - 1):
        for j in range(i, -1, -1):
            if abs(points[j + 1][0][1] - points[j][0][1]) < 10 and (points[j + 1][0][0] < points[j][0][0]):
                temp = points[j]
                points[j] = points[j + 1]
                points[j + 1] = temp
            else:
                break
    return points


def crop_image(image, points):
    assert len(points) == 4, "shape of points must be 4*2"
    crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                         np.linalg.norm(points[2] - points[3])))
    crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                          np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0],
                                [crop_width, 0],
                                [crop_width, crop_height],
                                [0, crop_height]])
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    image = cv2.warpPerspective(image,
                                matrix, (crop_width, crop_height),
                                borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    height, width = image.shape[0:2]
    if height * 1.0 / width >= 1.5:
        image = np.rot90(image, k=3)
    return image


class CTCDecoder(object):
    def __init__(self):

        self.character = ['blank', ' ', '!', '#', '$', '%', '&', "'", '(', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', 
                            '@', '_', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'Ө', 'П', 'Р', 'С', 'Т', 'У', 'Ү', 
                            'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 
                            'м', 'н', 'о', 'ө', 'п', 'р', 'с', 'т', 'у', 'ү', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
        
    def __call__(self, outputs):
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            outputs = outputs[-1]
        indices = outputs.argmax(axis=2)
        return self.decode(indices, outputs)

    def decode(self, indices, outputs):
        results = []
        confidences = []
        ignored_tokens = [0]  # for ctc blank
        for i in range(len(indices)):
            selection = np.ones(len(indices[i]), dtype=bool)
            selection[1:] = indices[i][1:] != indices[i][:-1]
            for ignored_token in ignored_tokens:
                selection &= indices[i] != ignored_token
            result = []
            confidence = []
            for j in range(len(indices[i][selection])):
                result.append(self.character[indices[i][selection][j]])
                confidence.append(outputs[i][selection][j][indices[i][selection][j]])
            results.append(''.join(result))
            confidences.append(confidence)
        return results, confidences

class Recognition:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.input_shape = [3, 48, 320]
        self.ctc_decoder = CTCDecoder()
        
    def resize(self, image, max_wh_ratio):
        input_h, input_w = self.input_shape[1], self.input_shape[2]

        assert self.input_shape[0] == image.shape[2]
        input_w = int((input_h * max_wh_ratio))
        w = self.inputs.shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            input_w = w
        h, w = image.shape[:2]
        ratio = w / float(h)
        if math.ceil(input_h * ratio) > input_w:
            resized_w = input_w
        else:
            resized_w = int(math.ceil(input_h * ratio))

        resized_image = cv2.resize(image, (resized_w, input_h))
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padded_image = np.zeros((self.input_shape[0], input_h, input_w), dtype=np.float32)
        padded_image[:, :, 0:resized_w] = resized_image
        return padded_image
    
    def __call__(self, images):
        batch_size = 6
        num_images = len(images)

        results = [['', 0.0]] * num_images
        confidences = [['', 0.0]] * num_images
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))

        for index in range(0, num_images, batch_size):
            input_h, input_w = self.input_shape[1], self.input_shape[2]
            max_wh_ratio = input_w / input_h
            norm_images = []
            for i in range(index, min(num_images, index + batch_size)):
                h, w = images[indices[i]].shape[0:2]
                max_wh_ratio = max(max_wh_ratio, w * 1.0 / h)
            for i in range(index, min(num_images, index + batch_size)):
                norm_image = self.resize(images[indices[i]], max_wh_ratio)
                norm_image = norm_image[np.newaxis, :]
                norm_images.append(norm_image)
            norm_images = np.concatenate(norm_images)

            outputs = self.session.run(None,
                                        {self.inputs.name: norm_images})
            result, confidence = self.ctc_decoder(outputs[0])
            for i in range(len(result)):
                results[indices[index + i]] = result[i]
                confidences[indices[index + i]] = confidence[i]
        return results, confidences
