import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np

# Load the model and encoder
model_path = '../cnn_character_recognition_model.h5'
model = tf.keras.models.load_model(model_path)

encoder_path = '../label_classes.npy'
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(encoder_path, allow_pickle=True)

# Resize, normalize and make prediction
def predict_char(image, model, label_encoder):
    resized_image = cv2.resize(image, (28, 28))
    resized_image = resized_image.astype('float32') / 255.0
    resized_image = np.expand_dims(resized_image, axis=-1)
    resized_image = np.expand_dims(resized_image, axis=0)

    predictions = model.predict(resized_image)
    predicted_index = np.argmax(predictions)

    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label