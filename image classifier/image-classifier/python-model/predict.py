
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

classes = ['glioma','meningioma','notumor','pituitary']

print("Image classification started")

# Check if image path is provided
if len(sys.argv) < 2:
    print("Error: Please provide an image path")
    print("Example: python predict.py brain.jpg")
    sys.exit()

image_path = sys.argv[1]

print("Image received:", image_path)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "brain_tumor_model.h5")

model = tf.keras.models.load_model(model_path)

img = image.load_img(image_path, target_size=(224,224))
img_array = image.img_to_array(img)

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

index = np.argmax(prediction)
result = classes[index]
confidence = prediction[0][index] * 100

print(f"Prediction: {result} Confidence: {confidence:.2f}%")