import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# img = cv2.imread('..\input\HAM10000_images_part_1\ISIC_0027419.jpg')
img = Image.open('..\input\HAM10000_images_part_1\ISIC_0027419.jpg').resize((100,75))
img = np.asarray(img, dtype=np.float32)/255
img = img.reshape(1, 75, 100, 3)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_3.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)