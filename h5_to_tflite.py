import os
import tensorflow as tf
from tensorflow import keras

My_TFlite_Model = tf.keras.models.load_model('model_2.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(My_TFlite_Model)
tflite_model = converter.convert()
open("My_TFlite_Model.tflite", "wb").write(tflite_model)