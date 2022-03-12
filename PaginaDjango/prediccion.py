import tensorflow as tf
#from tensorflow.python.keras.models import load_model
#from tensorflow.python.keras.models import model_from_json
from tensorflow import keras
import pathlib
import cv2
import numpy as np

# Cargar el modelo
new_model = tf.keras.models.load_model("modelo.model")

data_dir = "/home/pi/Desktop/Tesis/ProgramaRed/Imagenes"

data_dir = pathlib.Path(data_dir)

img_height, img_width = 180, 180
batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)

img_height, img_width = 180, 180
path = "/home/pi/Desktop/Tesis/ProgramaRed/prueba/prueba_2.jpg"
image = cv2.imread(path)
image_resized = cv2.resize(image, (img_height, img_width))
image = np.expand_dims(image_resized, axis=0)
class_name = train_ds.class_names

pred = new_model.predict(image)
output_class = class_name[np.argmax(pred)]
print("The predicted class is", output_class)
