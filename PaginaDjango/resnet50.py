import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
from keras.optimizers import adam_v2
from matplotlib.cbook import flatten
import pathlib


def Entrenar():
    direccionCristian = "PaginaDjango\Datos"
    direccionRasb = "/home/pi/Desktop/Tesis/ProgramaRed/Imagenes"
    dirPruebaRasb = "/home/pi/Desktop/Tesis/ProgramaRed/prueba/prueba_2.jpg"
    dirPruebaCris = "PaginaDjango\static\Imagenes\Pr.jpg"
    data_dir = direccionCristian
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

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_name = train_ds.class_names

    # Training model
    print("---------Entrenamiento-----------")
    resnet_model = Sequential()
    pretrained_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(
        180, 180, 3), pooling='avg', classes=5, weights='imagenet')

    for layer in pretrained_model.layers:
        layer.trainable = False

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(2, activation='softmax'))

    resnet_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=5)
    print("-----------FIN------------")

    # path = dirPruebaCris
    # image = cv2.imread(path)
    # image_resized = cv2.resize(image, (img_height, img_width))
    # image = np.expand_dims(image_resized, axis=0)
    # print(image.shape)

    # pred = resnet_model.predict(image)
    # print(pred)

    # output_class = class_name[np.argmax(pred)]
    # print("The predicted class is", output_class)

    # ----Guardar el modelo
    resnet_model.save('PaginaDjango/modelo.model')
