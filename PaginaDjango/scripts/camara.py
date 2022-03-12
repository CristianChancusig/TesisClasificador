import cv2
import numpy as np
import os
import urllib.request
from django.conf import settings
import tensorflow
frame = cv2.imread('D:/Cristian/Docu_Tesis/Prueba1/ImagenesP/fork_1.jpg')
multip = 100

x = 10
y = 10
w = 10
h = 10
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('nombre', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
