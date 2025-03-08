import numpy as np
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from skimage import color
import keras as keras
import tensorflow as tf
import cv2
import os

# Load training data
folder_path = 'Data/gray_smallData/'
images_gray = []
for img in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img)
    img = load_img(img_path, target_size=(100,100)) 
    img = img_to_array(img) / 255
    gray = color.rgb2gray(img)
    images_gray.append(gray)

# Load corresponding color data
folder_path = 'Data/smallData/'
images_color = []
for img in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img)
    img = load_img(img_path, target_size=(100,100))
    img = img_to_array(img) / 255
    lab_image = rgb2lab(img)
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    Y = lab_image_norm[:, :, 1:]
    images_color.append(Y)

# Ensure consistent shape
sizeX = max([img.shape[0] for img in images_gray])
sizeY = max([img.shape[1] for img in images_gray])

# Resize to match sizeX, sizeY
images_gray = [cv2.resize(img, (sizeY, sizeX)) for img in images_gray]
images_color = [cv2.resize(img, (sizeY, sizeX)) for img in images_color]

# Convert to numpy arrays
X = np.array(images_gray).reshape(-1, sizeX, sizeY, 1)
X_size = np.array([[img.shape[1], img.shape[0]] for img in images_gray])
Y = np.array(images_color).reshape(-1, sizeX, sizeY, 2)

# Define the model
x1 = keras.Input(shape=(None, None, 1)) # Dynamic input size
size_input = keras.Input(shape=(2,), dtype=tf.int32)
x2 = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(x1)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x4 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x3)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x6 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x5)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu', padding='same')(x9)
x11 = UpSampling2D((2, 2))(x10)
x12 = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x11) # 2 for AB channels
x,y = size_input[0]
x12 = tf.image.resize(x12,[x, y])
x12 = tf.reshape(x12,(1,x, y,2))

# Finish model
model = keras.Model(inputs=[x1, size_input], outputs=x12)

model.compile(optimizer='rmsprop', loss='mse')
model.fit([X, X_size], Y, batch_size=1, epochs=100, verbose=1)

model.evaluate([X, X_size], Y, batch_size=1)

# Test image
folder_path = 'Data/Test/'
img_path = os.path.join(folder_path, 'gray_ILSVRC2017_test_00000052.JPEG')

# Load the image
img = load_img(img_path, color_mode="grayscale")
img = img_to_array(img) / 255
ss=img.shape

# Convert to Numpy Array
X = np.array(img)
X = np.expand_dims(X, axis=2)
X=np.reshape(X,(1,ss[0],ss[1],1))

# Output has 2 channels for AB
size_t = np.array([ss[0], ss[1]]).reshape(1, 2)
output = model.predict([X, size_t])
output=np.reshape(output,(ss[0],ss[1],2))
output=cv2.resize(output,(ss[0],ss[1]))
AB_img = output

outputLAB = np.zeros((ss[0], ss[1], 3))

# If shapes are swapped, fix using transpose
if AB_img.shape[:2] != (ss[0], ss[1]):
    AB_img = np.transpose(AB_img, (1, 0, 2))

img=np.reshape(img,(ss[0],ss[1]))
outputLAB[:,:,0]=img
outputLAB[:,:,1:]=AB_img

# Denormalize LAB values
outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]

# Convert back to RGB
rgb_image = lab2rgb(outputLAB)

import matplotlib.pyplot as plt

imshow(rgb_image)
plt.show()