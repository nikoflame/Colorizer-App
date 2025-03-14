import numpy as np
from matplotlib.pyplot import imshow
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from skimage import color
import tensorflow as tf
import cv2
import os

# -----------------------
# 1. Load Training Data
# -----------------------

# Load grayscale (L) images
gray_folder = 'Data/Black_White/'
images_gray = []
for fname in os.listdir(gray_folder):
    img_path = os.path.join(gray_folder, fname)
    # Resize to 100x100 for training consistency
    img = load_img(img_path, target_size=(100, 100), color_mode="rgb")
    img = img_to_array(img) / 255.
    # Convert to grayscale (lightness) using skimage's rgb2gray (result is 2D)
    gray = color.rgb2gray(img)
    images_gray.append(gray)

# Load corresponding color images and convert to LAB space
color_folder = 'Data/colored/'
images_color = []
for fname in os.listdir(color_folder):
    img_path = os.path.join(color_folder, fname)
    img = load_img(img_path, target_size=(100, 100))
    img = img_to_array(img) / 255.
    # Convert to LAB
    lab_image = rgb2lab(img)
    # Normalize LAB channels:
    # L in [0, 100] -> divided by 100, A and B in [-128, 127] -> add 128 and divide by 255.
    lab_image_norm = np.empty_like(lab_image)
    lab_image_norm[..., 0] = lab_image[..., 0] / 100.  # L channel normalized to [0,1]
    lab_image_norm[..., 1] = (lab_image[..., 1] + 128) / 255.  # A channel normalized to [0,1]
    lab_image_norm[..., 2] = (lab_image[..., 2] + 128) / 255.  # B channel normalized to [0,1]
    # Use only the AB channels for training output (model input is L)
    Y = lab_image_norm[..., 1:]
    images_color.append(Y)

# Convert lists to arrays
# The grayscale images are 2D; add a channel dimension to get shape (height, width, 1)
X = np.array([np.expand_dims(img, axis=-1) for img in images_gray])
Y = np.array(images_color)

# -----------------------------
# 2. Define the Colorization Model
# -----------------------------
# We use a fixed input shape of (100, 100, 1)
inputs = tf.keras.Input(shape=(100, 100, 1))

# Encoder: downsampling
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)  # 50x50
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)  # 25x25
x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

# Decoder: upsampling via Conv2DTranspose
x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)  # 50x50
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)   # 100x100
# Final layer: output 2 channels (for A and B) use sigmoid so output is in [0, 1]
outputs = tf.keras.layers.Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='rmsprop', loss='mse')

model.summary()

# -----------------------------
# 3. Train the Model
# -----------------------------
model.fit(X, Y, batch_size=1, epochs=1000, verbose=1)
model.evaluate(X, Y, batch_size=1)

# Save the model for later use
model.save('colorizer_model_2.keras')

# -----------------------------
# 4. Test the Model on a New Image
# -----------------------------
test_folder = 'Data/Test/'
test_image_path = os.path.join(test_folder, 'gray_rural33.jpeg')

# Load the grayscale test image; use the same size as training (100x100) for consistency
test_img = load_img(test_image_path, target_size=(100, 100), color_mode="grayscale")
test_img = img_to_array(test_img) / 255. # shape: (100, 100, 1)

# Prepare test image: add batch dimension
test_img_batch = np.expand_dims(test_img, axis=0) # shape: (1, 100, 100, 1)

# Predict AB channels for the test image
predicted_AB = model.predict(test_img_batch)  # shape: (1, 100, 100, 2)
predicted_AB = predicted_AB[0]  # shape: (100, 100, 2)

# Reconstruct LAB image:
# For the L channel, scale test_img from [0,1] to [0,100]
L_channel = test_img[:, :, 0] * 100

# For AB channels, convert predicted values from [0,1] back to original range:
# A, B: (predicted_value * 255) - 128
A_channel = predicted_AB[:, :, 0] * 255 - 128
B_channel = predicted_AB[:, :, 1] * 255 - 128

# Combine into a LAB image
outputLAB = np.stack([L_channel, A_channel, B_channel], axis=-1)

# Clip values to valid LAB ranges
outputLAB[:, :, 0] = np.clip(outputLAB[:, :, 0], 0, 100)
outputLAB[:, :, 1] = np.clip(outputLAB[:, :, 1], -128, 127)
outputLAB[:, :, 2] = np.clip(outputLAB[:, :, 2], -128, 127)

# Convert LAB to RGB
rgb_image = lab2rgb(outputLAB)

# Display the result
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
imshow(rgb_image)
plt.title("Colorized Image")
plt.axis('off')
plt.show()
