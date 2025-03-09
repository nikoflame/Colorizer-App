import numpy as np
from matplotlib.pyplot import imshow
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from skimage import color
import tensorflow as tf
import cv2
import os

# ----------------------
# 1. Load Training Data
# ----------------------
# Set a target size for all training images
target_size = (100, 100) # Custom testing size

# Load grayscale (L) images
gray_folder = 'Data/gray_smallData/'
images_gray = []
for fname in os.listdir(gray_folder):
    img_path = os.path.join(gray_folder, fname)
    # Resize images to target_size for training consistency
    img = load_img(img_path, target_size=target_size, color_mode="rgb")
    img = img_to_array(img) / 255.0
    # Convert to grayscale (lightness); note that rgb2gray (result is 2D)
    gray = color.rgb2gray(img)
    images_gray.append(gray)

# Load corresponding color images and convert to LAB space
color_folder = 'Data/smallData/'
images_color = []
for fname in os.listdir(color_folder):
    img_path = os.path.join(color_folder, fname)
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img) / 255.0
    # Convert to LAB
    lab_image = rgb2lab(img)
    # Normalize LAB channels:
    #   L in [0,100] becomes [0,1]
    # L in [0, 100] -> divided by 100, A and B in [-128, 127] -> add 128 and divide by 255.
    lab_image_norm = np.empty_like(lab_image)
    lab_image_norm[..., 0] = lab_image[..., 0] / 100.0
    lab_image_norm[..., 1] = (lab_image[..., 1] + 128) / 255.0
    lab_image_norm[..., 2] = (lab_image[..., 2] + 128) / 255.0
    # Use only the AB channels for training output (model input is L)
    Y = lab_image_norm[..., 1:]
    images_color.append(Y)

# Convert lists to arrays
# The grayscale images are 2D; add a channel dimension to get shape (height, width, 1)
X = np.array([np.expand_dims(img, axis=-1) for img in images_gray])
Y = np.array(images_color)

# -----------------------------------------------------
# 2. Define a Convolutional Model with Dynamic Input Size
# -----------------------------------------------------
# Instead of a fixed input shape, we use (None, None, 1)
inputs = tf.keras.Input(shape=(None, None, 1))

# Encoder: downsampling
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)  # Downsample by 2
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)  # Downsample by 2 again
x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

# Decoder: upsampling via Conv2DTranspose
x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x) # Upsample by 2
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x) # Upsample by 2 again
# Final layer: output 2 channels (for A and B) use sigmoid so output is in [0, 1]
outputs = tf.keras.layers.Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='rmsprop', loss='mse')

model.summary()

# -----------------------------------------------------
# 3. Train the Model
# -----------------------------------------------------
# X and Y now have shape (num_samples, H, W, channels) with H=W=target_size
model.fit(X, Y, batch_size=1, epochs=400, verbose=1)
model.evaluate(X, Y, batch_size=1)

# Save the model for later use
model.save('colorizer_model_3.keras')

# -----------------------------------------------------
# 4. Test the Model on a Full-Sized Image
# -----------------------------------------------------
test_folder = 'Data/Test/'
test_image_path = os.path.join(test_folder, 'gray_ILSVRC2017_test_00000052.JPEG')

# Load the grayscale test image.
test_img = load_img(test_image_path, target_size=target_size, color_mode="grayscale") # target sized
#test_img = load_img(test_image_path, color_mode="grayscale") # full size
test_img = img_to_array(test_img) / 255.0  # shape: (H, W, 1)
H, W = test_img.shape[0], test_img.shape[1] # original image dimensions

# Prepare test image: add batch dimension
test_img_batch = np.expand_dims(test_img, axis=0)  # shape: (1, H, W, 1)

# Predict AB channels for the test image (output shape will be (1, H_out, W, 2))
predicted_AB = model.predict(test_img_batch)[0]  # shape: (H_out, W, 2)
predicted_AB = cv2.resize(predicted_AB, (W, H))  # resize to original image dimensions (H, W, 2)

# Reconstruct LAB image:
# For the L channel, scale test_img from [0,1] to [0,100]
L_channel = test_img[:, :, 0] * 100.0

# For AB channels, convert predicted values from [0,1] back to original range:
# A, B: (predicted_value * 255) - 128
A_channel = predicted_AB[:, :, 0] * 255.0 - 128.0
B_channel = predicted_AB[:, :, 1] * 255.0 - 128.0

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
plt.figure(figsize=(10, 10))
imshow(rgb_image)
plt.title("Colorized Image")
plt.axis('off')
plt.show()