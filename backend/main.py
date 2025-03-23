import io
import os
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
from skimage.color import lab2rgb

# Initialize the Flask app
app = Flask(__name__)
# Configure CORS with the allowed origins
CORS(app, origins=["https://colorizer-app.onrender.com", "https://localhost:5000", "https://colorizer-app-1.onrender.com", "https://localhost:10000", "206.251.89.73"])

# TF Debugger
#tf.debugging.set_log_device_placement(True)

# Test loading model from JSON
with open("models/colorizer_model_4.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("models/colorizer_model_4.h5")

# Load the saved model
#WEIGHT_PATH = "models/colorizer_model_weightless_test.h5"
#MODEL_PATH = "models/colorizer_model_weightless_test.keras"
#model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#model.load_weights(WEIGHT_PATH)

# TF Debugger
#model.summary()

def colorize_image(img_array):
    """
    Given a grayscale image (numpy array with shape (H, W, 1) scaled to [0,1]),
    this function runs the model and returns a colorized RGB image (uint8).
    """
    # Get image dimensions
    h, w = img_array.shape[:2]
    # Expand dimensions to create a batch of 1
    X = np.expand_dims(img_array, axis=0)  # Shape: (1, h, w, 1)
    # Create the size tensor as expected by the model (note: model expects [height, width])
    size_tensor = np.array([[h, w]], dtype=np.int32)
    
    # Predict AB channels using the model
    output = model.predict([X, size_tensor])
    # output shape: (1, h, w, 2); remove batch dimension
    AB_img = output[0]  # Shape: (h, w, 2)
    
    # Reconstruct LAB image:
    L_channel = img_array[:, :, 0] * 100  # Scale L from [0,1] to [0,100]
    A_channel = AB_img[:, :, 0] * 255 - 128  # Scale A from [0,1] to [-128,127]
    B_channel = AB_img[:, :, 1] * 255 - 128  # Scale B from [0,1] to [-128,127]
    
    # Stack channels to form a LAB image
    lab_image = np.stack([L_channel, A_channel, B_channel], axis=-1)
    
    # Clip to valid LAB ranges
    lab_image[:, :, 0] = np.clip(lab_image[:, :, 0], 0, 100)
    lab_image[:, :, 1] = np.clip(lab_image[:, :, 1], -128, 127)
    lab_image[:, :, 2] = np.clip(lab_image[:, :, 2], -128, 127)
    
    # Convert LAB to RGB
    rgb_image = lab2rgb(lab_image)
    # Convert to uint8
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
    return rgb_image_uint8

@app.route('/colorize/', methods=['POST'])
def colorize():
    """
    Endpoint that receives an image file,
    processes it using the trained model, and returns the colorized image.
    """
    # Ensure the file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the uploaded file
    contents = file.read()
    # Open the image using PIL and convert to grayscale
    image = Image.open(io.BytesIO(contents)).convert("L")
    
    # Convert the PIL image to a numpy array
    img_array = np.array(image)
    # If the image is 2D (H, W), add a channel dimension to get (H, W, 1)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    # Scale image to [0,1]
    img_array = img_array / 255.0

    # Colorize the image using the model
    colorized_img = colorize_image(img_array)
    
    # Convert the result back to a JPEG image in-memory
    result_img = Image.fromarray(colorized_img)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    
    # Return the image as a response
    return send_file(buf, mimetype="image/jpeg")

if __name__ == "__main__":
    # Run Flask app with online VMS details
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)
