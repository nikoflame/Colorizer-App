import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import cv2
from keras.preprocessing.image import img_to_array
from skimage.color import lab2rgb
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# FastAPI connects to the model and serves it as an API
app = FastAPI()

# Middleware to allow cross-origin requests
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model (adjust the path if necessary)
MODEL_PATH = "models/colorizer_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

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
    # output shape: (1, h, w, 2); get rid of batch dimension
    AB_img = output[0]  # Shape: (h, w, 2)
    
    # Reconstruct LAB image:
    # - The L channel comes from the input grayscale image. Scale L from [0,1] to [0,100]
    L_channel = img_array[:, :, 0] * 100
    # - For A and B, scale predicted output from [0,1] to [-128,127]
    A_channel = AB_img[:, :, 0] * 255 - 128
    B_channel = AB_img[:, :, 1] * 255 - 128

    # Stack channels together to form a LAB image
    lab_image = np.stack([L_channel, A_channel, B_channel], axis=-1)
    
    # Clip to valid LAB ranges (L: [0,100], A: [-128,127], B: [-128,127])
    lab_image[:, :, 0] = np.clip(lab_image[:, :, 0], 0, 100)
    lab_image[:, :, 1] = np.clip(lab_image[:, :, 1], -128, 127)
    lab_image[:, :, 2] = np.clip(lab_image[:, :, 2], -128, 127)
    
    # Convert LAB to RGB
    rgb_image = lab2rgb(lab_image)
    
    # Convert to uint8
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
    return rgb_image_uint8

@app.post("/colorize/")
async def colorize(file: UploadFile = File(...)):
    """
    Endpoint that receives an image file,
    processes it using the trained model, and returns the colorized image.
    """
    # Read the uploaded file
    contents = await file.read()
    # Open image using PIL and convert to grayscale
    image = Image.open(io.BytesIO(contents)).convert("L")
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    # If the image is 2D (H, W), add a channel dimension to get (H, W, 1)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    # Scale image to [0,1]
    img_array = img_array / 255.0

    # Colorize the image
    colorized_img = colorize_image(img_array)
    
    # Convert the result back to a JPEG image in-memory
    result_img = Image.fromarray(colorized_img)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/jpeg")

if __name__ == "__main__":
    # Run the app on localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
