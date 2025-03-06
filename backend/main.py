import io
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
import os

# Importing model fix
colorization_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../colorization"))
sys.path.append(colorization_path)

from colorizers import eccv16, siggraph17
from colorizers.util import preprocess_img, postprocess_tens, load_img

app = FastAPI()

# Load pre-trained model (Zhang et al.)
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

# Use GPU if available (Optional)
use_gpu = torch.cuda.is_available()
if use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

def colorize_image(image: Image.Image, model="siggraph17"):
    """ Convert grayscale image to colorized output. """

    # Convert PIL image to NumPy array (to match demo_release.py)
    img_path = "temp.jpg"
    image.save(img_path)
    
    # Load and preprocess image (demo_release.py way)
    img = load_img(img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

    if use_gpu:
        tens_l_rs = tens_l_rs.cuda()

    # Choose the model
    if model == "eccv16":
        out_img = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    else:
        out_img = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # Convert to JPEG for web response
    _, img_encoded = cv2.imencode(".jpg", out_img)
    return img_encoded.tobytes()

@app.post("/colorize/")
async def colorize(file: UploadFile):
    """ API endpoint to process grayscale images. """
    image = Image.open(io.BytesIO(await file.read()))
    colorized_image = colorize_image(image)
    return Response(content=colorized_image, media_type="image/jpeg")

# Run API: python -m uvicorn backend.main:app --reload