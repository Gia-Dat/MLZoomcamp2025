# IMPORTS
from io import BytesIO
from urllib import request
from PIL import Image
import onnxruntime as ort
import numpy as np

# Load the ONNX model from the container
sess = ort.InferenceSession("hair_classifier_empty.onnx")
print("Output node name:", sess.get_outputs()[0].name)

# Functions to download and preprocess images


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    return img


# Download and prepare the test image
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(url)
img_prepared = prepare_image(img)

# Convert to numpy and normalize
x = np.array(img_prepared).astype("float32") / 255
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
x_norm = (x - mean) / std

print(f"R channel of first pixel: {x_norm[0][0][0]}")

# Prepare input for ONNX model
x_input = np.transpose(x_norm, (2, 0, 1))[None, :, :, :].astype(np.float32)

# Run inference
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
pred = sess.run([output_name], {input_name: x_input})[0]
prob = pred[0][0]

print("Probability of positive class:", prob)
