import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
import io
import requests

# --- 1. MODEL ARCHITECTURE (Must match your train.py) ---


class AnimalClassifierMobileNet(nn.Module):
    def __init__(self, size_inner=100, num_classes=10):
        super(AnimalClassifierMobileNet, self).__init__()
        self.base_model = models.mobilenet_v2(weights=None)
        self.base_model.classifier = nn.Identity()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


# --- 2. INITIALIZE APP AND LOAD MODEL ---
app = FastAPI()

classes = ["butterfly", "cat", "chicken", "cow", "dog",
           "elephant", "horse", "sheep", "spider", "squirrel"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AnimalClassifierMobileNet(num_classes=10)
model.load_state_dict(torch.load("animal_model.pth", map_location=device))
model.to(device)
model.eval()

# --- 3. INPUT SCHEMA ---


class ImageURL(BaseModel):
    url: str

# --- 4. PREDICTION ENDPOINT ---


@app.post("/predict")
async def predict(data: ImageURL):
    # Download image
    response = requests.get(data.url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    img_t = preprocess(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_t)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    # Format result
    result = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return result
