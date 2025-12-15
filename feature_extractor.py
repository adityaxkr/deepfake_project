# feature_extractor.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import uuid

# --- Model Setup ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base_model.fc = nn.Linear(base_model.fc.in_features, 2)
        base_model.load_state_dict(torch.load("model_training/model.pth", map_location="cpu"))
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # remove final fc
        self.classifier = base_model.fc
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
        return x

model = FeatureExtractor()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def extract_and_save_features(image_path, save_dir="features_local/"):
    os.makedirs(save_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    features = model(input_tensor).squeeze().numpy()

    uid = str(uuid.uuid4())
    save_path = os.path.join(save_dir, f"{uid}.npy")
    np.save(save_path, features)
    print(f"✅ Feature saved: {save_path}")

# Example usage:
# extract_and_save_features("example_images/fake1.jpg")
