# streamlit_app/app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Load your trained model
model_path = "../model_training/model.pth"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the same transform used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Prediction Function
def predict_image(img):
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, 1)
        return prediction.item(), confidence.item()

# Streamlit App
st.set_page_config(page_title="DeepFakeNet Detector", page_icon="🧠")
st.title("🧠 DeepFake Image Detector")
st.write("Upload an image to detect if it's **real** or **AI-generated**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):
        label, confidence = predict_image(image)
        class_name = "Fake (AI-generated)" if label == 1 else "Real"
        st.markdown(f"### 🧾 Prediction: `{class_name}`")
        st.markdown(f"🔍 Confidence: `{confidence * 100:.2f}%`")
