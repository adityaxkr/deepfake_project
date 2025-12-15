import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import os

# --- ✨ SET PAGE CONFIG FIRST ---
st.set_page_config(page_title="DeepFakeNet 🔍", layout="wide")

# --- ✨ NEW: Grad-CAM Class (WITH FIX) ---
class GradCAM:
    """
    Class to generate Grad-CAM heatmaps. This encapsulates the logic and state.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register a permanent forward hook to the target layer
        self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        """Forward hook to save layer activations."""
        self.activations = output
        
        # --- ✨ THE FIX IS HERE ---
        # Only register the backward hook if the output tensor requires a gradient.
        # This prevents the hook from interfering with no_grad() contexts,
        # such as the feature extraction function.
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        """Backward hook to save gradients."""
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        """
        Makes the class instance callable. Performs a forward and backward pass
        to generate the heatmap.
        """
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        if self.gradients is None:
            st.error("Gradients could not be captured for Grad-CAM.")
            return None, output

        pooled_grad = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach().squeeze(0)
        for i in range(pooled_grad.shape[0]):
            activations[i, :, :] *= pooled_grad[i]
            
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        
        return heatmap, output

# --- Model and Grad-CAM Setup ---
@st.cache_resource
def load_model_and_gradcam():
    """Loads the model and instantiates the GradCAM class once."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model_training/model.pth", map_location="cpu"))
    model.eval()
    
    grad_cam = GradCAM(model, target_layer=model.layer4[1].conv2)
    
    return model, grad_cam

# --- SCRIPT EXECUTION STARTS HERE ---
model, grad_cam = load_model_and_gradcam()

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Grad-CAM Heatmap Overlay Utility ---
def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(image)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return superimposed_img

# --- Feature Extraction for Federated Learning ---
@torch.no_grad()
def extract_feature_vector(model, image_tensor):
    """
    This function will now run without errors because the Grad-CAM hook
    is now conditional and won't interfere with the no_grad() context.
    """
    x = image_tensor
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x.squeeze(0).cpu().numpy()

# --- Streamlit UI ---
st.title("🔍 DeepFake Image Detector with Grad-CAM & Federated Features")

uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # --- Grad-CAM Pass ---
    # Prepare a tensor that requires gradients for Grad-CAM
    grad_input_tensor = transform(image).unsqueeze(0)
    grad_input_tensor.requires_grad_()

    heatmap, output = grad_cam(grad_input_tensor)
    
    pred = torch.argmax(output, dim=1).item()
    confidence = F.softmax(output, dim=1)[0][pred].item()
    label = "✅ Real" if pred == 1 else "🧠 Fake"
    
    with col2:
        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")
        
        if heatmap is not None:
            cam_result = overlay_heatmap(heatmap, image)
            st.image(cam_result, caption="🔥 Grad-CAM Explanation", use_container_width=True)
        else:
            st.warning("Could not generate Grad-CAM visualization.")

    # --- Feedback Section ---
    st.markdown("---")
    st.subheader("📝 Feedback")
    # (Feedback logic remains the same)
    feedback_col1, feedback_col2 = st.columns(2)
    with feedback_col1:
        user_feedback = st.radio("Was the prediction correct?", ["Yes", "No"], index=0, key="feedback")

    correct_label = None
    if user_feedback == "No":
        with feedback_col2:
            correct_label = st.radio("Select the correct label:", ["Real", "Fake"], key="correct_label")
    
    user_comment = st.text_area("💬 Any comments? (Optional)", placeholder="Let us know what went wrong...")

    if st.button("Submit Feedback"):
        feedback_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": uploaded_file.name,
            "predicted_label": "Fake" if pred == 0 else "Real",
            "confidence": round(confidence, 4),
            "was_correct": user_feedback,
            "correct_label": correct_label if user_feedback == "No" else "N/A",
            "comment": user_comment
        }
        feedback_df = pd.DataFrame([feedback_data])
        feedback_file = "model_training/feedback_log.csv"
        os.makedirs("model_training", exist_ok=True)
        if not os.path.exists(feedback_file):
            feedback_df.to_csv(feedback_file, index=False)
        else:
            feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
        st.success("✅ Feedback submitted! Thank you.")

    # --- Federated Feature Vector Contribution ---
    st.markdown("---")
    st.subheader("🧬 Federated Learning Contribution")

    # Prepare a fresh tensor for feature extraction (no gradients needed)
    inference_input_tensor = transform(image).unsqueeze(0)
    
    # This call will now work without error
    feature_vector = extract_feature_vector(model, inference_input_tensor) 
    
    os.makedirs("feature_vectors", exist_ok=True)
    feature_filename = f"feature_vectors/{os.path.splitext(uploaded_file.name)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.npy"
    np.save(feature_filename, feature_vector)

    st.markdown(f"✅ Feature vector saved for federated learning at `{feature_filename}`")