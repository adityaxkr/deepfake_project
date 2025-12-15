# 🔍 DeepFakeNet — AI-Powered Deepfake Image Detector with Grad-CAM & Federated Learning


## 📌 Problem Statement

The rise of deepfake content poses a serious threat to digital trust, especially in the context of misinformation, identity theft, and AI-generated media manipulation. Despite existing detection tools, few provide **explainability**, **real-time inference**, and **federated feedback collection** in one system.

## 🚀 Solution: DeepFakeNet

**DeepFakeNet** is a fully functional, AI-powered web application that:

* Detects whether an image is real or AI-generated (deepfake)
* Visualizes model attention using **Grad-CAM**
* Collects human feedback to improve accuracy
* Stores anonymized feature vectors for **federated learning**

## 🎯 Key Features

| Feature                          | Description                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------- |
| 🧠 **Deepfake Image Detection**  | Classifies uploaded images as Real or Fake using a ResNet18-based classifier     |
| 🔥 **Grad-CAM Visualizations**   | Highlights the regions the model focused on while making predictions             |
| 📝 **Feedback System**           | Allows users to validate or correct predictions and leave comments               |
| 🧬 **Federated Feature Storage** | Extracts intermediate feature vectors and stores them for collaborative learning |
| 📊 **Admin Dashboard**           | Visualizes feedback stats and error patterns to guide model improvements         |



## 🧠 Architecture

```
User Upload → Image Preprocessing → Model Inference
        ➧                        ➧
     Grad-CAM           Federated Feature Vector
        ➧                        ➧
   Heatmap Display      Feature Saved as `.npy`
```

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Model**: `ResNet18` fine-tuned on deepfake vs real face images
* **Visualization**: `Grad-CAM` overlay using OpenCV
* **Backend Storage**: CSV logs, Numpy vector dumps
* **Deployment Ready**: Can be deployed on Streamlit Cloud or locally

---

## 📁 Project Structure

```
DeepFakeNet/
├── predict.py                ← Streamlit app
├── feedback_dashboard.py     ← Admin feedback dashboard
├── app.py                    ← Optional Flask API
├── model_training/
│   ├── model.pth             ← Trained model
│   └── feedback_log.csv      ← Logged user feedback
├── feature_vectors/          ← Saved feature vectors (.npy)
├── requirements.txt          ← Python dependencies
└── README.md                 ← You’re here
```

---

## 🧪 Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/adityaxkr/DeepFakeNet.git
cd DeepFakeNet
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the main app**

```bash
streamlit run predict.py
```

4. **Run the dashboard**

```bash
streamlit run feedback_dashboard.py
```

---

## 🧬 How Federated Feature Storage Works

Each uploaded image contributes a **high-dimensional feature vector** extracted from the CNN’s penultimate layer. These `.npy` files can be used in the future for:

* Federated model updates (local learning)
* Building explainable AI tools
* Building a privacy-preserving user database

---

## 💡 Future Scope

* 🔐 Integrate homomorphic encryption for privacy-preserving federated learning
* 📱 Add mobile support and API endpoints for wider adoption
* 📹 Expand to deepfake video and voice detection
* 🌐 Online dashboard to view feedback in real time

---

## 🙇‍♂️ Author

**Aditya Kumar**
📧 [aditya_202300518@smit.smu.edu.in]
🔗 [LinkedIn](https://linkedin.com/in/adityax.kr) · [GitHub](https://github.com/adityaxkr)

---

## 📄 License

This project is under the MIT License — use, remix, and build freely.

---

> 🔁 *Built with real explainability and community-driven intelligence.*
> Let’s fight deepfakes together — with AI transparency, not just accuracy.
