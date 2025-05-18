#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa

# -------------------- Page Config --------------------
st.set_page_config(page_title="Bird Sound Season Classifier", layout="centered")

# -------------------- CSS Styling --------------------
st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            color: white;
            background: #4CAF50;
            border-radius: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Constants --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Autumn 🍂", "Monsoon 🌧️", "Spring 🌸", "Summer ☀️", "Winter ❄️"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------- Model Definition --------------------
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -------------------- Model Loader --------------------
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("season_cnn_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# -------------------- Audio Processing --------------------
def audio_to_image_tensor(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = (255 * (S_D_B := (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()))).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1)
    img = Image.fromarray(img)
    return transform(img).unsqueeze(0).to(device)

# -------------------- Main Streamlit App --------------------
def main():
    st.markdown("<h1 style='text-align: center;'>🐦 Bird Sound → Season Classifier 🌤️</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a bird call to predict its seasonal context using audio deep learning.</p>", unsafe_allow_html=True)
    st.markdown("---")

    model = load_model()
    uploaded_file = st.file_uploader("🎵 Upload a bird audio file", type=["wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with st.spinner("🔍 Analyzing bird song and predicting..."):
            try:
                input_tensor = audio_to_image_tensor(uploaded_file)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                    predicted_season = class_names[pred]

                st.markdown("---")
                st.markdown(
                    f"<div style='text-align: center; padding: 20px; border-radius: 12px; background-color: #d3d3d3; font-size: 24px;'>"
                    f"🎯 Predicted Season: <strong>{predicted_season}</strong></div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"⚠️ Error processing the file: {e}")

# -------------------- Run App Safely --------------------
if __name__ == "__main__":
    main()


# In[ ]:




