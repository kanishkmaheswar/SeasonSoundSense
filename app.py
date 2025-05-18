#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa
import io

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (update if needed)
class_names = ["Autumn", "Spring", "Summer", "Winter", "Unknown"]

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Model architecture
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

# Load trained model
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("season_cnn_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Function to convert audio to spectrogram image array
def audio_to_image_tensor(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Normalize to 0-255 and convert to 3-channel image
    img = (255 * (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1)  # Convert to 3-channel
    img = Image.fromarray(img)
    return transform(img).unsqueeze(0).to(device)

# Streamlit interface
st.title("Bird Sound Season Classifier")
st.write("Upload a bird sound file to predict the season. No spectrogram will be shown.")

uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    try:
        input_tensor = audio_to_image_tensor(uploaded_file)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            st.success(f"Predicted Season: **{class_names[pred]}**")
    except Exception as e:
        st.error(f"Error processing file: {e}")


# In[ ]:




