# 🎙️ Audio Emotion Recognition Engine

A Deep Learning project that recognizes human emotions from speech using **MFCC (Mel-Frequency Cepstral Coefficients)** and a **1D Convolutional Neural Network (CNN)**. The system can classify emotions from audio files and perform **real-time emotion prediction using a live microphone**.

---

## 📌 Project Overview

Speech carries valuable emotional information. This project extracts acoustic features from speech recordings using **MFCCs** and trains a **CNN model** to classify different human emotions.

The trained model can also predict emotions from live microphone recordings, making it suitable for real-time applications.

---

## 🎯 Objectives

- Extract MFCC features from speech audio.
- Train a CNN model for emotion classification.
- Evaluate model performance using different metrics.
- Generate training graphs and confusion matrix.
- Predict emotions from live microphone audio.

---

## 😊 Supported Emotions

- Angry 😠
- Calm 😌
- Disgust 🤢
- Fearful 😨
- Happy 😀
- Neutral 😐
- Sad 😢
- Surprised 😲

---

## 📂 Dataset

**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- 24 Professional Actors
- 1440 Audio Files
- 8 Emotion Classes

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SoundDevice
- SoundFile
- Joblib

---

## 📁 Project Structure

```
Audio_Emotion_Recognition/
│
├── dataset/
│   ├── Actor_01
│   ├── Actor_02
│   └── ...
│
├── graphs/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
│
├── models/
│   ├── emotion_model.keras
│   └── scaler.pkl
│
├── recordings/
│
├── feature_extraction.py
├── train_model.py
├── live_prediction.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Audio_Emotion_Recognition.git
```

Go to project directory

```bash
cd Audio_Emotion_Recognition
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Step 1

Extract MFCC Features

```bash
python feature_extraction.py
```

### Step 2

Train CNN Model

```bash
python train_model.py
```

### Step 3

Predict Emotion from Live Microphone

```bash
python live_prediction.py
```

---

## 📈 Model Performance

| Metric | Value |
|---------|--------|
| Test Accuracy | **66.32%** |
| Emotion Classes | 8 |
| Audio Samples | 1440 |
| Features | 40 MFCC |

---

## 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📸 Project Outputs

The project automatically generates:

- Accuracy Graph
  <img width="350" height="220" alt="image" src="https://github.com/user-attachments/assets/3bb7ddfe-d4e4-4bad-8b20-85aec1b6a8b0" />

- Loss Graph
  <img width="400" height="220" alt="image" src="https://github.com/user-attachments/assets/ebd56741-d00a-48b2-94a5-2825a13db3da" />

- Confusion Matrix
  <img width="350" height="220" alt="image" src="https://github.com/user-attachments/assets/6fa0fc1a-83c0-4e6c-9cff-df5fcb0897cb" />

- Trained CNN Model
- Saved Scaler
- Live Audio Recording

---

## 🚀 Features

✅ MFCC Feature Extraction

✅ CNN-Based Emotion Recognition

✅ Audio Preprocessing

✅ Real-Time Microphone Prediction

✅ Confidence Score Display

✅ Silence Detection

✅ Model Saving

✅ Training Graphs

✅ Confusion Matrix

---

## 🔮 Future Improvements

- Improve accuracy using LSTM
- Data Augmentation
- Noise Reduction
- Real-Time GUI
- Speech-to-Text Integration
- Web Deployment using Flask
- Streamlit Dashboard

---

## 👨‍💻 Author

Muhammad Hammad

BS Data Science

University of Haripur

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.
