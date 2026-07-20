import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import joblib
from tensorflow.keras.models import load_model

# ==========================
# Load Model and Scaler
# ==========================

print("Loading model...")

model = load_model("models/emotion_model.keras")
scaler = joblib.load("models/scaler.pkl")

# Emotion labels
emotion_labels = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

# ==========================
# Recording Settings
# ==========================

SAMPLE_RATE = 22050
DURATION = 4          # seconds
CHANNELS = 1

# Silence threshold
SILENCE_THRESHOLD = 0.01

# ==========================
# Record Audio
# ==========================

input("\nPress Enter and start speaking for 4 seconds...")

print("\nRecording...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32"
)

sd.wait()

print("Recording finished.")

# Save recording
sf.write("recordings/live.wav", audio, SAMPLE_RATE)

# ==========================
# Check for Silence
# ==========================

volume = np.max(np.abs(audio))

print(f"Detected Volume: {volume:.4f}")

if volume < SILENCE_THRESHOLD:
    print("\nNo speech detected.")
    print("Please try again and speak clearly into the microphone.")
    exit()

# ==========================
# Feature Extraction
# ==========================

signal, sr = librosa.load("recordings/live.wav", sr=None)

mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=40
)

mfcc = np.mean(mfcc.T, axis=0)

# Apply same scaler used during training
mfcc = scaler.transform(mfcc.reshape(1, -1))

# Reshape for CNN
mfcc = mfcc.reshape(1, 40, 1)

# ==========================
# Prediction
# ==========================

prediction = model.predict(mfcc, verbose=0)

predicted_index = np.argmax(prediction)

predicted_emotion = emotion_labels[predicted_index]

confidence = prediction[0][predicted_index] * 100

# ==========================
# Results
# ==========================

print("\n==============================")
print("Emotion Prediction Result")
print("==============================")

print(f"Detected Emotion : {predicted_emotion.upper()}")
print(f"Confidence       : {confidence:.2f}%")

print("\nProbability of Each Emotion:")

for emotion, prob in zip(emotion_labels, prediction[0]):
    print(f"{emotion:<12}: {prob*100:.2f}%")

print("\nPrediction Complete.")