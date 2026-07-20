import os
import numpy as np
import librosa

# Dataset path
DATASET_PATH = "dataset"

# Emotion labels
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features = []
labels = []

# Read all actor folders
for actor_folder in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            file_path = os.path.join(actor_path, file)

            # Extract emotion code
            emotion_code = file.split("-")[2]
            emotion = emotion_dict[emotion_code]

            # Load audio
            audio, sample_rate = librosa.load(file_path, sr=None)

            # Extract 40 MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )

            # Take mean across time
            mfcc = np.mean(mfcc.T, axis=0)

            features.append(mfcc)
            labels.append(emotion)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

print("Feature Shape:", features.shape)
print("Labels Shape:", labels.shape)

print("\nSample Labels:")
print(labels[:10])