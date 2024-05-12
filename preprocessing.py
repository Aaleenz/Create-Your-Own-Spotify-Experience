import os
import librosa
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['music_database']
collection = db['music_collection']

# Function to extract MFCC features from an audio file
def extract_mfcc(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        if len(y) == 0:  # Check if the audio file is empty
            print(f"Empty audio file: {audio_path}")
            return None
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        return mfccs
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Function to normalize or standardize features
def normalize_features(features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

# Iterate through folders and process audio files
for folder in os.listdir('fma_large'):
    folder_path = os.path.join('fma_large', folder)
    if not os.path.isdir(folder_path):
        print(f"Skipping {folder_path}: Not a directory")
        continue
    for audio_file in os.listdir(folder_path):
        audio_path = os.path.join(folder_path, audio_file)
        if not audio_path.endswith('.mp3'):  # Skip non-MP3 files
            print(f"Skipping {audio_path}: Not an MP3 file")
            continue
        mfcc_features = extract_mfcc(audio_path)
        if mfcc_features is not None:
            normalized_features = normalize_features(mfcc_features)
            
            # Insert features into MongoDB
            document = {
                'audio_file': audio_file,
                'features': normalized_features.tolist()
                # Add more fields if necessary
            }
            collection.insert_one(document)

