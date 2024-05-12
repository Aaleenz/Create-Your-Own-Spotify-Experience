import numpy as np
from io import BytesIO
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient

# Connect to MongoDB and retrieve training data
client = MongoClient('localhost', 27017)
db = client['music_database']
collection = db['music_collection']
training_data = np.array([doc['features'] for doc in collection.find({}, {'features': 1})])

# Define the scaler and fit it on training data
scaler = StandardScaler()
scaler.fit(training_data)

def convert_to_mp3(audio_data, sample_rate=22050):
    # Convert audio_data to NumPy array
    audio_data_np = np.array(audio_data)

    # Inverse Normalization
    # Apply the inverse transformation using the fitted scaler
    audio_data_inverse = scaler.inverse_transform(audio_data_np.reshape(1, -1)).reshape(-1)

    # Convert audio data to the appropriate data type (e.g., int16)
    audio_data_int = (audio_data_inverse * (2**15 - 1)).astype(np.int16)

    # Convert numerical data back to audio samples
    audio_segment = AudioSegment(audio_data_int.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

    # Create a BytesIO object to store the audio data
    audio_bytes = BytesIO()

    # Export the audio segment to MP3 format and save it to the BytesIO object
    audio_segment.export(audio_bytes, format='mp3')

    # Reset the pointer of the BytesIO object to the beginning
    audio_bytes.seek(0)

    return audio_bytes

