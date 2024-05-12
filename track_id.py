import os
import pandas as pd
import json

from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['music_db']
collection = db['music_col']

# Query MongoDB to retrieve track IDs and preprocessed data
try:
    cursor = collection.find({}, {"audio_file": 1, "features": 1, "_id": 0})
    data_from_mongo = [(doc['audio_file'], doc['features']) for doc in cursor]
    if len(data_from_mongo) == 0:
        print("Error: No data found in MongoDB collection")
        exit()
except Exception as e:
    print(f"Error querying data from MongoDB: {e}")
    exit()

# Insert track ID into MongoDB documents alongside corresponding data
for audio_file, features in data_from_mongo:
    track_id = audio_file.split('.')[0].lstrip('0')  # Remove leading zeros
    
    # Update document to include track_id field
    try:
        collection.update_one({"audio_file": audio_file}, {"$set": {"track_id": track_id}})
        print(f"Successfully updated document with track ID: {track_id}")
    except Exception as e:
        print(f"Error updating document with track ID: {track_id}. Error: {e}")

# Optional: Print track IDs without leading zeros and corresponding preprocessed data
for audio_file, features in data_from_mongo:
    track_id = audio_file.split('.')[0].lstrip('0')  # Remove leading zeros
    print("Track ID:", track_id)
    print("Preprocessed data:", features)

