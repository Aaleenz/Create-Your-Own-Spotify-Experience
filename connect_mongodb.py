import bson
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['music_database']
collection = db['music_collection']

# Open BSON file and insert data into MongoDB
with open('/home/asmariaz/Downloads/music_data.bson', 'rb') as f:
    bson_data = bson.decode_all(f.read())
    collection.insert_many(bson_data)

print("Data imported successfully.")
