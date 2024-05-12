from flask import Flask, render_template, send_file
from pymongo import MongoClient
from importing_data import convert_to_mp3

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['music_db']
collection = db['music_col']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream_audio')
def stream_audio():
    track_id = '62569'  # Replace with the actual track_id extracted from the filename
    document = collection.find_one({'track_id': track_id})
    
    if document and 'features' in document:
        audio_data = document['features']
        audio_bytes = convert_to_mp3(audio_data)
        return send_file(audio_bytes, mimetype='audio/mp3')
    else:
        return 'Error: Audio data not found'
        

if __name__ == '__main__':
    app.run(debug=True)

