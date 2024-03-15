from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import pickle
from keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('./finalModel.h5')
with open('./scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to extract audio features
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    frame_length = 3 * sr  # 3 seconds
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length)
    all_features = []

    for frame in frames.T:
        chroma_stft = librosa.feature.chroma_stft(y=frame, sr=sr)
        rms = librosa.feature.rms(y=frame)
        spectral_centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=frame, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=frame)
        harmony = librosa.effects.harmonic(y=frame)
        perceptr = librosa.feature.spectral_flatness(y=frame)
        tempo, _ = librosa.beat.beat_track(y=frame, sr=sr)
        mfccs = librosa.feature.mfcc(y=frame, sr=sr)

        features = []
        features.append(np.mean(chroma_stft))
        features.append(np.var(chroma_stft))
        features.append(np.mean(rms))
        features.append(np.var(rms))
        features.append(np.mean(spectral_centroid))
        features.append(np.var(spectral_centroid))
        features.append(np.mean(spectral_bandwidth))
        features.append(np.var(spectral_bandwidth))
        features.append(np.mean(spectral_rolloff))
        features.append(np.var(spectral_rolloff))
        features.append(np.mean(zero_crossing_rate))
        features.append(np.var(zero_crossing_rate))
        features.append(np.mean(harmony))
        features.append(np.var(harmony))
        features.append(np.mean(perceptr))
        features.append(np.var(perceptr))
        features.append(tempo)

        for i in range(1, 21):
            features.append(np.mean(mfccs[i-1]))
            features.append(np.var(mfccs[i-1])) 
        
        all_features.append(features)

    return all_features

# Function to get the prediction
def get_pred(feature):
    genre = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    feature = np.array(feature)
    examp = feature.reshape(1, -1)
    examp = scaler.transform(examp)
    preds = model.predict(examp)
    preds = preds.flatten()
    gen = genre[np.where(preds==max(preds))[0][0]]
    return gen

@app.route('/')
def hello():
    return 'hello!'

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'myfile' not in request.files:
        return 'No file part'
    myfile = request.files['myfile']
    if myfile.filename == '':
        return 'No selected file'
    features = extract_features(myfile)
    arr = [get_pred(feature) for feature in features]
    ans = max(set(arr), key=arr.count)
    return jsonify({'prediction': ans})

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=True for development purposes
