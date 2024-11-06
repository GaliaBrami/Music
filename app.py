import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import aubio
import numpy as np
import librosa
from pydub import AudioSegment
import noisereduce as nr
from scipy.io.wavfile import write
import tempfile

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins, or specify your frontend IP


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded file
    file = request.files['audio']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Process the file to detect notes
    notes = detect_notes_with_fft(file_path)
    # notes = detect_notes(file_path)

    # Remove the file after processing
    os.remove(file_path)

    return jsonify({"notes": notes})
import librosa
import numpy as np
import noisereduce as nr
from scipy.io import wavfile

def detect_notes(audio_file):
    # Step 1: Load audio file and reduce background noise
    sample_rate, audio_data = wavfile.read(audio_file)
    
    # Apply noise reduction
    reduced_noise_data = nr.reduce_noise(y=audio_data, sr=sample_rate)

    # Save the reduced noise data to a temporary file if needed
    # wavfile.write("temp_reduced_noise.wav", sample_rate, reduced_noise_data)

    # Step 2: Use Librosa for pitch detection on the noise-reduced data
    y = reduced_noise_data.astype(np.float32)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sample_rate)

    detected_notes = []
    # Loop through the pitch matrix and extract the most prominent pitch for each frame
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()  # Find the frequency with the highest magnitude
        pitch = pitches[index, t]
        
        if pitch > 0:  # If a pitch is detected
            # Convert the pitch frequency to the nearest musical note
            note = librosa.hz_to_note(pitch)
            detected_notes.append(note)

    # Clean up detected notes by removing duplicates and preserving the sequence
    unique_notes = []
    for note in detected_notes:
        if not unique_notes or unique_notes[-1] != note:
            unique_notes.append(note)

    return unique_notes

# def detect_notes(file_path):

#     # if file_path.endswith('.m4a'):
#     #     audio = AudioSegment.from_file(file_path, format="m4a")
#     #     file_path = file_path.replace('.m4a', '.wav')
#     #     audio.export(file_path, format="wav")

#     # Load the audio file
#     y, sr = librosa.load(file_path, sr=None)
    
#     # Pitch detection using librosa
#     pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
#     detected_notes = []

#     for pitch, magnitude in zip(pitches, magnitudes):
#         # Filter out pitches with low magnitude
#         if magnitude.any() > 0:
#             pitch_value = pitch[magnitude.argmax()]
#             if pitch_value > 0:  # Ignore non-zero pitches
#                 detected_notes.append(librosa.hz_to_note(pitch_value))
    
#     # Remove duplicates for cleaner results
#     return list(set(detected_notes))

import numpy as np
import librosa
from scipy.fft import fft, fftfreq

# def detect_notes_with_fft(audio_file_path):
#     # Load the audio file
#     y, sr = librosa.load(audio_file_path)
    
#     # Set up parameters for FFT
#     frame_size = 2048  # Number of samples per frame
#     hop_size = 512     # Number of samples between frames
#     threshold = 0.1    # Threshold for filtering out background noise
    
#     # Frequency to note mappings (A4 = 440 Hz)
#     note_frequencies = {
#         'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
#         'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
#         'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
#     }
#     # Tolerance range for identifying close frequencies to a note
#     tolerance = 2.0  # Hz
    
#     detected_notes = []
    
#     # Loop through each frame of the audio signal
#     for start in range(0, len(y) - frame_size, hop_size):
#         # Extract frame and apply FFT
#         frame = y[start:start + frame_size]
#         windowed_frame = frame * np.hamming(frame_size)
#         fft_result = np.abs(fft(windowed_frame))[:frame_size // 2]  # Only positive frequencies
#         freqs = fftfreq(frame_size, 1 / sr)[:frame_size // 2]
        
#         # Find the peak frequency in this frame
#         peak_idx = np.argmax(fft_result)
#         peak_freq = freqs[peak_idx]
        
#         # Skip if peak is below threshold
#         if fft_result[peak_idx] < threshold * max(fft_result):
#             continue
        
#         # Map peak frequency to the closest note
#         closest_note = min(note_frequencies, key=lambda note: abs(peak_freq - note_frequencies[note]))
#         # Ensure detected frequency is close to a note's frequency
#         if abs(peak_freq - note_frequencies[closest_note]) < tolerance:
#             detected_notes.append(closest_note)

#     # Consolidate consecutive same notes
#     final_notes = []
#     for i, note in enumerate(detected_notes):
#         if i == 0 or note != detected_notes[i - 1]:
#             final_notes.append(note)

#     return final_notes

import numpy as np
from scipy.fft import fft
from scipy.io import wavfile
from collections import Counter

# def detect_notes_with_fft(audio_path):
#     # Read audio file and get sampling rate
#     sample_rate, data = wavfile.read(audio_path)
    
#     # If stereo, convert to mono by averaging channels
#     if len(data.shape) == 2:
#         data = data.mean(axis=1)
    
#     # Define the FFT window size and overlap
#     window_size = 8192
#     hop_size = window_size // 2
    
#     # List to store detected notes
#     detected_notes = []

#     # Loop through audio in chunks
#     for start in range(0, len(data) - window_size, hop_size):
#         chunk = data[start:start + window_size]
        
#         # Apply FFT
#         spectrum = np.abs(fft(chunk))[:window_size // 2]
        
#         # Get the frequency of the peak
#         peak_freq = np.argmax(spectrum) * sample_rate / window_size

#         # Convert frequency to note
#         note = frequency_to_note(peak_freq)
        
#         # Only add if it’s a new note or a strong frequency
#         if note and (len(detected_notes) == 0 or note != detected_notes[-1]):
#             detected_notes.append(note)

#     # Return only unique notes in order of appearance
#     return detected_notes

# def frequency_to_note(freq):
#     # Mapping frequency to musical notes (simplified version)
#     A4 = 440.0
#     note_names = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
#     if freq <= 0:
#         return None
#     semitones_from_A4 = int(round(12 * np.log2(freq / A4)))
#     octave = 4 + (semitones_from_A4 // 12)
#     note_index = semitones_from_A4 % 12
#     return f"{note_names[note_index]}{octave}"

import numpy as np
import librosa
from scipy.fft import fft, fftfreq

def detect_notes_with_fft(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path)
    
    # Parameters for FFT
    frame_size = 2048
    hop_size = 512
    threshold = 0.1
    tolerance = 2.0  # Hz

    # Hanning window for better spectral resolution
    window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, frame_size, False)))

    # Frequency to note mappings
    note_frequencies = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }

    detected_notes = []

    # First pass: determine max amplitude for scaling
    max_amplitude = 0
    for start in range(0, len(y) - frame_size, hop_size):
        frame = y[start:start + frame_size]
        fft_result = np.abs(fft(frame * window)[:frame_size // 2])
        max_amplitude = max(max_amplitude, np.max(fft_result))

    # Second pass: detect notes with scaled FFT
    for start in range(0, len(y) - frame_size, hop_size):
        frame = y[start:start + frame_size]
        windowed_frame = frame * window
        fft_result = np.abs(fft(windowed_frame))[:frame_size // 2] / max_amplitude
        freqs = fftfreq(frame_size, 1 / sr)[:frame_size // 2]

        # Peak detection with threshold
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]

        if fft_result[peak_idx] >= threshold:
            # Closest note matching within tolerance
            closest_note = min(note_frequencies, key=lambda note: abs(peak_freq - note_frequencies[note]))
            if abs(peak_freq - note_frequencies[closest_note]) < tolerance:
                detected_notes.append(closest_note)

    # Consolidate consecutive same notes
    final_notes = []
    for i, note in enumerate(detected_notes):
        if i == 0 or note != detected_notes[i - 1]:
            final_notes.append(note)

    return final_notes


if __name__ == '__main__':
    # Ensure the uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
