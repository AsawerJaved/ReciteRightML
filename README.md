# Quranic Recitation Recognition with DeepSpeech

This project uses Mozilla's DeepSpeech speech-to-text engine to recognize and transcribe Quranic recitations in Arabic.

## Features
-✅ Real-time audio processing and transcription
-✅ Automatic comparison with actual Quranic verses
-✅ Intermediate transcription display during recitation
-✅ Save recordings as WAV files
-✅ Support for microphone or pre-recorded .wav files

## Installation
1. Clone this repository
2. Install dependencies from requirements.txt <br/>
   ```pip install -r requirements.txt```
3. Download the pre-trained model and scorer files (place them in the models directory)
   - quran_model.pb (model)
   - quran.scorer (scorer)

## Usage
1. Basic Usage <br/>
   ```python quran_recognition.py```
2. Command Line Options <br/>
   ```python quran_recognition.py [options]```
   - Options:
    ```
     -v, --vad_aggressiveness   VAD aggressiveness (0-3, default: 3)
     --nospinner                Disable loading spinner
     -w, --savewav PATH         Save WAV files to directory
     -f, --file FILE            Process WAV file instead of microphone
     -m, --model PATH           Path to DeepSpeech model
     -s, --scorer PATH          Path to scorer file
     -d, --device INDEX         Audio device index
     -r, --rate RATE            Input sample rate (default: 16000)
    ```
   ### Example: <br/>
   ```python quran_recognition.py -m models/quran_model.pb -s models/quran.scorer -w recordings```
