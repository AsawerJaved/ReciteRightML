# Quranic Recitation Recognition with DeepSpeech

This project uses Mozilla's DeepSpeech speech-to-text engine to recognize and transcribe Quranic recitations in Arabic.

## Features
- Real-time audio processing with Voice Activity Detection (VAD)
- Accurate recognition of Quranic verses using a custom-trained DeepSpeech model
- Intermediate results display during recognition
- WAV file saving capability for recorded utterances

## Installation
1. Clone this repository
2. Install dependencies from requirements.txt <br/>
   ```pip install -r requirements.txt```
3. Download the pre-trained model and scorer files (place them in the models directory)
   - output_graph_imams_tusers_v2.pb (model)
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
   ```python quran_recognition.py -m models/output_graph_imams_tusers_v2.pb -s models/quran.scorer -w recordings```
