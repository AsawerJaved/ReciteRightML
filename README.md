# Quranic Recitation Recognition with DeepSpeech

This project uses Mozilla's DeepSpeech speech-to-text engine to recognize and transcribe Quranic recitations in Arabic.

## Features
- Real-time audio processing and transcription
- Automatic comparison with actual Quranic verses
- Intermediate transcription display during recitation
- Save recordings as WAV files
- Support for microphone or pre-recorded .wav files

## Installation
Clone this repository

```bash
https://github.com/AsawerJaved/ReciteRightML.git
```

Install dependencies from requirements.txt

   ```bash
   pip install -r requirements.txt
   ```
Download the pre-trained model and scorer files (place them in the models directory)

   - ```quran_model.pb``` (model)
   - ```quran.scorer``` (scorer)

## Usage
1. Basic Usage

   ```bash
   python quran_recognition.py
   ```
3. Command Line Options

   ```bash
   python quran_recognition.py [options]
   ```
   Options:

    ```
      --nospinner                Disable loading spinner
      -f, --file FILE            Path to .wav file to read from
      -m, --model PATH           Path to DeepSpeech model
      -s, --scorer PATH          Path to scorer file
      -r, --rate RATE            Input sample rate (default: 16000)
      --quran_file               Path to Quran text file
      --surah                    Surah number (default: 78 -- surah naba)
      --ayah                     Ayah number (default: 1)
    ```
   ### Example Usage:

   ```bash
   python quran_recognition.py -f audio.wav -m models/quran_model.pb \
   -s models/quran.scorer --quran_file quran/quran-uthmani.txt
   ```
