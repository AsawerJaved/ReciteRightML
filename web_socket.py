import os
import time
import wave
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import deepspeech
from scipy import signal
from TextComparator import TextComparator

app = FastAPI()

MODEL_PATH = "quran_model.pb"
SCORER_PATH = "quran.scorer"
QURAN_TEXT_PATH = "quran-uthmani.txt"
TARGET_SAMPLE_RATE = 16000
SAVE_AUDIO_DIR = "saved_audio"

os.makedirs(SAVE_AUDIO_DIR, exist_ok=True)

# Load model once
ds_model = deepspeech.Model(MODEL_PATH)
if SCORER_PATH:
    ds_model.enableExternalScorer(SCORER_PATH)


def load_quran_text(path):
    quran = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                surah, ayah, text = int(parts[0]), int(parts[1]), parts[2]
                quran[(surah, ayah)] = text
    return quran


quran_texts = load_quran_text(QURAN_TEXT_PATH)


def resample_audio(audio_bytes, input_rate):
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(audio) == 0:
        return b''
    resample_size = int(len(audio) / input_rate * TARGET_SAMPLE_RATE)
    resampled = signal.resample(audio, resample_size)
    resampled = np.asarray(resampled, dtype=np.int16)
    return resampled.tobytes()


@app.websocket("/ws/transcribe/{surah}/{ayah}")
async def transcribe_quran(websocket: WebSocket, surah: int, ayah: int):
    await websocket.accept()
    key = (surah, ayah)

    if key not in quran_texts:
        await websocket.send_json({"error": f"Surah {surah} Ayah {ayah} not found"})
        await websocket.close()
        return

    expected_text = quran_texts[key]
    comparator = TextComparator(expected_text)
    stream = ds_model.createStream()
    audio_frames = []

    # Added only for debugging
    await websocket.send_json({
        "expected_text": expected_text,
        "surah": surah,
        "ayah": ayah
    })

    try:
        while True:
            message = await websocket.receive_bytes()
            audio_frames.append(message)

            audio = np.frombuffer(message, dtype=np.int16)
            stream.feedAudioContent(audio)

            partial = stream.intermediateDecode().strip()

            if partial:
                new_word_completed = comparator.process_partial(partial)
                if new_word_completed:
                    incorrect = comparator.compare_latest_word()
                    if incorrect:
                        await websocket.send_json({
                            "incorrect_word": incorrect[0],
                            "surah": surah,
                            "ayah": ayah
                        })

    except WebSocketDisconnect:
        print("Client disconnected")

        final_text = stream.finishStream()
        print(f"Final: {final_text}")

        # Save audio and text for each session only for debugging
        filename = f"{SAVE_AUDIO_DIR}/session_{int(time.time())}.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(b''.join(audio_frames))
        print(f"Saved audio to {filename}")

        os.makedirs("transcripts", exist_ok=True)
        with open(f"transcripts/session_{int(time.time())}.txt", "w", encoding="utf-8") as f:
            f.write(final_text)
