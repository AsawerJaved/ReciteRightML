# old code (working)

from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

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


def load_quran_text(file_path):
    quran = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                surah, ayah, text = int(parts[0]), int(parts[1]), parts[2]
                quran[(surah, ayah)] = text
    return quran

def get_next_ayah(surah, ayah):
    if (surah, ayah) in quran_texts:
        return quran_texts[(surah, ayah)], surah, ayah
    else:
        # Move to next surah
        surah += 1
        ayah = 1
        if (surah, ayah) in quran_texts:
            return quran_texts[(surah, ayah)], surah, ayah
        else:
            return None, None, None
        
quran_texts = load_quran_text(QURAN_TEXT_PATH)

# MongoDB setup
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client["recite-right"]
session_collection = db["recitation_session"]


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

    session_id = f"session_{int(time.time())}"

    await websocket.send_json({
        "session_id": session_id,
        "expected_text": expected_text,
        "surah": surah,
        "ayah": ayah
    })

    last_partial = ""
    same_partial_count = 0
    max_same_count = 3
    total_words = 0
    incorrect_words = []

    try:
        while True:
            raw_msg = await websocket.receive()
            # Check if it's text (JSON from frontend)
            if "text" in raw_msg:
                import json
                try:
                    message = json.loads(raw_msg["text"])
                    if message.get("type") == "end":
                        break
                except Exception as e:
                    print("JSON decode error:", e)

            # If it's binary, treat it as audio
            elif "bytes" in raw_msg:
                audio_frames.append(raw_msg["bytes"])
                resampled_bytes = resample_audio(raw_msg["bytes"], input_rate=TARGET_SAMPLE_RATE)  # or detect actual rate
                audio = np.frombuffer(resampled_bytes, dtype=np.int16)
                stream.feedAudioContent(audio)

                partial = stream.intermediateDecode().strip()
                if partial:
                    if comparator.last_word:
                        same_partial_count = same_partial_count + 1 if partial == last_partial else 0
                        last_partial = partial
                        if same_partial_count >= max_same_count:
                            word_completed = comparator.process_partial(partial + ' ')
                            if word_completed:
                                incorrect = comparator.compare_word()
                                if incorrect:
                                    await websocket.send_json({
                                        "incorrect_word": incorrect,
                                        "surah": surah,
                                        "ayah": ayah
                                    })
                                    incorrect_words.append((incorrect, surah, ayah))

                                print(f"\nAyah ({surah}, {ayah}): {partial}")
                                print(f"Incorrect words: {incorrect_words}")

                                total_words += len(comparator.actual_words)
                                actual_text, new_surah, new_ayah = get_next_ayah(surah, ayah + 1)
                                if actual_text:
                                    comparator = TextComparator(actual_text)
                                    stream = ds_model.createStream()
                                    surah, ayah = new_surah, new_ayah
                                    same_partial_count = 0
                                    last_partial = ""
                                else:
                                    await websocket.send_json({"info": "No next ayah found."})
                                    break
                                
                    else:
                        word_completed = comparator.process_partial(partial)
                        if word_completed:
                            incorrect = comparator.compare_word()
                            if incorrect:
                                await websocket.send_json({
                                    "incorrect_word": incorrect,
                                    "surah": surah,
                                    "ayah": ayah
                                })
                                incorrect_words.append((incorrect, surah, ayah))
        #  Final processing after user sends "end"
        final_text = stream.finishStream()
        if final_text:
            print("final text: ", final_text)
            last_word_completed = comparator.process_partial(final_text + ' ')
            if last_word_completed:
                incorrect = comparator.compare_word()
                if incorrect:
                    await websocket.send_json({
                        "incorrect_word": incorrect,
                        "surah": surah,
                        "ayah": ayah
                    })
                    incorrect_words.append((incorrect, surah, ayah))

        progress = (total_words - len(incorrect_words)) / total_words * 100 if total_words > 0 else 0

        # 💾 Mongo insert
        start_surah, start_ayah = key
        end_surah, end_ayah = surah, ayah

        detailed_mistakes = [
            {
                "word": word,
                "surah": s,
                "ayah": a,
                "index": None
            }
            for word, s, a in incorrect_words
        ]

        session_doc = {
            "sessionId": session_id,
            "sessionDate": datetime.utcnow(),
            "mistakeCount": len(incorrect_words),
            "progressRate": progress,
            
            "surahRange": {
                "start": {"surah": start_surah, "ayah": start_ayah},
                "end": {"surah": end_surah, "ayah": end_ayah}
            },
            "mistakes": detailed_mistakes
        }

        await session_collection.insert_one(session_doc)

        await websocket.send_json({
            "type": "summary",
            "incorrect_words": incorrect_words,
            "progress_rate": f"{progress:.0f}%"
        })

        print("Saved to MongoDB")

        # Save audio
        filename = f"{SAVE_AUDIO_DIR}/session_{int(time.time())}.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(b''.join(audio_frames))
        print(f"Saved audio to {filename}")

        # Save transcript
        os.makedirs("transcripts", exist_ok=True)
        with open(f"transcripts/session_{int(time.time())}.txt", "w", encoding="utf-8") as f:
            f.write(final_text)

        await websocket.close()

    except WebSocketDisconnect:
        print("Client disconnected (unexpectedly)")

# uvicorn web_socket:app --host localhost