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

    await websocket.send_json({
        "expected_text": expected_text,
        "surah": surah,
        "ayah": ayah
    })

    last_partial = ""
    same_partial_count = 0
    max_same_count = 25
    total_words = 0
    incorrect_words = []

    try:
        while True:
            message = await websocket.receive_bytes()
            audio_frames.append(message)

            audio = np.frombuffer(message, dtype=np.int16)
            stream.feedAudioContent(audio)

            partial = stream.intermediateDecode().strip()

            if partial:
                if comparator.last_word:
                    same_partial_count = same_partial_count + 1 if partial == last_partial else 0
                    last_partial = partial
                    if same_partial_count >= max_same_count:
                        word_completed = comparator.process_partial(partial + ' ')
                        if word_completed:
                            result = comparator.compare_latest_word()
                            if result[0]:
                                await websocket.send_json({
                                    "incorrect_word": result[1],
                                    "surah": surah,
                                    "ayah": ayah
                                })
                                incorrect_words.append((result[1], surah, ayah))
                            else:
                                await websocket.send_json({
                                    "correct_word": result[1],
                                    "surah": surah,
                                    "ayah": ayah
                                })

                            total_words += len(comparator.actual_words)
                            actual_text, new_surah, new_ayah = get_next_ayah(surah, ayah + 1)
                            if actual_text:
                                comparator = TextComparator(actual_text)
                                stream = ds_model.createStream()
                                surah, ayah = new_surah, new_ayah
                                await websocket.send_json({
                                    "expected_text": actual_text,
                                    "surah": surah,
                                    "ayah": ayah
                                })
                                same_partial_count = 0
                                last_partial = ""
                            else:
                                await websocket.send_json({"info": "No next ayah found."})
                                break
                else:
                    word_completed = comparator.process_partial(partial)
                    if word_completed:
                        result = comparator.compare_latest_word()
                        if result[0]:
                            await websocket.send_json({
                                "incorrect_word": result[1],
                                "surah": surah,
                                "ayah": ayah
                            })
                            incorrect_words.append((result[1], surah, ayah))
                        else:
                            await websocket.send_json({
                                "correct_word": result[1],
                                "surah": surah,
                                "ayah": ayah
                            })

    except WebSocketDisconnect:
        print("Client disconnected")

        final_text = stream.finishStream()
        print(f"Final: {final_text}")

        await websocket.send_json({
            "summary": {
                "final_transcription": final_text,
                "progress_rate": f"{(total_words - len(incorrect_words)) / total_words * 100:.0f}%" if total_words > 0 else "0%",
                "incorrect_words": incorrect_words
            }
        })

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