'''
DeepSpeech Audio Transcription (File)
- Supports WAV file input
- Auto-resamples audio to 16kHz for DeepSpeech
- Real-time partial transcription feedback
- Quran-specific: Automatically loads and compares ayah from quran-uthmani.txt
- Returns list of tuples in format (incorrect word, surah, ayah)
'''

import time, logging
from datetime import datetime
import queue, os, wave
import deepspeech
import numpy as np
import pyaudio
from halo import Halo
from scipy import signal
from TextComparator import *

logging.basicConfig(level=20)

class Audio:
    FORMAT = pyaudio.paInt16
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            if callback:
                callback(in_data)
            return (None, pyaudio.paContinue)

        if callback is None:
            callback = lambda in_data: self.buffer_queue.put(in_data)

        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')
            kwargs['rate'] = self.wf.getframerate()
            self.input_rate = self.wf.getframerate()

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        data16 = np.frombuffer(data, dtype=np.int16)
        if len(data16) == 0:
            return b''
        resample_size = int(len(data16) / input_rate * self.RATE_PROCESS)
        resampled = signal.resample(data16, resample_size)
        resampled = np.asarray(resampled, dtype=np.int16)
        return resampled.tobytes()

    def read_resampled(self):
        raw_audio = self.read()
        if len(raw_audio) == 0:
            return b''
        return self.resample(data=raw_audio, input_rate=self.input_rate)

    def read(self):
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

def load_quran_text(file_path):
    quran = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                surah, ayah, text = int(parts[0]), int(parts[1]), parts[2]
                quran[(surah, ayah)] = text
    return quran

def main(ARGS):
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'quran_model.pb')
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    model = deepspeech.Model(ARGS.model)
    if ARGS.scorer:
        model.enableExternalScorer(ARGS.scorer)

    quran = load_quran_text(ARGS.quran_file)
    surah, ayah = ARGS.surah, ARGS.ayah

    def get_next_ayah(surah, ayah):
        if (surah, ayah) in quran:
            return quran[(surah, ayah)], surah, ayah
        else:
            # Move to next surah
            surah += 1
            ayah = 1
            if (surah, ayah) in quran:
                return quran[(surah, ayah)], surah, ayah
            else:
                return None, None, None

    actual_text, surah, ayah = get_next_ayah(surah, ayah)
    if not actual_text:
        print("No ayah found.")
        return
    comparator = TextComparator(actual_text)

    audio = Audio(file=ARGS.file)
    print("Listening (ctrl-C to exit)...")

    spinner = None if ARGS.nospinner else Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()
    incorrect_words = []
    total_words = 0

    last_partial = ""
    same_partial_count = 0
    max_same_count = 25  # number of consecutive times partial is unchanged

    try:
        while True:
            frame = audio.read_resampled()
            if len(frame) == 0:
                break

            if spinner: spinner.start()
            stream_context.feedAudioContent(np.frombuffer(frame, dtype=np.int16))

            partial = stream_context.intermediateDecode().strip()
            if partial:
                if comparator.last_word:
                    same_partial_count = same_partial_count + 1 if partial == last_partial else 0
                    last_partial = partial
                    if same_partial_count >= max_same_count:
                        word_completed = comparator.process_partial(partial + ' ')
                        if word_completed:
                            result = comparator.compare_word()
                            if result[0]:
                                print(f"\nIncorrect word detected: {result[1]}")
                                incorrect_words.append((result[1], surah, ayah))
                            else:
                                print("correct: ", result[1], surah, ayah)
                            print(f"\nAyah ({surah}, {ayah}): {partial}")
                            print(f"Incorrect words: {incorrect_words}")

                            total_words += len(comparator.actual_words)

                            actual_text, new_surah, new_ayah = get_next_ayah(surah, ayah + 1)
                            if not actual_text:
                                print("\nNo next ayah/surah.")
                                break
                            
                            surah, ayah = new_surah, new_ayah
                            comparator = TextComparator(actual_text)
                            stream_context = model.createStream()
                            same_partial_count = 0
                            last_partial = ""
                else:
                    word_completed = comparator.process_partial(partial)
                    if word_completed:
                        result = comparator.compare_word()
                        if result[0]:
                            print(f"\nIncorrect word detected: {result[1]}")
                            incorrect_words.append((result[1], surah, ayah))
                        else:
                            print("correct: ", result[1], surah, ayah)

    except KeyboardInterrupt:
        print("\nInterrupting...")

    finally:
        if spinner: spinner.stop()
        final_text = stream_context.finishStream()
        if final_text:
            last_word_completed = comparator.process_partial(final_text + ' ')
            if last_word_completed:
                result = comparator.compare_word()
                if result[0]:
                    print(f"Incorrect word detected: {result[1]}")
                    incorrect_words.append((result[1], surah, ayah))
                else:
                    print("correct: ", result[1], surah, ayah)

            print(f"\nAyah ({surah}, {ayah}): {final_text}")
        print(f"Incorrect words: {incorrect_words}")
        print(f"Progress rate: {(total_words - len(incorrect_words)) / total_words * 100:.0f}%")
        audio.destroy()        
        print("Processing complete.")

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Real Time Recitation Recognition and Mistake Detection from Audio File")
    parser.add_argument('--nospinner', action='store_true', help="Disable spinner")
    parser.add_argument('-f', '--file', help="WAV file to read from")
    parser.add_argument('-m', '--model', help="Path to model folder")
    parser.add_argument('-s', '--scorer', help="Path to scorer file")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE, help="Input device sample rate")
    parser.add_argument('--quran_file', help="Path to Quran text file")
    parser.add_argument('--surah', type=int, default=78, help="Surah number")
    parser.add_argument('--ayah', type=int, default=1, help="Ayah number")

    ARGS = parser.parse_args()
    main(ARGS)