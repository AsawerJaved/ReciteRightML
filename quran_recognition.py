import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
from difflib import SequenceMatcher

logging.basicConfig(level=20)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
                if len(in_data) == 0:  # End of file
                    return (None, pyaudio.paComplete)
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
        # if not default device
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
        """Resample input audio to 16000Hz for DeepSpeech."""
        data16 = np.frombuffer(data, dtype=np.int16)
        resample_size = int(len(data16) / input_rate * self.RATE_PROCESS)
        resampled = signal.resample(data16, resample_size)
        resampled = np.asarray(resampled, dtype=np.int16)
        return resampled.tobytes()

    def read_resampled(self):
        """Return audio block resampled to 16000Hz."""
        raw_audio = self.read()
        return self.resample(data=raw_audio, input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

class TextComparator:
    def __init__(self, actual_text):
        self.original_actual = actual_text
        self.actual_words = actual_text.split()
        self.incorrect_words = set()
        self.last_complete_words = []  # To track complete words in prediction
        self.current_partial = ""  # To track the current incomplete word
    
    def normalize_word(self, word):
        """Normalize word by removing diacritics only"""
        return ''.join(c for c in word if not (0x064B <= ord(c) <= 0x065F))
    
    def process_partial(self, partial_text):
        """Process partial text and return True if a new complete word is detected"""
        new_words = []
        words = partial_text.split(' ')
        
        # If we have more segments than before, a new word was completed
        if len(words) > len(self.last_complete_words) + 1:
            # All words except last are complete
            new_complete_words = words[:-1]
            self.last_complete_words = new_complete_words
            self.current_partial = words[-1]
            return True
        else:
            # Just update the current partial word
            if words:
                self.current_partial = words[-1]
            return False
    
    def get_current_prediction(self):
        """Get the full current prediction (complete words + current partial)"""
        return ' '.join(self.last_complete_words + [self.current_partial]) if self.current_partial else ' '.join(self.last_complete_words)
    
    def compare_last_word(self):
        """Compare the last complete word with actual text"""
        if not self.last_complete_words:
            return []
        
        # Get the index of the last complete word
        last_word_index = len(self.last_complete_words) - 1
        
        # Check if we have an actual word at this position
        if last_word_index >= len(self.actual_words):
            extra_word = self.last_complete_words[last_word_index]
            self.incorrect_words.add(f"[Extra: {extra_word}]")
            return [extra_word]
        
        actual_word = self.actual_words[last_word_index]
        predicted_word = self.last_complete_words[last_word_index]
        
        if self.normalize_word(actual_word) != self.normalize_word(predicted_word):
            self.incorrect_words.add(actual_word)
            return [actual_word]
        
        return []

def main(ARGS):
    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'quran_model.pb')
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    print('Initializing model...')
    logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model)
    if ARGS.scorer:
        logging.info("ARGS.scorer: %s", ARGS.scorer)
        model.enableExternalScorer(ARGS.scorer)

    # Initialize TextComparator with actual text
    if not ARGS.actual_text:
        print("Error: Please provide --actual_text for comparison")
        return
    comparator = TextComparator(ARGS.actual_text)

    # Start plain audio (no VAD, continuous stream)
    audio = Audio(device=ARGS.device, input_rate=ARGS.rate, file=ARGS.file)
    print("Recording... Press Ctrl+C to stop.")

    # Stream processing
    spinner = None if ARGS.nospinner else Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()

    try:
        while True:
            frame = audio.read_resampled()
            if spinner: spinner.start()

            # Feed audio to model
            audio_frame = np.frombuffer(frame, dtype=np.int16)
            stream_context.feedAudioContent(audio_frame)

            # Get partial transcription
            partial_text = stream_context.intermediateDecode().strip()
            if partial_text:
                new_word_completed = comparator.process_partial(partial_text)
                if new_word_completed:
                    incorrect = comparator.compare_last_word()
                    if incorrect:
                        print(f"\nIncorrect word detected: {incorrect[0]}")
                current_pred = comparator.get_current_prediction()
                print(f"Progress: {current_pred}", end='\r')

            if ARGS.savewav:
                wav_data.extend(frame)

    except KeyboardInterrupt:
        print("\nStopping...")

        # Final result
        if spinner: spinner.stop()
        final_text = stream_context.finishStream()
        print(f"\nFinal recognized: {final_text}")

        # Final comparison
        comparator.process_partial(final_text + ' ')
        incorrect_words = comparator.compare_last_word()

        all_incorrect = []
        for i in range(len(comparator.last_complete_words)):
            if i >= len(comparator.actual_words):
                all_incorrect.append(f"[Extra: {comparator.last_complete_words[i]}]")
            elif (comparator.normalize_word(comparator.actual_words[i]) != 
                  comparator.normalize_word(comparator.last_complete_words[i])):
                all_incorrect.append(comparator.actual_words[i])

        comparator.incorrect_words.update(all_incorrect)

        if all_incorrect:
            print(f"Incorrect words: {', '.join(all_incorrect)}")
        else:
            print("All words matched perfectly!")

        # Save .wav file
        if ARGS.savewav and wav_data:
            filename = datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")
            audio.write_wav(os.path.join(ARGS.savewav, filename), wav_data)

    finally:
        audio.destroy()
        print("Processing complete.")

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 44100

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav', 
                         help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file', 
                        help="Read from .wav file instead of microphone")
    parser.add_argument('-m', '--model', 
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-s', '--scorer', 
                        help="Path to the external scorer file.")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")
    parser.add_argument('-a', '--actual_text', type=str, 
                        help="The actual text to compare against the predicted text")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
