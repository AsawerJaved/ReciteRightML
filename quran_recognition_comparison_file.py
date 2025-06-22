'''
DeepSpeech Audio Transcription (File & Mic)
- Supports both microphone and WAV file input
- Auto-resamples audio to 16kHz for DeepSpeech
- Real-time partial transcription feedback
- Saves recordings to WAV (optional)
- Better error handling & cleanup
'''

import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
from halo import Halo
from scipy import signal

logging.basicConfig(level=20)

class Audio(object):
    """Streams raw audio from microphone or file. Data is received in a separate thread, and stored in a buffer."""

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
        """Resamples audio data to target processing rate (16kHz)."""

        data16 = np.frombuffer(data, dtype=np.int16)
        if len(data16) == 0:
            return b''
        resample_size = int(len(data16) / input_rate * self.RATE_PROCESS)
        resampled = signal.resample(data16, resample_size)
        resampled = np.asarray(resampled, dtype=np.int16)
        return resampled.tobytes()

    def read_resampled(self):
        """Reads and resamples audio data from buffer."""

        raw_audio = self.read()
        if len(raw_audio) == 0:
            return b''
        return self.resample(data=raw_audio, input_rate=self.input_rate)

    def read(self):
        return self.buffer_queue.get()

    def destroy(self):
        """Releases audio resources and cleans up streams."""

        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def write_wav(self, filename, data):
        """Saves audio data to WAV file."""

        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

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

    # Start audio stream
    audio = Audio(file=ARGS.file)
    print("Listening (ctrl-C to exit)...")

    spinner = None if ARGS.nospinner else Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()

    try:
        while True:
            frame = audio.read_resampled()
            if len(frame) == 0:
                break  # End of file

            if spinner: spinner.start()

            stream_context.feedAudioContent(np.frombuffer(frame, dtype=np.int16))

            partial = stream_context.intermediateDecode().strip()
            if partial:
                print(f"Partial: {partial}", end='\r')

            if ARGS.savewav:
                wav_data.extend(frame)

        if spinner: spinner.stop()
        text = stream_context.finishStream()
        print(f"\nRecognized: {text}")

        if ARGS.savewav and wav_data:
            filename = datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S.wav")
            audio.write_wav(os.path.join(ARGS.savewav, filename), wav_data)

    except KeyboardInterrupt:
        print("\nInterrupted. Finalizing...")
        if spinner:
            spinner.stop()
        text = stream_context.finishStream()
        print(f"\nRecognized: {text}")

    finally:
        audio.destroy()
        print("Processing complete.")

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 44100

    import argparse
    parser = argparse.ArgumentParser(description="Stream from mic or audio file")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
                         help="Save .wav file of the stream")
    parser.add_argument('-f', '--file', 
                        help="WAV file to read instead of microphone")
    parser.add_argument('-m', '--model', 
                        help="Path to model folder or model .pb file")
    parser.add_argument('-s', '--scorer', 
                        help="Path to scorer file")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate (default: {DEFAULT_SAMPLE_RATE})")

    ARGS = parser.parse_args()
    if ARGS.savewav:
        os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
