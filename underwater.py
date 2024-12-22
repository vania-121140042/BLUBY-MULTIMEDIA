import pyaudio
import numpy as np
import scipy.signal as signal
from pydub import AudioSegment
from pydub.playback import play
import threading

# Constants
SAMPLING_RATE = 44100
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono audio
LOW_PASS_CUT_OFF = 200  # Low pass filter cutoff frequency (Hz)
Q_FACTOR = 0.71  # Quality factor for the low pass filter
REVERB_EFFECT_MIX = 0.5  # Reverb effect mix (50%)
REVERB_EFFECT_LEVEL = 8  # Level of reverb (in dB)
DELAY = 0.03  # Small delay in seconds (e.g., 30ms)

# Create PyAudio instance
p = pyaudio.PyAudio()

# Load the MP3 file using pydub
mp3_file = AudioSegment.from_mp3("underwater-ambience-heavy-rumbling-ftus-1-00-17.mp3")
mp3_file = mp3_file.set_frame_rate(SAMPLING_RATE).set_channels(1).set_sample_width(2)  # Convert to mono, 16-bit

# Low-pass filter function
def low_pass_filter(data, cutoff, fs, Q):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(1, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

# Manual Reverb Effect (simple echo effect)
def add_reverb(audio, mix, level, fs):
    delay_samples = int(0.2 * fs)  # 200ms delay for reverb effect
    reverb_audio = np.zeros_like(audio)
    
    reverb_audio[delay_samples:] = audio[:-delay_samples]
    
    return mix * reverb_audio + (1 - mix) * audio

# Audio delay effect
def apply_delay(audio, delay_time, fs):
    delay_samples = int(delay_time * fs)
    delayed_audio = np.zeros_like(audio)
    delayed_audio[delay_samples:] = audio[:-delay_samples]
    return audio + delayed_audio

# Function to play the MP3 file in a separate thread
def play_bubbling_sound():
    while True:
        play(mp3_file)  # This will loop the bubbling sound indefinitely

# Stream callback function
def callback(in_data, frame_count, time_info, status):
    # Convert byte data to numpy array
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)

    # Apply Low Pass Filter (muddier effect)
    filtered_audio = low_pass_filter(audio_data, LOW_PASS_CUT_OFF, SAMPLING_RATE, Q_FACTOR)

    # Add Reverb effect
    reverb_audio = add_reverb(filtered_audio, REVERB_EFFECT_MIX, REVERB_EFFECT_LEVEL, SAMPLING_RATE)

    # Apply a slight delay (optional)
    final_audio = apply_delay(reverb_audio, DELAY, SAMPLING_RATE)

    # Convert the processed audio back to bytes
    out_data = final_audio.astype(np.int16).tobytes()
    return (out_data, pyaudio.paContinue)

# Open PyAudio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLING_RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=callback)

# Start the MP3 playback thread
bubbling_thread = threading.Thread(target=play_bubbling_sound, daemon=True)
bubbling_thread.start()

# Start stream
print("Processing and playback with underwater effect...")
stream.start_stream()

try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    pass

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()
