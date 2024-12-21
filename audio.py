import pyaudio
import numpy as np

# Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 500  # Adjust this threshold based on your environment

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

try:
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        peak = np.abs(data).max()
        if peak > THRESHOLD:
            print("Sound detected!")
except KeyboardInterrupt:
    print("Stopping...")

# Close stream
stream.stop_stream()
stream.close()
p.terminate()