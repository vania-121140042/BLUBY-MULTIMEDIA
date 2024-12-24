import cv2
import mediapipe as mp
import math
import time
import random
import numpy as np
import pyaudio
import scipy.signal as signal
from pydub import AudioSegment
from pydub.playback import play
import threading

# Flag to indicate when to stop the application
stop_flag = threading.Event()

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

# Disable TensorFlow and Mediapipe logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' to show only errors, '3' to disable logs completely

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Create PyAudio instance
p = pyaudio.PyAudio()

# Load the MP3 file using pydub
mp3_file = AudioSegment.from_mp3("underwater-ambience-heavy-rumbling-ftus-1-00-17.mp3")
mp3_file = mp3_file.set_frame_rate(SAMPLING_RATE).set_channels(1).set_sample_width(2)  # Convert to mono, 16-bit


# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Function to detect if the mouth is open
def is_mouth_open(landmarks):
    upper_lip = landmarks[13]  # Upper lip point
    lower_lip = landmarks[14]  # Lower lip point
    distance = euclidean_distance((upper_lip.x, upper_lip.y), (lower_lip.x, lower_lip.y))
    threshold = 0.02  # Threshold for detecting open mouth
    return distance > threshold

# Function to overlay PNG image (bubble) onto the frame
def overlay_png(frame, bubble_png, x, y, size):
    bubble_resized = cv2.resize(bubble_png, (size, size), interpolation=cv2.INTER_AREA)
    b_h, b_w, _ = bubble_resized.shape
    alpha = bubble_resized[:, :, 3] / 255.0
    bubble_rgb = bubble_resized[:, :, :3]
    
    y1, y2 = max(0, y - b_h // 2), min(frame.shape[0], y + b_h // 2)
    x1, x2 = max(0, x - b_w // 2), min(frame.shape[1], x + b_w // 2)
    b_y1 = max(0, -y + b_h // 2)
    b_y2 = b_y1 + (y2 - y1)
    b_x1 = max(0, -x + b_w // 2)
    b_x2 = b_x1 + (x2 - x1)

    if y1 < y2 and x1 < x2 and b_y1 < b_y2 and b_x1 < b_x2:
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (
                frame[y1:y2, x1:x2, c] * (1 - alpha[b_y1:b_y2, b_x1:b_x2]) +
                bubble_rgb[b_y1:b_y2, b_x1:b_x2, c] * alpha[b_y1:b_y2, b_x1:b_x2]
            )

# Bubble class to manage individual bubbles
class Bubble:
    def __init__(self, x, y, radius, delay):
        self.x = x
        self.y = y
        self.radius = radius
        self.delay = delay  # Time delay for the bubble to appear
        self.start_time = time.time()  # Record the start time
        self.active = False  # Whether the bubble is active/visible

    def move(self):
        if not self.active and (time.time() - self.start_time) > self.delay:
            self.active = True
        if self.active:
            self.y -= random.randint(1, 5)
            if self.y < -self.radius:
                self.reset()

    def reset(self):
        self.y = random.randint(480, 500)
        self.x = random.randint(0, 640)
        self.radius = random.randint(5, 15)
        self.delay = random.uniform(0.5, 3)
        self.start_time = time.time()
        self.active = False


# Apply the underwater effect with floating bubbles and more pronounced camera distortion
def add_underwater_effect(frame, bubbles, bubble_png, time_factor):
    overlay = frame.copy()

    # Apply a basic blue filter: Boosting the blue channel
    overlay[:, :, 0] = cv2.add(overlay[:, :, 0], 100)  # Enhance blue (B)
    overlay[:, :, 1] = cv2.subtract(overlay[:, :, 1], 50)  # Reduce green (G)
    overlay[:, :, 2] = cv2.subtract(overlay[:, :, 2], 50)  # Reduce red (R)

    # Ensure that values stay within the valid range (0-255)
    overlay = np.clip(overlay, 0, 255)

    # Create the distortion effect using a slow, sinusoidal wave
    rows, cols, _ = frame.shape
    distortion_map_x = np.tile(np.linspace(0, cols - 1, cols), (rows, 1)).astype(np.float32)
    distortion_map_y = np.tile(np.linspace(0, rows - 1, rows), (cols, 1)).T.astype(np.float32)

    # Sinusoidal distortion (more pronounced wave effect)
    wave_amplitude = 15  # Increase amplitude for more pronounced distortion
    wave_frequency = 0.01  # Frequency of the wave
    wave_speed = 0.05  # Speed of the wave over time (keep the same speed)

    distortion_map_x += wave_amplitude * np.sin(wave_frequency * distortion_map_y + time_factor * wave_speed).astype(np.float32)
    distortion_map_y += wave_amplitude * np.cos(wave_frequency * distortion_map_x + time_factor * wave_speed).astype(np.float32)

    # Apply the distortion map to the frame
    distorted_frame = cv2.remap(overlay, distortion_map_x, distortion_map_y, interpolation=cv2.INTER_LINEAR)

    # Add bubbles to the distorted frame
    for bubble in bubbles:
        bubble.move()
        if bubble.active:
            overlay_png(distorted_frame, bubble_png, bubble.x, bubble.y, bubble.radius * 2)

    # Apply blur to the frame after adding bubbles
    distorted_frame = cv2.GaussianBlur(distorted_frame, (5, 5), 0)
    
    return distorted_frame


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
    while not stop_flag.is_set():  # Check for the stop flag
        play(mp3_file)  # This will loop the bubbling sound indefinitely


# Stream callback function
def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    filtered_audio = low_pass_filter(audio_data, LOW_PASS_CUT_OFF, SAMPLING_RATE, Q_FACTOR)
    reverb_audio = add_reverb(filtered_audio, REVERB_EFFECT_MIX, REVERB_EFFECT_LEVEL, SAMPLING_RATE)
    final_audio = apply_delay(reverb_audio, DELAY, SAMPLING_RATE)
    out_data = final_audio.astype(np.int16).tobytes()
    return (out_data, pyaudio.paContinue)

# Function to process audio
def process_audio():
    # Open PyAudio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLING_RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=callback)
    # Start the audio stream
    stream.start_stream()
    
    # Keep the stream running until the stop flag is set
    while not stop_flag.is_set():
        time.sleep(0.1)

    # Stop the stream and close PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

# Start the video and audio processing
# Function to process video and handle mouth detection
def face_mesh_mouth_detection_with_audio():
    # Start the audio processing thread
    audio_thread = threading.Thread(target=process_audio)
    audio_thread.start()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera!")
        stop_flag.set()  # Signal to stop the audio thread
        return

    # Load assets from the 'assets/' directory
    bubble_png = cv2.imread('assets/bubble.png', cv2.IMREAD_UNCHANGED)

    bubbles = [Bubble(random.randint(0, 640), random.randint(480, 500), 
               random.randint(5, 15), random.uniform(0.5, 3)) for _ in range(30)]
    time_factor = 0
    last_bubble_time = 0
    bubble_delay = 1

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame!")
            break
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        
        # Removed the background change (background image is no longer applied)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                landmarks = face_landmarks.landmark
                if is_mouth_open(landmarks):
                    upper_lip = landmarks[13]
                    lower_lip = landmarks[14]
                    mouth_center = ((upper_lip.x + lower_lip.x) / 2, 
                                    (upper_lip.y + lower_lip.y) / 2)
                    current_time = time.time()
                    if current_time - last_bubble_time > bubble_delay:
                        bubbles.append(Bubble(
                            int(mouth_center[0] * frame.shape[1]),
                            int(mouth_center[1] * frame.shape[0]),
                            40, random.uniform(0.5, 3)))
                        last_bubble_time = current_time
                    cv2.putText(frame, "Mouth Open", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Mouth Closed", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add underwater effect (bubbles)
        underwater_frame = add_underwater_effect(frame, bubbles, bubble_png, time_factor)
        time_factor += 0.1
        
        cv2.imshow("Face Mesh - Mouth Detection with Bubbles", underwater_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()  # Signal to stop the audio thread
            break
    
    cap.release()
    cv2.destroyAllWindows()
    stop_flag.set()  # Ensure the audio thread stops if it hasn't already

# Start the MP3 playback thread
bubbling_thread = threading.Thread(target=play_bubbling_sound)
bubbling_thread.start()

# Start the video processing
face_mesh_mouth_detection_with_audio()
