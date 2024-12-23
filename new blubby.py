#library yang digunakan
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

# Constants
SAMPLING_RATE = 44100
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono audio
LOW_PASS_CUT_OFF = 200  # Low pass filter cutoff fuency (Hz)
Q_FACTOR = 0.71  # Quality factor for the low pass filter
REVERB_EFFECT_MIX = 0.5  # Reverb effect mix (50%)
REVERB_EFFECT_LEVEL = 8  # Level of reverb (in dB)
DELAY = 0.03  # Small delay in seconds (e.g., 30ms)

# Menonaktifkan log TensorFlow dan Mediapipe
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' hanya menampilkan error, '3' untuk sepenuhnya menonaktifkan log

# Inisialisasi Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Fungsi untuk menghitung jarak Euclidean antara dua titik
def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Fungsi untuk mendeteksi apakah mulut terbuka
def is_mouth_open(landmarks):
    upper_lip = landmarks[13]  # Titik 13 adalah bibir atas
    lower_lip = landmarks[14]  # Titik 14 adalah bibir bawah
    distance = euclidean_distance((upper_lip.x, upper_lip.y), (lower_lip.x, lower_lip.y))
    threshold = 0.02  # Threshold untuk mendeteksi mulut terbuka
    return distance > threshold

# Fungsi untuk menambahkan PNG bubble ke frame
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

# Apply the underwater effect with floating bubbles, blue tint, and wave distortion
def add_underwater_effect(frame, bubbles, bubble_png, time_factor):
    #fungsi efek distorsi laut
    rows, cols, _ = frame.shape
    distortion_map_x, distortion_map_y = np.meshgrid(np.arange(cols), np.arange(rows))

    random_shift = random.uniform(-5, 5)

    distortion_map_x= distortion_map_x + 2 * np.sin(distortion_map_y / 60 + time_factor + random_shift)
    distortion_map_y = distortion_map_y + 2 * np.sin(distortion_map_x / 40 + time_factor)

    distortion_map_x= np.clip(distortion_map_x, 0, cols-1).astype(np.float32)
    distortion_map_y= np.clip(distortion_map_y, 0, rows-1).astype(np.float32)
    frame = cv2.remap(frame, distortion_map_x, distortion_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    #fungsi efect biru laut 
    ocean_blue = frame.copy()
    ocean_blue[:, :, 0] = cv2.add(ocean_blue[:, :, 0], 50)  
    ocean_blue[:, :, 1] = cv2.add(ocean_blue[:, :, 1], 20)  
    ocean_blue[:, :, 2] = cv2.subtract(ocean_blue[:, :, 2], 10)
    frame = ocean_blue.copy()
    
    for bubble in bubbles:
        bubble.move()
        if bubble.active:
            overlay_png(frame, bubble_png, bubble.x, bubble.y, bubble.radius * 2)

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame

#Fungsi untuk load video background
def video():
    background_video = cv2.VideoCapture('video4.mp4')

    #Mencheck apakah video bisa dibuka atau tidak
    if not background_video.isOpened():
        print("Tidak bisa membuka video!")
        return None
    return background_video

# Fungsi untuk mendapatkan frame video
def video_frame(background_video, frame_shape):
    success, bg_frame = background_video.read()
    
    #Membaca frame satu persatu sampai selesai 
    if not success:
        background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, bg_frame = background_video.read()
    
    # Resize frame video agar sesuai dengan ukuran frame kamera
    bg_frame = cv2.resize(bg_frame, (frame_shape[1], frame_shape[0]))
    
    return bg_frame

# Fungsi menambah background
def add_background(frame, video_background):

    #Resize background agar sesuai dengan ukuran frame
    background_resized = cv2.resize(video_background, (frame.shape[1], frame.shape[0]))
    
    #Membuat mask untuk menngambil objek dengaan background
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    #Menggunakan module selfie_segmentation dari mediapipe untuk mendapatkan mask
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    # Proses frame untuk mendapatkan mask
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segment.process(rgb_frame)
    
    #cek apakah mask ada atau tidak
    if results.segmentation_mask is not None:
        mask = (results.segmentation_mask > 0.1).astype(np.uint8) * 255
        
    # Gabungkan background dengan frame asli
    mask_3d = np.stack((mask,) * 3, axis=-1) / 255.0
    frame_final = frame * mask_3d + background_resized * (1 - mask_3d)
    
    return frame_final.astype(np.uint8)

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
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    filtered_audio = low_pass_filter(audio_data, LOW_PASS_CUT_OFF, SAMPLING_RATE, Q_FACTOR)
    reverb_audio = add_reverb(filtered_audio, REVERB_EFFECT_MIX, REVERB_EFFECT_LEVEL, SAMPLING_RATE)
    final_audio = apply_delay(reverb_audio, DELAY, SAMPLING_RATE)
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
bubbling_thread = threading.Thread(target=play_bubbling_sound)
bubbling_thread.start()

# Fungsi utama untuk menjalankan face mesh dengan deteksi mulut, efek gelembung, backgorund, dan efek suara
def face_mesh_mouth_detection_with_bubbles():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Gagal membuka kamera!")
        return
    
    bubble_png = cv2.imread('gelembung2.png', cv2.IMREAD_UNCHANGED)
    background_video = video()
    frame_shape = (720, 1280)
    
    bubbles = [Bubble(random.randint(0, 640), random.randint(480, 500), 
               random.randint(5, 15), random.uniform(0.5, 3)) for _ in range(30)]
    time_factor = 0
    last_bubble_time = 0
    bubble_delay = 0.5
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame!")
            break

        frame = cv2.flip(frame, 1)
        
        bg_frame = video_frame(background_video, frame_shape)
        frame_with_bg = add_background(frame, bg_frame)

        frame_rgb = cv2.cvtColor(frame_with_bg, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #mp_drawing.draw_landmarks(frame_with_bg, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                # code di atas untuk meanmpilkan landmark wajah pas oncam. Cuman untuk final code, di comment aja jadinya 
                
                landmarks = face_landmarks.landmark
                if is_mouth_open(landmarks):
                    upper_lip = landmarks[13]
                    lower_lip = landmarks[14]
                    mouth_center = ((upper_lip.x + lower_lip.x) / 2, 
                                  (upper_lip.y + lower_lip.y) / 2)
                    
                    current_time = time.time()
                    if current_time - last_bubble_time > bubble_delay:
                        bubbles.append(Bubble(
                            int(mouth_center[0] * frame_with_bg.shape[1]),
                            int(mouth_center[1] * frame_with_bg.shape[0]),
                            40, random.uniform(0.5, 3)))
                        last_bubble_time = current_time
                        
                    cv2.putText(frame_with_bg, "Mouth Open", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame_with_bg, "Mouth Closed", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        underwater_frame = add_underwater_effect(frame_with_bg, bubbles, 
                                              bubble_png, time_factor)
        time_factor += 0.1
        
        cv2.imshow("Face Mesh - Mouth Detection with Bubbles", underwater_frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if bubbling_thread is None or not bubbling_thread.is_alive():
            stop_event.clear()
            bubbling_thread= threading.Thread(target=play_audio, args=("underwater-ambience-heavy-rumbling-ftus-1-00-17.mp3", stop_event))
            bubbling_thread.start()
    
    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    p.terminate()

# Jalankan program
face_mesh_mouth_detection_with_bubbles()