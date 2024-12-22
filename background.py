import cv2
import mediapipe as mp
import math
import time
import random
import numpy as np

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

# Apply the underwater effect with floating bubbles
def add_underwater_effect(frame, bubbles, bubble_png, time_factor):
    overlay = frame.copy()
    overlay[:, :, 0] = cv2.addWeighted(overlay[:, :, 0], 1, np.full_like(overlay[:, :, 0], 50), 0.5, 0)  # Blue
    overlay[:, :, 1] = cv2.addWeighted(overlay[:, :, 1], 1, np.full_like(overlay[:, :, 1], 30), 0.5, 0)  # Green
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    rows, cols, _ = frame.shape
    distortion_map_x = np.tile(np.linspace(0, cols - 1, cols), (rows, 1)).astype(np.float32)
    distortion_map_y = np.tile(np.linspace(0, rows - 1, rows), (cols, 1)).T.astype(np.float32)
    random_shift = random.uniform(-5, 5)
    sinusoid = 10 * np.sin(2 * np.pi * (distortion_map_y / 180 + time_factor) + random_shift).astype(np.float32)
    distortion_map_x += sinusoid

    for bubble in bubbles:
        bubble.move()
        if bubble.active:
            overlay_png(frame, bubble_png, bubble.x, bubble.y, bubble.radius * 2)

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame

# Fungsi menambah background underwater
def add_background(frame, image_background):

    #Resize background agar sesuai dengan ukuran frame
    background_resized = cv2.resize(image_background, (frame.shape[1], frame.shape[0]))
    
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

# Fungsi utama untuk menjalankan face mesh dengan deteksi mulut, efek gelembung, backgorund, dan efek underwater
def face_mesh_mouth_detection_with_bubbles():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Gagal membuka kamera!")
        return
    
    #Ambil gambar background dan gelembung
    image_background = cv2.imread('sea.jpg')  
    bubble_png = cv2.imread('bubble.png', cv2.IMREAD_UNCHANGED)
    
    bubbles = [Bubble(random.randint(0, 640), random.randint(480, 500), 
               random.randint(5, 15), random.uniform(0.5, 3)) for _ in range(30)]
    time_factor = 0
    last_bubble_time = 0
    bubble_delay = 1
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame!")
            break
            
        frame = cv2.flip(frame, 1)
        
        frame_with_bg = add_background(frame, image_background)

        frame_rgb = cv2.cvtColor(frame_with_bg, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame_with_bg, face_landmarks, 
                                       mp_face_mesh.FACEMESH_CONTOURS)
                
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
    
    cap.release()
    cv2.destroyAllWindows()

# Jalankan program
face_mesh_mouth_detection_with_bubbles()