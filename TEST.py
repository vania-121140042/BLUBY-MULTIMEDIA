import cv2
import mediapipe as mp
import math
import time

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
    # Landmark mulut (bagian atas dan bawah bibir)
    upper_lip = landmarks[13]  # Titik 13 adalah bibir atas (bisa juga 0-17)
    lower_lip = landmarks[14]  # Titik 14 adalah bibir bawah

    # Hitung jarak antara bibir atas dan bawah
    distance = euclidean_distance((upper_lip.x, upper_lip.y), (lower_lip.x, lower_lip.y))

    # Tentukan threshold untuk menentukan apakah mulut terbuka
    threshold = 0.02  # Threshold ini bisa disesuaikan (lebih besar = lebih mudah mendeteksi mulut terbuka)

    if distance > threshold:
        return True
    else:
        return False

# Fungsi utama untuk mendeteksi wajah dan status mulut terbuka
def face_mesh_mouth_detection_with_bubbles():
    # Membuka webcam
    cap = cv2.VideoCapture(0)

    # Periksa apakah webcam dibuka
    if not cap.isOpened():
        print("Gagal membuka kamera!")
        return

    # List untuk menyimpan status setiap bubble
    bubbles = []
    bubble_radius = 20  # Ukuran bubble
    bubble_speed = 2    # Kecepatan gerakan bubble (float)
    
    # Timer untuk memberikan delay antara bubble spawn
    last_bubble_time = 0
    bubble_delay = 1  # Delay waktu dalam detik antara bubble spawn

    # Looping utama
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame!")
            break

        # Flip frame agar seperti cermin
        frame = cv2.flip(frame, 1)
        # Mengubah frame ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deteksi wajah menggunakan Mediapipe Face Mesh
        results = face_mesh.process(frame_rgb)

        # Gambar deteksi landmark wajah dan status mulut
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Gambar landmark wajah
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                
                # Tentukan apakah mulut terbuka
                landmarks = face_landmarks.landmark
                if is_mouth_open(landmarks):
                    # Tentukan posisi tengah mulut (antara bibir atas dan bawah)
                    upper_lip = landmarks[13]
                    lower_lip = landmarks[14]
                    mouth_center = ((upper_lip.x + lower_lip.x) / 2, (upper_lip.y + lower_lip.y) / 2)

                    # Jika mulut terbuka dan cukup waktu telah berlalu, buat bubble baru
                    current_time = time.time()
                    if current_time - last_bubble_time > bubble_delay:
                        # Buat bubble baru di mulut
                        bubbles.append({
                            'position': [int(mouth_center[0] * frame.shape[1]), int(mouth_center[1] * frame.shape[0])],
                            'created': True
                        })
                        last_bubble_time = current_time  # Update waktu terakhir bubble dibuat

                        # Tampilkan teks status mulut terbuka
                        cv2.putText(frame, "Mouth Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # Tampilkan teks status mulut tertutup
                    cv2.putText(frame, "Mouth Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Proses setiap bubble yang ada
                for bubble in bubbles:
                    # Gerakkan bubble ke atas
                    bubble['position'][1] -= bubble_speed  # Bubble bergerak ke atas

                    # Gambar bubble (lingkaran biru)
                    cv2.circle(frame, tuple(bubble['position']), bubble_radius, (255, 0, 0), -1)

                    # Jika bubble mencapai tepi atas layar, teruskan bergerak
                    if bubble['position'][1] < bubble_radius:
                        # Bubble akan terus bergerak ke atas setelah mencapai batas atas layar
                        pass

        # Tampilkan frame
        cv2.imshow("Face Mesh - Mouth Detection with Bubbles", frame)

        # Break jika pengguna menekan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Jalankan program
face_mesh_mouth_detection_with_bubbles()
