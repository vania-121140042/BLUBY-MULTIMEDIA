import cv2
import mediapipe as mp
import numpy as np
from pydub import AudioSegment
import pyaudio

# 1. Initialize face detection with MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# 2. Open the webcam feed
cap = cv2.VideoCapture(0)

# 3. Function to apply an underwater effect (blur and tint)
def apply_underwater_effect(frame):
    # Apply a blue-green tint to simulate underwater
    tinted = cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
    # Apply a blur to simulate underwater distortion
    blurred = cv2.GaussianBlur(tinted, (15, 15), 0)
    return blurred

# 4. Start capturing frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Apply the underwater effect
    frame = apply_underwater_effect(frame)

    # Display the resulting frame
    cv2.imshow('Underwater Effect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
