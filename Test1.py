import cv2
import numpy as np
import random

def add_underwater_effect(frame):
    # Apply bluish-green color overlay
    overlay = frame.copy()
    overlay[:, :, 0] = cv2.addWeighted(overlay[:, :, 0], 1, np.full_like(overlay[:, :, 0], 50), 0.5, 0)  # Blue
    overlay[:, :, 1] = cv2.addWeighted(overlay[:, :, 1], 1, np.full_like(overlay[:, :, 1], 30), 0.5, 0)  # Green
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Add distortion effect (wave-like pattern)
    rows, cols, _ = frame.shape
    distortion_map_x = np.tile(np.linspace(0, cols - 1, cols), (rows, 1)).astype(np.float32)
    distortion_map_y = np.tile(np.linspace(0, rows - 1, rows), (cols, 1)).T.astype(np.float32)
    sinusoid = 10 * np.sin(2 * np.pi * distortion_map_y / 180).astype(np.float32)
    distortion_map_x += sinusoid

    frame = cv2.remap(frame, distortion_map_x, distortion_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Add random bubble-like particles
    for _ in range(30):
        radius = random.randint(5, 15)
        x = random.randint(0, cols)
        y = random.randint(0, rows)
        color = (255, 255, 255)  # White
        thickness = -1  # Filled circle
        cv2.circle(frame, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)

    # Apply a slight blur to make the effect more realistic
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    return frame

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Apply the underwater effect
    underwater_frame = add_underwater_effect(frame)

    # Display the processed frame
    cv2.imshow('Underwater Effect', underwater_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
