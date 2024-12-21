import cv2
import numpy as np
import random
import time

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
        # Activate the bubble after the delay
        if not self.active and (time.time() - self.start_time) > self.delay:
            self.active = True

        # Move the bubble upward if it's active
        if self.active:
            self.y -= random.randint(1, 5)
            # If the bubble goes above the screen, reset its position and delay
            if self.y < -self.radius:
                self.reset()

    def reset(self):
        self.y = random.randint(480, 500)  # Reset to bottom
        self.x = random.randint(0, 640)  # Randomize horizontal position
        self.radius = random.randint(5, 15)  # Randomize size
        self.delay = random.uniform(0.5, 3)  # Randomize delay
        self.start_time = time.time()  # Reset the timer
        self.active = False  # Deactivate until delay passes

def add_underwater_effect(frame, bubbles):
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

    # Draw bubbles
    for bubble in bubbles:
        bubble.move()
        if bubble.active:
            cv2.circle(frame, (bubble.x, bubble.y), bubble.radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # Apply a slight blur to make the effect more realistic
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Generate initial bubbles with random delays
bubble_count = 30
bubbles = [
    Bubble(random.randint(0, 640), random.randint(480, 500), random.randint(5, 15), random.uniform(0.5, 3))
    for _ in range(bubble_count)
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Apply the underwater effect with floating bubbles
    underwater_frame = add_underwater_effect(frame, bubbles)

    # Display the processed frame
    cv2.imshow('Underwater Effect', underwater_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
