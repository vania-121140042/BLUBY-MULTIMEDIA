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

def overlay_png(frame, bubble_png, x, y, size):
    # Resize the PNG bubble to match the given size
    bubble_resized = cv2.resize(bubble_png, (size, size), interpolation=cv2.INTER_AREA)

    # Extract the alpha channel and RGB channels
    b_h, b_w, _ = bubble_resized.shape
    alpha = bubble_resized[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
    bubble_rgb = bubble_resized[:, :, :3]

    # Ensure the bubble is within the frame bounds
    y1, y2 = max(0, y - b_h // 2), min(frame.shape[0], y + b_h // 2)
    x1, x2 = max(0, x - b_w // 2), min(frame.shape[1], x + b_w // 2)

    # Adjust bubble and alpha dimensions if cropped by the frame
    b_y1 = max(0, -y + b_h // 2)
    b_y2 = b_y1 + (y2 - y1)
    b_x1 = max(0, -x + b_w // 2)
    b_x2 = b_x1 + (x2 - x1)

    # Check for valid crop dimensions
    if y1 < y2 and x1 < x2 and b_y1 < b_y2 and b_x1 < b_x2:
        # Blend the bubble with the frame using the alpha channel
        for c in range(3):  # RGB channels
            frame[y1:y2, x1:x2, c] = (
                frame[y1:y2, x1:x2, c] * (1 - alpha[b_y1:b_y2, b_x1:b_x2]) +
                bubble_rgb[b_y1:b_y2, b_x1:b_x2, c] * alpha[b_y1:b_y2, b_x1:b_x2]
            )

def add_underwater_effect(frame, bubbles, bubble_png, time_factor):
    # Apply bluish-green color overlay
    overlay = frame.copy()
    overlay[:, :, 0] = cv2.addWeighted(overlay[:, :, 0], 1, np.full_like(overlay[:, :, 0], 50), 0.5, 0)  # Blue
    overlay[:, :, 1] = cv2.addWeighted(overlay[:, :, 1], 1, np.full_like(overlay[:, :, 1], 30), 0.5, 0)  # Green
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Add dynamic distortion effect (wave-like pattern)
    rows, cols, _ = frame.shape
    distortion_map_x = np.tile(np.linspace(0, cols - 1, cols), (rows, 1)).astype(np.float32)
    distortion_map_y = np.tile(np.linspace(0, rows - 1, rows), (cols, 1)).T.astype(np.float32)
    
    # Time-based randomness added to the sinusoidal distortion to simulate underwater movement
    random_shift = random.uniform(-5, 5)  # Random variation in the distortion
    sinusoid = 10 * np.sin(2 * np.pi * (distortion_map_y / 180 + time_factor) + random_shift).astype(np.float32)
    distortion_map_x += sinusoid

    # Apply the distortion map to the frame
    # frame = cv2.remap(frame, distortion_map_x, distortion_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Draw bubbles
    for bubble in bubbles:
        bubble.move()
        if bubble.active:
            overlay_png(frame, bubble_png, bubble.x, bubble.y, bubble.radius * 2)

    # Apply a slight blur to make the effect more realistic
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame

# Load the bubble PNG with alpha channel
bubble_png = cv2.imread('bubble.png', cv2.IMREAD_UNCHANGED)  # Ensure the image has an alpha channel

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Generate initial bubbles with random delays
bubble_count = 30
bubbles = [
    Bubble(random.randint(0, 640), random.randint(480, 500), random.randint(5, 15), random.uniform(0.5, 3))
    for _ in range(bubble_count)
]

# Initialize time factor for underwater movement
time_factor = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Apply the underwater effect with floating bubbles, passing the time factor
    underwater_frame = add_underwater_effect(frame, bubbles, bubble_png, time_factor)

    # Display the processed frame
    cv2.imshow('Underwater Effect', underwater_frame)

    # Increment the time factor to animate the distortion slowly
    time_factor += 0.1  # Slow down the rate of change for the distortion movement

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
