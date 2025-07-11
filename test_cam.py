import cv2
from picamera2 import Picamera2
import time

# Initialize Picamera2
picam2 = Picamera2()
picam2.start()

# Give camera time to warm up
time.sleep(0.1)

print("Press 'q' to quit.")
while True:
    img = picam2.capture_array()
    if img is None:
        print("Warning: no frame captured")
        continue

    cv2.imshow("Camera Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
