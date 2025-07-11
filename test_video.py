import cv2
import numpy as np

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Provide the path to your MP4 video
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1

    # Lightweight progress indicator every 10 frames
    if frame_count % 10 == 0:
        print(f"Processing frame {frame_count}...", end='\r')

    # Convert to 3-channel BGR if needed
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Preprocess for MobileNet SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Check each detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Class 15 is 'person'
                print(f"[Frame {frame_count}] Person detected with confidence: {confidence:.2f}")

cap.release()
print("\n[INFO] Video processing complete.")
