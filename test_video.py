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

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = "output_detected.mp4"

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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

    h, w = frame.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Class 15 is 'person'
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw rectangle and label
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                print(f"[Frame {frame_count}] Person detected with confidence: {confidence:.2f}")

    # Write the annotated frame to output video
    out.write(frame)

    # Optionally show resized frame to reduce display load
    display_frame = cv2.resize(frame, (640, 360))
    cv2.imshow("Video - Press 'q' to Quit", display_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Video processing complete. Results saved to '{output_path}'.")
