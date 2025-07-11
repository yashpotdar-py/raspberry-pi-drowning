import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Load MobileNet SSD model (only needed once at startup)
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Start Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))  # Small res = faster
picam2.start()
time.sleep(0.1)  # Warm-up time

print("Press 'q' to quit.")
while True:
    frame = picam2.capture_array()

    # Convert to 3-channel BGR if needed
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    # Preprocess image for MobileNet SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Preprocess image for MobileNet SSD
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    #                              0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Class 15 is 'person'
                box = detections[0, 0, i, 3:7] * \
                    np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and label (optional)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"[INFO] Human detected with confidence: {confidence:.2f}")

    cv2.imshow("Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
