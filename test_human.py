import cv2
import numpy as np

# Load only the MobileNet SSD model for "person" detection
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Use V4L2 (good for Raspberry Pi cameras)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set low resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare blob for the neural net
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only proceed if confidence is high
        if confidence > 0.6:
            class_id = int(detections[0, 0, i, 1])

            # Only detect 'person' (class_id == 15)
            if class_id == 15:
                print(f"Human detected with confidence {confidence:.2f}")

    # Optionally show video (comment out for performance)
    # cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
