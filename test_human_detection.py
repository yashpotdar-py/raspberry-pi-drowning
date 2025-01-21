import cv2
from human_detection import HumanDetector

# Paths to model files
model_path = "mobilenetv2.caffemodel"
config_path = "deploy.prototxt"

# Initialize HumanDetector
detector = HumanDetector(model_path, config_path)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to quit the program.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        break

    # Detect humans
    humans = detector.detect_humans(frame)

    for (startX, startY, endX, endY) in humans:
        # Draw bounding box around detected human
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Human Detection", frame)

    # Exit loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
