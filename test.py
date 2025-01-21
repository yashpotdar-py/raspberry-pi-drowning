import cv2
import numpy as np

class HumanDetector:
    def __init__(self, model_path, config_path, confidence_threshold=0.5):
        """
        Initialize the human detector with a pre-trained MobileNet SSD model.

        :param model_path: Path to the model weights (e.g., .caffemodel).
        :param config_path: Path to the model configuration file (e.g., .prototxt).
        :param confidence_threshold: Minimum confidence threshold for detecting humans.
        """
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)  # Load the model
        self.input_size = (300, 300)  # Input size for the model
        self.confidence_threshold = confidence_threshold  # Confidence threshold

    def detect_humans(self, frame):
        """
        Detect humans in the given frame.

        :param frame: Input image from the camera.
        :return: List of bounding boxes [(startX, startY, endX, endY)] for detected humans.
        """
        # Prepare the frame for the model
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()

        humans = []
        h, w = frame.shape[:2]  # Frame dimensions
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                humans.append((startX, startY, endX, endY))

        return humans


def main():
    # Paths to model files
    model_path = "mobilenet_iter_73000.caffemodel"
    config_path = "deploy.prototxt"

    # Initialize HumanDetector
    detector = HumanDetector(model_path, config_path)

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Use the default camera
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Starting human detection. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read the frame.")
                break

            # Detect humans
            humans = detector.detect_humans(frame)

            if humans:
                print(f"Human detected! Bounding boxes: {humans}")
            else:
                print("No humans detected.")

    except KeyboardInterrupt:
        print("\nStopping detection...")

    finally:
        # Release resources
        cap.release()
        print("Camera released.")

if __name__ == "__main__":
    main()
