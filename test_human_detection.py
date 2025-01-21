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
        :return: List of tuples [(startX, startY, endX, endY, position)] for detected humans.
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

                # Calculate the center of the bounding box
                centerX = (startX + endX) // 2
                centerY = (startY + endY) // 2

                # Determine the position on the screen
                position = self.get_position(centerX, centerY, w, h)

                # Append the bounding box and position
                humans.append((startX, startY, endX, endY, position))

        return humans

    def get_position(self, centerX, centerY, frame_width, frame_height):
        """
        Determine the relative position of the detected human on the screen.

        :param centerX: X-coordinate of the bounding box center.
        :param centerY: Y-coordinate of the bounding box center.
        :param frame_width: Width of the frame.
        :param frame_height: Height of the frame.
        :return: A string describing the position (e.g., 'top-left', 'center', etc.).
        """
        # Define thresholds for dividing the frame into regions
        width_third = frame_width // 3
        height_third = frame_height // 3

        if centerY < height_third:
            vertical = "top"
        elif centerY > 2 * height_third:
            vertical = "bottom"
        else:
            vertical = "middle"

        if centerX < width_third:
            horizontal = "left"
        elif centerX > 2 * width_third:
            horizontal = "right"
        else:
            horizontal = "center"

        return f"{vertical}-{horizontal}"


def main():
    # Paths to model files
    model_path = "mobilenetv2.caffemodel"
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

            # Detect humans and their positions
            humans = detector.detect_humans(frame)

            if humans:
                for (startX, startY, endX, endY, position) in humans:
                    print(f"Human detected at position: {position}, Bounding box: ({startX}, {startY}, {endX}, {endY})")
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
