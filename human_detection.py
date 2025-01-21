import cv2
import numpy as np

class HumanDetector:
    def __init__(self, model_path, config_path):
        """
        Initialize the human detector with a pre-trained MobileNet SSD model.

        :param model_path: Path to the model weights (e.g., .caffemodel).
        :param config_path: Path to the model configuration file (e.g., .prototxt).
        """
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)  # Load model
        self.input_size = (300, 300)  # Input size for the model
        self.confidence_threshold = 0.5  # Minimum confidence for detection

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
