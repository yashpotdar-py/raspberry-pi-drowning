import cv2
import numpy as np
from picamera2 import Picamera2, Preview

class HumanDetector:
    def __init__(self, model_path, config_path, confidence_threshold=0.5):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.confidence_threshold = confidence_threshold
        self.input_size = (300, 300)

    def detect_humans(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.input_size,
                                     (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()

        humans = []
        h, w = frame.shape[:2]
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = box.astype(int)
                cx, cy = (sx + ex) // 2, (sy + ey) // 2
                pos = self.get_position(cx, cy, w, h)
                humans.append((sx, sy, ex, ey, pos))
        return humans

    def get_position(self, cx, cy, fw, fh):
        w3, h3 = fw // 3, fh // 3
        vert = 'top' if cy < h3 else 'bottom' if cy > 2*h3 else 'middle'
        horz = 'left' if cx < w3 else 'right' if cx > 2*w3 else 'center'
        return f"{vert}-{horz}"

def main():
    model_path = "mobilenetv2.caffemodel"
    config_path = "deploy.prototxt"
    detector = HumanDetector(model_path, config_path)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()  # warm-up happens internally by Picamera2 :contentReference[oaicite:1]{index=1}

    print("Starting... press Ctrl+C to exit")
    try:
        while True:
            frame = picam2.capture_array()
            humans = detector.detect_humans(frame)
            for sx, sy, ex, ey, pos in humans:
                cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
                cv2.putText(frame, pos, (sx, sy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Humans", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        picam2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
