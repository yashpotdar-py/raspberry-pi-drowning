import cv2
import numpy as np
import time
import torch

def draw_detections(frame, detections, scale_x, scale_y):
    """Draw detection boxes on frame"""
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.4 and int(cls) == 0:  # Person class
            # Scale to original frame
            startX = int(x1 * scale_x)
            startY = int(y1 * scale_y)
            endX = int(x2 * scale_x)
            endY = int(y2 * scale_y)
            
            # Clamp coordinates
            startX = max(0, min(startX, w-1))
            startY = max(0, min(startY, h-1))
            endX = max(0, min(endX, w-1))
            endY = max(0, min(endY, h-1))
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
    return frame

# Load local .pt file
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.4
model.iou = 0.5
model.cpu()
model.eval()

# Disable gradient computation for speed
torch.set_grad_enabled(False)

# Video setup
cap = cv2.VideoCapture("test.mp4")
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Ultra-lightweight processing
process_width, process_height = 160, 120
scale_x = frame_width / process_width
scale_y = frame_height / process_height

# Output setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_detected.mp4", fourcc, fps, (frame_width, frame_height))

frame_count = 0
skip_frames = 6  # Process every 7th frame
last_detections = []
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process only every 7th frame
    if frame_count % skip_frames == 0:
        # Tiny resolution for speed
        small_frame = cv2.resize(frame, (process_width, process_height))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Run inference at tiny size
        results = model(rgb_frame, size=160)
        detections = results.pandas().xyxy[0].values
        last_detections = detections
        
        if frame_count % 210 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed
            print(f"Frame {frame_count}, FPS: {fps_current:.1f}")
    
    # Draw detections if available
    if len(last_detections) > 0:
        frame = draw_detections(frame, last_detections, scale_x, scale_y)
    
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

elapsed_total = time.time() - start_time
avg_fps = frame_count / elapsed_total
print(f"Complete. Average FPS: {avg_fps:.1f}")
