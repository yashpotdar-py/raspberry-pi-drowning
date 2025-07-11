import cv2
import numpy as np
import threading
import queue
import time

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Set DNN backend to OpenCV (more stable on RPi)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Video path
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Reduce processing resolution for speed
process_width = 320
process_height = 240

# Calculate scaling factors
scale_x = frame_width / process_width
scale_y = frame_height / process_height

output_path = "output_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Thread-safe queues for frame processing
frame_queue = queue.Queue(maxsize=5)
detection_queue = queue.Queue(maxsize=5)

def detection_worker():
    """Worker thread for running detection inference"""
    while True:
        try:
            frame_data = frame_queue.get(timeout=1)
            if frame_data is None:
                break
            
            frame_count, small_frame = frame_data
            
            # Preprocess for MobileNet SSD (smaller input size)
            blob = cv2.dnn.blobFromImage(small_frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            detection_queue.put((frame_count, detections))
            frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Detection error: {e}")
            continue

# Start detection worker thread
detection_thread = threading.Thread(target=detection_worker, daemon=True)
detection_thread.start()

frame_count = 0
skip_frames = 2  # Process every 3rd frame for speed
last_detections = None

# Performance tracking
start_time = time.time()
processed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Skip frames to maintain fps
    if frame_count % skip_frames != 0:
        # Use last known detections for skipped frames
        if last_detections is not None:
            h, w = frame.shape[:2]
            frame = draw_detections(frame, last_detections, w, h, scale_x, scale_y)
        out.write(frame)
        continue
    
    # Progress indicator
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        current_fps = processed_frames / elapsed if elapsed > 0 else 0
        print(f"Frame {frame_count}, FPS: {current_fps:.1f}", end='\r')
    
    # Convert to 3-channel BGR if needed
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Resize for processing (much faster)
    small_frame = cv2.resize(frame, (process_width, process_height))
    
    # Add to detection queue (non-blocking)
    try:
        frame_queue.put_nowait((frame_count, small_frame))
    except queue.Full:
        pass  # Skip if queue is full
    
    # Check for completed detections
    try:
        detection_frame_count, detections = detection_queue.get_nowait()
        last_detections = detections
        processed_frames += 1
    except queue.Empty:
        detections = last_detections
    
    # Draw detections if available
    if detections is not None:
        h, w = frame.shape[:2]
        frame = draw_detections(frame, detections, w, h, scale_x, scale_y)
    
    # Write frame
    out.write(frame)
    
    # Optional display (comment out for max performance)
    # display_frame = cv2.resize(frame, (480, 270))
    # cv2.imshow("Video - Press 'q' to Quit", display_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

def draw_detections(frame, detections, w, h, scale_x, scale_y):
    """Draw detection boxes on frame"""
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Lower threshold for better performance
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Person class
                box = detections[0, 0, i, 3:7] * np.array([process_width, process_height, process_width, process_height])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Scale back to original frame size
                startX = int(startX * scale_x)
                startY = int(startY * scale_y)
                endX = int(endX * scale_x)
                endY = int(endY * scale_y)
                
                # Ensure coordinates are within frame bounds
                startX = max(0, min(startX, w-1))
                startY = max(0, min(startY, h-1))
                endX = max(0, min(endX, w-1))
                endY = max(0, min(endY, h-1))
                
                # Draw rectangle and label
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return frame

# Cleanup
frame_queue.put(None)  # Signal worker to stop
detection_thread.join(timeout=2)

cap.release()
out.release()
cv2.destroyAllWindows()

elapsed_total = time.time() - start_time
avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
print(f"\n[INFO] Processing complete. Average FPS: {avg_fps:.1f}")
print(f"[INFO] Output saved to '{output_path}'.")
