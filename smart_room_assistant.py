import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model (for speed)
model = YOLO('yolov8n.pt')  # pre-trained on COCO (80 classes)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Starting Smart Room Assistant... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference
    results = model(frame, imgsz=320)  # optimize speed

    # Annotate detections on the frame
    annotated_frame = results[0].plot()

    # Show detections
    cv2.imshow("Smart Room Assistant", annotated_frame)

    # Print detected objects
    detected_objects = set([model.names[int(box.cls)] for box in results[0].boxes])
    if detected_objects:
        print(f"Detected: {', '.join(detected_objects)}")

    # AI trigger example
    if 'tv' in detected_objects:
        print("🎬 TV detected! Want to turn it on?")
    if 'couch' in detected_objects:
        print("🛋️ Couch detected. Time to relax?")

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()