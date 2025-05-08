from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Load model
detected_objects = set()    # Store latest detections

# Webcam capture
cap = cv2.VideoCapture(0)

def generate_frames():
    global detected_objects
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame, imgsz=320)
        annotated_frame = results[0].plot()

        # Update detected objects
        detected_objects = set([model.names[int(box.cls)] for box in results[0].boxes])

        # Convert frame to bytes
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    return jsonify(list(detected_objects))

if __name__ == '__main__':
    app.run(debug=True)