# Import necessary modules and classes from Flask and other libraries
from flask import Flask, render_template, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import cvzone
import math

# Create a Flask web application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for all routes in the Flask app
CORS(app)

# Accessing the video feed from a file
cap = cv2.VideoCapture("pure4.mp4")

# Initialize the YOLO model with pre-trained weights
model = YOLO('best.pt')

# Define a list of class names related to detections in the YOLO model
classnames = ['male-external-genital', 'female-vulva', 'female-breast', 'mouth and male-external-genital']

# Define a function to generate video frames for streaming
def generate_frames():
    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()

        # Check if the frame is empty or the video has ended
        if not ret or frame is None or frame.size == 0:
            print("Frame is empty or video ended")
            break

        # Resize the frame to a square shape (640x640 pixels)
        frame = cv2.resize(frame, (640, 640))

        # Get predictions from the YOLO model for the current frame
        result = model(frame, stream=True)

        # Process the YOLO model's output and modify the frame accordingly
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])

                # If the confidence is greater than 50%, apply a blur effect to the detected object
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Create a region of interest (ROI) within the bounding box
                    roi = frame[y1:y2, x1:x2]

                    # Apply a blur effect to the ROI using GaussianBlur
                    blurred_roi = cv2.GaussianBlur(roi, (71, 71), 0)

                    # Place the blurred ROI back into the frame
                    frame[y1:y2, x1:x2] = blurred_roi

        # Encode the modified frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame bytes with boundary for multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# Define the route for the home page, which renders an HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for the video feed, which returns a Response object for streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
