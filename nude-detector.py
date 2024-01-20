from ultralytics import YOLO  # Importing YOLO model from the ultralytics library
import cv2  # Importing OpenCV, a computer vision library
import cvzone  # Importing cvzone, a library built on top of OpenCV for computer vision tasks
import math  # Importing the math module for mathematical operations

# Accessing the video feed from a file 
cap = cv2.VideoCapture("pure4.mp4")

# Initializing the YOLO model with the pre-trained weights ('best.pt')
model = YOLO('best.pt')

# Defining a list of class names related to detections in the YOLO model
classnames = ['male-external-genital', 'female-vulva', 'female-breast', 'mouth and male-external-genital']
#classnames = ['vagina', 'breast', 'penis']

# Running a continuous loop for processing each frame of the video
while True:
    ret, frame = cap.read()  # Reading a frame from the video feed

    # Handling cases where the frame is empty or the video has ended
    if not ret or frame is None or frame.size == 0:
        print("Frame is empty or video ended")
        break  # Exiting the loop if the video has ended or the frame is empty

    frame = cv2.resize(frame, (640, 640))  # Resizing the frame to a square shape (640x640 pixels)

    result = model(frame, stream=True)  # Getting predictions from the YOLO model for the current frame

    # Extracting information from the YOLO model's output (bounding boxes, confidences, and class names)
    for info in result:
        boxes = info.boxes  # Extracting bounding box information
        for box in boxes:
            confidence = box.conf[0]  # Extracting the confidence of the detected object
            confidence = math.ceil(confidence * 100)  # Converting the confidence to a percentage (ceiling)
            Class = int(box.cls[0])  # Extracting the class label of the detected object

            # Displaying the bounding box and the label if the confidence is greater than 50%
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]  # Extracting coordinates of the bounding box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Converting coordinates to integers
                
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Drawing a red rectangle around the object
            #     cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
            #                        scale=1.8, thickness=1)  # Adding text label to the bounding box
            
                # Creating a region of interest (ROI) within the bounding box
                roi = frame[y1:y2, x1:x2]

                # Applying a blur effect to the ROI
                blurred_roi = cv2.GaussianBlur(roi, (71, 71), 0)

                # Placing the blurred ROI back into the frame
                frame[y1:y2, x1:x2] = blurred_roi
                
                # Creating a region of interest (ROI) within the bounding box
                # roi = frame[y1:y2, x1:x2]

                # # Applying a blur effect to the ROI
                # blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)

                # # Placing the blurred ROI back into the frame
                # frame[y1:y2, x1:x2] = blurred_roi

                # # Drawing a red rectangle around the object
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

                # # Adding text label to the bounding box
                # cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                #                 scale=1.8, thickness=1)
                

    cv2.imshow('frame', frame)  # Displaying the frame with detected objects

    # Checking for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exiting the loop if 'q' key is pressed

# Releasing the video capture resources and closing OpenCV windows
cap.release()
cv2.destroyAllWindows()
