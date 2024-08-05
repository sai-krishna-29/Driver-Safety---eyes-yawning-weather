import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from playsound import playsound
from threading import Thread

# Load the trained drowsiness detection model
eye_model = load_model("model/drowiness_new7.h5")

# Define a function to play the alarm sound
def start_alarm(sound):
    playsound('data/alarm.mp3')

# Define the eye status classes
eye_classes = ['Closed', 'Open']

# Load the Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")

# Load the trained yawning detection model
yawn_model = tf.keras.models.load_model('model/yawn_model_new.h5')

# Load the trained weather prediction model
weather_model = tf.keras.models.load_model("model/wheather_prediction.h5")
label_mapping = {0: 'cloudy', 1: 'foggy', 2: 'rainy', 3: 'shine', 4: 'sunrise'}

# Define a function to predict yawning
def predict_yawn(frame):
    # Preprocess the image
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0

    # Make a prediction
    result = yawn_model.predict(np.expand_dims(frame_normalized, axis=0))

    # Determine the yawning status
    if result[0][0] > 0.02:
        status = "Yawning"
    else:
        status = "Not Yawning"

    return status + " " + str(result[0][0])

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Initialize variables for drowsiness detection
count = 0
alarm_on = False
alarm_sound = "data/alarm.mp3"
status1 = ''
status2 = ''

while True:
    # Read a frame from the camera
    _, frame = cap.read()
    height = frame.shape[0]

    # Convert the frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect left and right eyes within the detected face region
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        
        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            
            # Extract and preprocess the left eye image
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            
            # Predict the drowsiness status for the left eye
            pred1 = eye_model.predict(eye1)
            status1 = np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            
            # Extract and preprocess the right eye image
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            
            # Predict the drowsiness status for the right eye
            pred2 = eye_model.predict(eye2)
            status2 = np.argmax(pred2)
            break

        # If both eyes are closed, start counting
        if status1 == 0 and status2 == 0:
            count += 1
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            
            # If eyes are closed for 2 consecutive frames, start the alarm
            if count >= 1:
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    
                    # Play the alarm sound in a new thread
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            count = 0
            alarm_on = False

        # Extract the face region for yawning detection
        face_roi = frame[y:y + h, x:x + w]
        
        # Predict yawning status within the face region
        yawn_status = predict_yawn(face_roi)

        # Display yawning status at the top of the screen
        cv2.putText(frame, "Yawning Status: " + yawn_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Preprocess image for weather prediction
        img = cv2.resize(frame, (150, 150))
        img = np.expand_dims(img, axis=0) / 255.0

        # Make weather prediction
        weather_prediction = weather_model.predict(img)
        predicted_label = label_mapping[np.argmax(weather_prediction)]

        # Display weather prediction on the frame
        cv2.putText(frame, f"Weather: {predicted_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with integrated detection information
    cv2.imshow("Integrated Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
