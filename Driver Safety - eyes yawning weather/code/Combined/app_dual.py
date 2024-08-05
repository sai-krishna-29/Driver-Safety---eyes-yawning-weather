import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread

# Load the trained drowsiness detection model
drowsiness_model = load_model("model/drowiness_new7.h5")

# Load the trained yawning detection model
yawn_model = tf.keras.models.load_model('model/yawn_model_new.h5')

# Load the Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")

# Define a function to play the alarm sound
def start_alarm(sound):
    playsound('data/alarm.mp3')

# Define the eye status classes
eye_classes = ['Closed', 'Open']

# Create a window to display the video feed
cv2.namedWindow('Drowsiness and Yawning Detection')

# Define a function to predict yawning
def predict_yawn(frame):
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0
    result = yawn_model.predict(np.expand_dims(frame_normalized, axis=0))
    if result[0][0] > 0.02:
        status = "Yawning"
    else:
        status = "Not Yawning"
    return status + " " + str(result[0][0])

# Open the camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Initialize variables for drowsiness detection
count = 0
alarm_on = False
alarm_sound = "data/alarm.mp3"
status1 = ''
status2 = ''

while True:
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)

            for (x1, y1, w1, h1) in left_eye:
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                eye1 = roi_color[y1:y1+h1, x1:x1+w1]
                eye1 = cv2.resize(eye1, (145, 145))
                eye1 = eye1.astype('float') / 255.0
                eye1 = img_to_array(eye1)
                eye1 = np.expand_dims(eye1, axis=0)
                pred1 = drowsiness_model.predict(eye1)
                status1 = np.argmax(pred1)
                break

            for (x2, y2, w2, h2) in right_eye:
                cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
                eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                eye2 = cv2.resize(eye2, (145, 145))
                eye2 = eye2.astype('float') / 255.0
                eye2 = img_to_array(eye2)
                eye2 = np.expand_dims(eye2, axis=0)
                pred2 = drowsiness_model.predict(eye2)
                status2 = np.argmax(pred2)
                break

            if status1 == 2 and status2 == 2:
                count += 1
                cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                if count >= 1:
                    cv2.putText(frame, "Drowsiness Alert!!!", (100, frame.shape[0]-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    if not alarm_on:
                        alarm_on = True
                        t = Thread(target=start_alarm, args=(alarm_sound,))
                        t.daemon = True
                        t.start()
            else:
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                count = 0
                alarm_on = False

            # Predict yawning status within the face region
            status_yawn = predict_yawn(roi_color)

        cv2.putText(frame, "Yawning Status: " + status_yawn, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Drowsiness and Yawning Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
