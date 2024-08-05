import cv2
import numpy as np
import tensorflow as tf

# Load the trained yawning detection model
model = tf.keras.models.load_model('model/yawn_model_new.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Create a window to display the video feed
cv2.namedWindow('Yawning Detection')

# Define a function to predict yawning
def predict_yawn(frame):
    # Preprocess the image
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0

    # Make a prediction
    result = model.predict(np.expand_dims(frame_normalized, axis=0))

    # Determine the yawning status
    if result[0][0] > 0.02:
        status = "Yawning"
    else:
        status = "Not Yawning"

    return status + " " + str(result[0][0])

# Open the camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was read successfully
    if ret:
        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the face region for yawning detection
            face_roi = frame[y:y + h, x:x + w]

            # Predict yawning status within the face region
            status = predict_yawn(face_roi)

        # Display the yawning status at the top of the screen
        cv2.putText(frame, "Yawning Status: " + status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Yawning Detection', frame)

        # Wait for 1 millisecond or until the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
