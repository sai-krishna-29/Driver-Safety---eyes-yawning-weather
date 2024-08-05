import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("model/wheather_prediction.h5")

# Dictionary to map numeric labels to weather categories
label_mapping = {0: 'cloudy', 1: 'foggy', 2: 'rainy', 3: 'shine', 4: 'sunrise'}

# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess image for the model
    img = cv2.resize(frame, (150, 150))
    img = np.expand_dims(img, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img)
    predicted_label = label_mapping[np.argmax(prediction)]

    # Display weather prediction on the frame
    cv2.putText(frame, f"Weather: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Weather Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
