import sys
import os
import glob
import re
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import Flask , render_template , request , url_for
import pickle

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import img_to_array

import math
import matplotlib.pyplot as plt

from matplotlib.image import imread
import cv2
from PIL import Image

app = Flask('__name__')


###############################--- home page / Index page --#################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sever_home')
def sever_home():
    return render_template('index.html')

@app.route('/pages')
def pages():
    return render_template('features.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/appointment')
def appointment():
    return render_template('appointment.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/testimonial')
def testimonial():
    return render_template('testimonial.html')

@app.route('/_404')
def _404():
    return render_template('404.html')

###################################################################################

@app.route('/weather')
def weather():
    return render_template('weather.html')

model_path1 = 'weather_prediction.h5'

weather_model = load_model(model_path1)

weather_labels = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

@app.route("/weather_predict", methods=["GET", "POST"])
def weather_predict():
    if request.method == 'POST':
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Preprocess the input image
        image = Image.open(file_path)
        image = image.resize((150, 150))  # Resize to match model's expected sizing
        image = image.convert('RGB')  # Ensure the image has 3 channels (RGB)
        img = np.array(image).astype('float') / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict the weather type
        prediction = weather_model.predict(img)[0]
        weather_label = weather_labels[np.argmax(prediction)]

        # Format probabilities for rendering
        probabilities = [f"{100 * prob:.9f}%" for prob in prediction]

        # Prepare the results to be sent to the HTML template
        result = f'The predicted weather type is \'{weather_label}\''
        p1, p2, p3, p4, p5 = probabilities[:5]  # Assuming you want the top 5 probabilities
        return render_template("result1.html", result=result, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5)


###################################################################################

@app.route('/eyes')
def eyes():
    return render_template('eyes.html')

model_path2 = 'drowiness_new7.h5'

eye_model = load_model(model_path2)

drowsiness_classes = ["yawn", "no_yawn", "Closed", "Open"]

@app.route("/eyes_predict", methods=["GET", "POST"])
def eyes_predict():
    if request.method == 'POST':
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Normalize and preprocess the ROI for emotion prediction
        image = Image.open(f)
        image = image.resize((145, 145)) 
        img = np.array(image).astype('float') / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Predict the emotion from the ROI
        prediction = eye_model.predict(img)[0]
        eye_label = drowsiness_classes[prediction.argmax()]



        l3 = 100 * prediction[2]
        l4 = 100 * prediction[3]

        l3f = "{:.9f}%".format(l3)
        l4f = "{:.9f}%".format(l4)

        p3 = "\n  : " + str(l3f)
        p4 = "\n  : " + str(l4f)

        result = 'The predicted emotion is \'' + eye_label + '\''

        return render_template("result2.html", result=result, p3=p3, p4=p4)


###################################################################################

@app.route('/yawn')
def yawn():
    return render_template('yawn.html')

model_path3 = 'yawn_model_new.h5'

yawn_model = load_model(model_path3)

yawn_classes = ["No Yawn", "Yawn"]

@app.route("/yawn_predict", methods=["GET", "POST"])
def yawn_predict():
    if request.method == 'POST':
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Normalize and preprocess the ROI for emotion prediction
        image = Image.open(f)
        image = image.resize((64, 64))
        image = np.array(image).astype('float') / 255.0

        result = yawn_model.predict(np.expand_dims(image, axis=0))

        if result < 0.5:
        	status="yawn"
        else:
        	status="no yawn"

        return render_template("result3.html", result=status, p3=result)

###################################################################################

if __name__ == "__main__":
    app.run(debug = True)