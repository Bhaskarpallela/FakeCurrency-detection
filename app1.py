from flask import Flask, render_template, request, redirect, flash
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app= Flask(__name__)

# Load the trained models
counterfeit_model = load_model('model3.h5')
classification_model = load_model('model2.h5')

# Manually compile the loaded models
counterfeit_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a function to preprocess the input image for counterfeit detection
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)
    img_resized = cv2.resize(img_edges, (128, 128))
    img_processed = np.expand_dims(img_resized, axis=0)
    return img_processed

# Define a function to preprocess the image for classification
def preprocess_image_classification(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (150, 150))
    img_processed = np.expand_dims(img_resized, axis=0)
    return img_processed / 255.0  # Normalize pixel values
'''
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Perform authentication (replace this with your actual authentication logic)
        if username == 'admin@gmail.com' and password == 'password':
            return redirect('/home')
        else:
            flash('Invalid username or password')
    return render_template('index.html')'''

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the image file from the request
        imagefile = request.files['imagefile']
        # Save the image to a temporary directory
        image_path = "./images/" + secure_filename(imagefile.filename)
        imagefile.save(image_path)
        # Preprocess the image for counterfeit detection
        input_image = preprocess_image(image_path)
        # Make predictions for counterfeit detection
        counterfeit_prediction = counterfeit_model.predict(input_image)
        counterfeit_result = "real" if counterfeit_prediction[0][0] > 0.5 else "fake"
        # Preprocess the image for classification
        input_image_classification = preprocess_image_classification(image_path)
        # Make predictions for image classification
        classification_prediction = classification_model.predict(input_image_classification)
        classification_result = np.argmax(classification_prediction)  # Get the predicted class
        denominations = {0: '10', 1: '100', 2: '20', 3: '200', 4: '2000', 5: '50', 6: '500'}
        classification_result=denominations[classification_result]
        print("Counterfeit Result:", counterfeit_result)
        print("Classification Result:", classification_result)
        return render_template('home.html', counterfeit_result=counterfeit_result, classification_result=classification_result)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=False)
