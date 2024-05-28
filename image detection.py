import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_WIDTH, IMG_HEIGHT = 128, 128

input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)  # Update input shape for grayscale images

fake_dir="C://Users//Bhaskar//Desktop//Fake Currency//Train//fake"

real_dir="C://Users//Bhaskar//Desktop//Fake Currency//Train//real"

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Function to preprocess images (including grayscale conversion and edge detection)

def preprocess_images(directory):
    images = []
    labels=[]
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert RGB to grayscale
        img_edges = cv2.Canny(img_gray, 100, 200)  # Apply Canny edge detection
        img_edges = cv2.resize(img_edges, (IMG_WIDTH, IMG_HEIGHT))  # Resize edges image
        img_edges = np.expand_dims(img_edges, axis=-1)  # Add channel dimension
        images.append(img_edges)
    return np.array(images)

# Load fake and real currency images

fake_images = preprocess_images(fake_dir)

real_images = preprocess_images(real_dir)

# Create labels
fake_labels = np.zeros(len(fake_images))
real_labels = np.ones(len(real_images))

# Concatenate images and labels
all_images = np.concatenate((fake_images, real_images), axis=0)
all_labels = np.concatenate((fake_labels, real_labels), axis=0)

# Shuffle the data
indices = np.arange(all_images.shape[0])
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

# Split data into training and validation sets
split_idx = int(0.8 * len(all_images))
train_images, val_images = all_images[:split_idx], all_images[split_idx:]
train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

# Build and train your CNN model

model = create_model()

model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model on the validation set

loss, accuracy = model.evaluate(val_images, val_labels)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Loading the User Image

input_image_path ="C://Users//Bhaskar//Desktop//Fake Currency//Test//real//test (15).jpg"


# Check if the image path is valid

if not os.path.exists(input_image_path):
    print("Invalid image path.")
    exit()

# Read the image

input_image = cv2.imread(input_image_path)

# Check if the image was loaded successfully

if input_image is None:
    print("Could not load the image.")
    exit()

# Preprocessing the user Image


# Convert grayscale image to RGB

if len(input_image.shape) == 2:

  input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

# Convert RGB to grayscale

input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection

input_image_edges = cv2.Canny(input_image_gray, 100, 200)

# Resize edges image

input_image_edges = cv2.resize(input_image_edges, (IMG_WIDTH, IMG_HEIGHT))

# Add batch dimension

input_image_edges = np.expand_dims(input_image_edges, axis=0)

# Predict using the model

prediction = model.predict(input_image_edges)

# Interpret the prediction

if prediction[0][0] > 0.5:
    print("The currency note is predicted to be real.")
else:
    print("The currency note is predicted to be fake.")





import pickle

with open('model.pkl', 'wb') as f:  # Replace 'model.pkl' with your desired filename
    pickle.dump(model, f)  # 'model' is your trained model object

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Interpret the prediction

if prediction[0][0] > 0.6:

    print("The currency note is predicted to be real.")

else:
    print("The currency note is predicted to be fake.")