#Image Classification 

import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths to your dataset

train_data_dir = "C://Users//Bhaskar//Desktop//FakeCurrDataset//Train//Real"

test_data_dir = "C://Users//Bhaskar//Desktop//FakeCurrDataset//Test//Real"

input_shape = (150, 150, 3)  # Input shape for your images


# Define data generators for training and testing

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

# Define your CNN mode
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(7, activation='softmax')  # Change 7 to the number of classes you have
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train your model
history = model.fit(
      train_generator,
      steps_per_epoch=8,  # You may need to adjust this value based on your dataset size
      epochs=15,
      validation_data=validation_generator,
      validation_steps=8,  # You may need to adjust this value based on your dataset size
      verbose=2)
'''
# Plot training and validation accuracy and loss

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']


loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model



from glob import glob

glob("C://Users//Bhaskar//Desktop//FakeCurrDataset//Train//Real//*//")

# Define denominations
denominations = {0: '10', 1: '100', 2: '20', 3: '200', 4: '2000', 5: '50', 6: '500'}

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values


# Ask user for image path

image_path ="C://Users//Bhaskar//Desktop//FakeCurrDataset//Test//Real//200//2.jpg"
# Preprocess the image

image = preprocess_image(image_path)

# Make predictions   

predictions = model.predict(image)

# Get the predicted class and its probability

print(predictions)

predicted_class = np.argmax(predictions)

probability = predictions[0][predicted_class]

# Print the result
print("Predicted denomination:", denominations[predicted_class])
print("Probability:", probability)



# Save the model
model.save('model2.h5')

# Load the model
loaded_model = load_model('model2.h5')



