import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import layers, models
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data for CNN (add channel dimension)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
print("Model training completed.")

# Function to preprocess the image
def preprocess_image(image):
    """Preprocesses the image for the model."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to extract digits from the image
def extract_digits(image):
    """Extracts digits from the image."""
    thresh = preprocess_image(image)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 15 and w > 15:  # Filter out small contours
            digit_image = thresh[y:y+h, x:x+w]
            # Resize and normalize
            digit_image = cv2.resize(digit_image, (28, 28))
            digit_image = digit_image.astype('float32') / 255.0
            digit_image = np.expand_dims(digit_image, axis=-1)  # Add channel dimension
            digit_images.append(digit_image)

    return digit_images

# Function to predict digits in the images
def predict_digits(digit_images):
    """Predicts the digits in the images."""
    predictions = []
    for img in digit_images:
        img = np.reshape(img, [1, 28, 28, 1])  # Reshape for model input
        input_prediction = model.predict(img)
        input_pred_label = np.argmax(input_prediction)
        predictions.append(input_pred_label)
    return predictions

import os

if os.name == 'nt':  # For Windows
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)  # Fixes display scaling issues


# Create a Tkinter window
def upload_image():
    """Uploads an image using a file dialog."""
    root = tk.Tk()
    root.withdraw()  # Hide the main root window
    root.attributes('-topmost', True)  # Ensure dialog is on top
    print("Select an image file...")
    file_path = filedialog.askopenfilename(title="Select an Image File",
                                           filetypes=[("Image Files", ".jpg;.jpeg;*.png")])
    root.destroy()  # Close the root window after the dialog
    if not file_path:
        print("No file selected. Exiting...")
        exit()
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load the image. Ensure it's a valid image file.")
        exit()
    return image

# Main program flow
input_image = upload_image()

# Show the input image
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.show()

# Process and predict digits
digit_images = extract_digits(input_image)
if not digit_images:
    print("No digits found in the image.")
else:
    predictions = predict_digits(digit_images)
    print("The Handwritten Digits are recognized as:",predictions)
