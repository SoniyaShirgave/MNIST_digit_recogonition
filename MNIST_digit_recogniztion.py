import numpy as np
import cv2
from google.colab import drive, files
from google.colab.patches import cv2_imshow
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import layers, models

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
model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))

# Save the model to Google Drive
drive.mount('/content/drive')
model_path = '/content/drive/My Drive/mnist_cnn_model.h5'
model.save(model_path)

# Load the model from Google Drive
model = keras.models.load_model(model_path)

def preprocess_image(image):
    """Preprocess the image for the model."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def extract_digits(image):
    """Extract digits from the image."""
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

def predict_digits(digit_images):
    """Predict the digits in the images."""
    predictions = []
    for img in digit_images:
        img = np.reshape(img, [1, 28, 28, 1])  # Reshape for model input
        input_prediction = model.predict(img)
        input_pred_label = np.argmax(input_prediction)
        predictions.append(input_pred_label)
    return predictions

# Option to upload an image file
uploaded = files.upload()  # Uncomment this line to upload the image
input_image_path = list(uploaded.keys())[0]
input_image = cv2.imread(input_image_path)

# Check if the image was loaded correctly
if input_image is None:
    raise ValueError("Image not found. Please check the path or upload the image again.")

cv2_imshow(input_image)
digit_images = extract_digits(input_image)
predictions = predict_digits(digit_images)

print('The Handwritten Digits are recognized as:', predictions)