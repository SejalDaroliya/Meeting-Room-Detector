import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import cv2
import os

# Load the UTKFace dataset
def load_utkface_dataset(path):
    images = []
    ages = []
    genders = []
    for filename in os.listdir(path):
        print(f"Processing file: {filename}")
        img = cv2.imread(os.path.join(path, filename))
        if img is None:
            print(f"Error: Could not read image {filename}")  # Check if image was read successfully
            continue
        img = cv2.resize(img, (100,100))
        img = img / 255.0
        images.append(img)
        age = int(filename.split('_')[0])
        ages.append(age)
        gender = int(filename.split('_')[1])
        genders.append(gender)
    images = np.array(images)
    ages = np.array(ages)
    genders = np.array(genders)
    return images, ages, genders

# Load the dataset
print("Loading the dataset.....")
images, ages, genders = load_utkface_dataset('UTKFace')
print("Loaded the dataset...")

# Split the dataset into training and testing sets
print("Splitting the dataset....")
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(images, ages, genders, test_size=0.2, random_state=42)
print("Dataset splitted successfully....")

# Define the age detection model
age_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

age_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the age detection model
age_model.fit(X_train, y_age_train, epochs=10, batch_size=32, validation_data=(X_test, y_age_test))

# Define the gender detection model
gender_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

gender_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the gender detection model
gender_model.fit(X_train, tf.keras.utils.to_categorical(y_gender_train), epochs=10, batch_size=32, validation_data=(X_test, tf.keras.utils.to_categorical(y_gender_test)))

# Save the models
age_model.save('utkface_age_detection_model.h5')
gender_model.save('utkface_gender_detection_model.h5')