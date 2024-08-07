import os
import cv2
import numpy as np
import pandas as pd
import np_utils
from sklearn.model_selection import train_test_split
import keras_utils

# Directory where the UTKFace dataset is stored
dataset_dir = 'UTKFace'

# Initialize lists to hold the data
images = []
ages = []
genders = []
shirt_colors = []

# Define the shirt colors (for simplicity, we assume each image has a shirt color associated)
shirt_color_options = ['White', 'Black', 'Red', 'Blue']

# Iterate over the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        # Extract the age, gender, and ethnicity from the filename
        age, gender, _ = filename.split('_')[:3]
        
        # Load the image
        img = cv2.imread(os.path.join(dataset_dir, filename))
        img = cv2.resize(img, (64, 64))  # Resize to a fixed size
        
        # Append the data to the lists
        images.append(img)
        ages.append(int(age))
        genders.append(int(gender))
        shirt_colors.append(np.random.choice(shirt_color_options))

# Convert lists to arrays
images = np.array(images)
ages = np.array(ages)
genders = np.array(genders)
shirt_colors = np.array(shirt_colors)

# One-hot encode the genders and shirt colors
genders = np_utils.to_categorical(genders, 2)
shirt_colors_encoded = pd.get_dummies(shirt_colors).values

# Split the dataset into training and testing sets
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test, y_shirt_train, y_shirt_test = train_test_split(
    images, ages, genders, shirt_colors_encoded, test_size=0.2, random_state=42)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Create the age model
age_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')
])

age_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
age_model.summary()

# Create the gender model
gender_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

gender_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
gender_model.summary()

# Train the models
age_model.fit(X_train, y_age_train, validation_split=0.2, epochs=10, batch_size=32)
gender_model.fit(X_train, y_gender_train, validation_split=0.2, epochs=10, batch_size=32)
def adjust_age_based_on_shirt_color(age, shirt_color):
    if shirt_color == 'White':
        return 23
    elif shirt_color == 'Black':
        return 10
    else:
        return age
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def load_and_preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    return img

def predict_age_and_gender(img):
    age = age_model.predict(img)[0][0]
    gender = np.argmax(gender_model.predict(img), axis=1)[0]
    return age, gender

def display_results():
    if len(genders) < 2:
        messagebox.showerror("Error", "Not enough people in the meeting room.")
        return

    # Adjust age based on shirt color
    adjusted_ages = [adjust_age_based_on_shirt_color(ages[i], shirt_colors[i]) for i in range(len(ages))]

    # Count the number of females and males
    num_females = np.sum(genders == 1)
    num_males = np.sum(genders == 0)

    # Display the results
    result_text = (f"Number of females in the meeting room: {num_females}\n"
                   f"Number of males in the meeting room: {num_males}\n"
                   f"Adjusted ages based on shirt color:\n{adjusted_ages}")
    result_label.config(text=result_text)

def load_image():
    filepath = filedialog.askopenfilename()
    img = load_and_preprocess_image(filepath)
    age, gender = predict_age_and_gender(img)
    shirt_color = np.random.choice(['White', 'Black', 'Red', 'Blue'])
    ages.append(age)
    genders.append(gender)
    shirt_colors.append(shirt_color)
    display_results()

# Create the main window
root = tk.Tk()
root.title("Meeting Room Analysis")

# Create and place widgets
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=20)

display_button = tk.Button(root, text="Display Results", command=display_results)
display_button.pack(pady=20)

result_label = tk.Label(root, text="", justify="left")
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
