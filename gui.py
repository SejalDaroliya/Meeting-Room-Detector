import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf

# Load the age and gender detection models
age_model = load_model('utkface_age_detection_model.h5')
gender_model = load_model('utkface_gender_detection_model.h5')

# Create the GUI window
window = tk.Tk()
window.title("Age and Gender Detection")

# Create a label to display the selected images
image_label = tk.Label(window, text="Select images:")
image_label.pack()

# Create a list to store the selected images
selected_images = []

# Create a function to select images
def select_images():
    files = filedialog.askopenfilenames(title="Select images", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if len(files) < 2:
        tk.messagebox.showerror("Error", "Please select at least 2 images")
        return
    for file in files:
        img = Image.open(file)
        img = img.resize((48, 48))
        img = np.array(img) / 255.0
        selected_images.append(img)
    display_images()

# Create a button to select images
select_button = tk.Button(window, text="Select images", command=select_images)
select_button.pack()

# Create a function to display the selected images
def display_images():
    image_frame = tk.Frame(window)
    image_frame.pack()
    for img in selected_images:
        img_tk = ImageTk.PhotoImage(Image.fromarray((img * 255).astype(np.uint8)))
        img_label = tk.Label(image_frame, image=img_tk)
        img_label.pack(side=tk.LEFT)

# Create a function to predict ages and genders
def predict_ages_genders():
    male_count = 0
    female_count = 0
    blue_shirt_count = 0
    red_shirt_count = 0
    
    for img in selected_images:
        # Convert the image to BGR format for OpenCV
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Check if the shirt color is blue or red
        shirt_color = img_bgr.mean(axis=(0, 1)).astype(int)
        if np.allclose(shirt_color, (255, 0, 0)):  # Red shirt
            red_shirt_count += 1
        elif np.allclose(shirt_color, (0, 0, 255)):  # Blue shirt
            blue_shirt_count += 1
        
        # Convert the image back to RGB format
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
        
        # Reshape the image for the models
        for img in selected_images:
        # Check the image shape
            #if img.shape != (48, 48, 3):
            img = img.reshape((1, 48, 48, 3))  # Assuming the image is 48x48x3
            img = tf.image.resize(img, (100, 100))  # Resize the image to 100x100
            img = img.numpy().reshape((1, 100, 100, 3)) 
        
        # Predict the age and gender
        age_pred = age_model.predict(img)
        gender_pred = gender_model.predict(img)
        age = np.argmax(age_pred)
        gender = np.argmax(gender_pred)
        
        if gender == 0:  # Male
            male_count += 1
        else:  # Female
            female_count += 1
    
    display_results(male_count, female_count, blue_shirt_count, red_shirt_count)

# Create a button to predict ages and genders
predict_button = tk.Button(window, text="Predict ages and genders", command=predict_ages_genders)
predict_button.pack()

# Create a function to display the results
def display_results(male_count, female_count, blue_shirt_count, red_shirt_count):
    result_frame = tk.Frame(window)
    result_frame.pack()
    result_label = tk.Label(result_frame, text=f"No. of Males: {male_count}, No. of Females: {female_count}, Blue Shirts: {blue_shirt_count}, Red Shirts: {red_shirt_count}")
    result_label.pack()

# Start the GUI event loop
window.mainloop()