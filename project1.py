import os
import cv2
import numpy as np
import tkinter as tk
import tensorflow
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras import layers, models

# Step 1 : Dataset Creation
def create_dataset():
    os.makedirs("Meeting_room_dataset", exist_ok=True)
    num_samples = 1000
    for i  in range(num_samples):
        num_people = np.random.randint(2,6)
        genders = np.random.randint(0, 2, size=num_people)
        ages = np.random.randint(18, 60, size=num_people)
        image = np.zeros((200, 300, 3), dtype="uint8")
        filename = f"meeting_room_{1}.jpg"
        cv2.imwrite(os.path.join("Meeting_room_dataset", filename), image)
        with open(os.path.join("Meeting_room_dataset", f"{filename}.txt"), "w") as f:
            f.write(f"Number of people: {num_people}\n")
            for j in range(num_people):
                f.write(f"Person {j+1}: Gender: {"Male" if genders[j] == 1 else "Female"}, Age = {ages[j]}\n")

# Step 2 : Data Preprocessing
def extract_shirt_color(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixels = cv2.countNonZero(binary)
    total_pixels = image.shape[0]*image.shape[1]
    if white_pixels/total_pixels > 0.5:
        return "white"
    else:
        return "black"
    
def preprocess_dataset():
    dataset_dir = "Meeting_room_dataset"
    images = []
    genders = []
    ages = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            images.append(image)
            annotation_path = os.path.join(dataset_dir, f"{filename}.txt")
            with open(annotation_path,"r") as f:
                lines = f.readlines()
                try:
                    num_people = int(lines[0].split(': ')[1])
                except IndexError:
                    print(f"Error reading number of people in {annotation_path}")
                    continue
                for line in lines[1:num_people+1]:
                    try:
                        gender = 1 if "Male" in line else 0
                        age = int(line.split("Age: ")[1])
                        genders.append(gender)
                        ages.append(age)
                    except IndexError:
                        print(f"Error parsing line in {annotation_path}: {line}")
                        continue
    shirt_colors = [extract_shirt_color(image) for image in images]
    
    return np.array(images), np.array(genders), np.array(ages),np.array(shirt_colors)


# Step 3 : Training the gender prediction model
def train_gender_prediction_model(images, genders,shirt_colors):
    batch_size = 32
    X_train = np.concatenate((genders.reshape(-1,1), shirt_colors.reshape(-1,1)), axis = 1)
    Y_train = genders
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    model = models.Sequential([layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (200, 300, 3)),
                               layers.MaxPooling2D((2, 2)),
                               layers.Conv2D(64, (3, 3), activation = "relu"),
                               layers.MaxPooling2D((2, 2)),
                               layers.Conv2D(128, (3, 3), activation = "relu"),
                               layers.Flatten(),
                               layers.Dense(128, activation = "relu"),
                               layers.Dense(1, activation = "sigmoid")])
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    model.fit(X_train, Y_train, epochs = 10, batch_size = batch_size)
    return model

# Step 4 : Training the age prediction model
def train_age_prediction_model(images, ages, genders, shirt_colors):
    batch_size = 32
    ages_processed = np.where(shirt_colors == "white", 23, np.where(shirt_colors == "black", "child", ages))
    X_train = np.concatenate((genders.rehsape(-1,1), ages_processed.reshape(-1,1)), axis = 1)
    Y_train = ages
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    model = models.Sequential([layers.Dense(64, activation = "relu", input_shape = (2,)),
                               layers.Dense(32, activation = "relu"),
                               layers.Dense(1)])
    model.compile(optimizer = "adam", loss = "mse", metrics = ["mae"])
    model.fit( X_train, Y_train, epochs = 10, batch_size = batch_size)
    return model

# Step 5 : Integrating with GUI
def preprocess_image(image):
    resiezed_image = cv2.resize(image, (300, 200))
    normalized_image = resiezed_image/255.0
    return np.expand_dims(normalized_image, axis =0)

def extract_features(image):
    shirt_color = extract_shirt_color(image)
    return shirt_color

def predict_gender_and__age(image, gender_model, age_model):
    preprocessed_image = preprocess_image(image)
    gender_prediction = gender_model.predict(preprocessed_image)
    shirt_color = extract_features(image)
    age_prediction = 30 if gender_prediction > 0.5 and shirt_color == "white" else "child"
       
    return gender_prediction, age_prediction

def display_results(gender_prediction, age_prediction):
    gender = "Male" if gender_prediction > 0.5 else "Female"
    result_str = f"Predicted Gender: {gender}\n"
    result_str += f"Predicted Age: {age_prediction}\n"
    global result_label
    result_label.config(text = result_str)

def process_image(filepath):
    image = cv2.imread(filepath)
    gender_prediction, age_prediction = predict_gender_and__age(image, gender_model, age_model)

    display_results(gender_prediction, age_prediction)
    

def open_file_dialog():
    filepath = filedialog.askopenfilename()
    if filepath:
        process_image(filepath)

#Create the dataset
create_dataset()

#Preprocess the dataset
images, genders, ages, shirt_colors = preprocess_dataset()
print("genders shape: ", genders.shape)
print("shirtcolor shape: ", shirt_colors.shape)


#Train the gender prediction model
gender_model = train_gender_prediction_model(images, genders,shirt_colors)

#Train the age prediction model
age_model = train_age_prediction_model(images, ages, genders, shirt_colors)

#GUI
root = tk.Tk()
root.title("Meeting Room Analyzer")

load_button = tk.Button(root, text = "Load Image", command = open_file_dialog())
load_button.pack(pady = 10)

result_label = tk.Label(root, text = "")
result_label.pack(pady = 10)

root.mainloop()