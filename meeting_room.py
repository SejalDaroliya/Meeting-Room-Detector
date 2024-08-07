import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import tkinter as tk
from tkinter import messagebox

# Generate synthetic dataset (Run this block only once to create the dataset)
def create_synthetic_dataset():
    np.random.seed(42)
    num_samples = 100
    ages = np.random.randint(18, 60, size=num_samples)
    genders = np.random.choice(['Male', 'Female'], size=num_samples)
    shirt_colors = np.random.choice(['White', 'Black', 'Red', 'Blue'], size=num_samples)

    data = {'Age': ages, 'Gender': genders, 'Shirt Color': shirt_colors}
    df = pd.DataFrame(data)
    df.to_csv('synthetic_meeting_room_dataset.csv', index=False)
    print("Synthetic dataset created and saved as 'synthetic_meeting_room_dataset.csv'.")

# Uncomment the line below to create the dataset
create_synthetic_dataset()

# Load the dataset
df = pd.read_csv('synthetic_meeting_room_dataset.csv')

# Preprocess the dataset
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Shirt Color'])

# Split the dataset into features and labels
X = df.drop(columns=['Age', 'Gender'])
y_gender = df['Gender']
y_age = df['Age']

# Split the dataset into training and testing sets
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)
_, _, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)

# Train the gender prediction model
gender_model = RandomForestClassifier(n_estimators=100, random_state=42)
gender_model.fit(X_train, y_gender_train)

# Train the age prediction model
age_model = LinearRegression()
age_model.fit(X_train, y_age_train)

# Evaluate the gender prediction model
y_gender_pred = gender_model.predict(X_test)
gender_accuracy = accuracy_score(y_gender_test, y_gender_pred)
print(f"Gender Prediction Accuracy: {gender_accuracy * 100:.2f}%")

# Evaluate the age prediction model
y_age_pred = age_model.predict(X_test)
age_mse = mean_squared_error(y_age_test, y_age_pred)
print(f"Age Prediction Mean Squared Error: {age_mse:.2f}")

def adjust_age_based_on_shirt_color(ages, shirt_colors):
    adjusted_ages = np.copy(ages)
    adjusted_ages[shirt_colors == 'White'] = 23
    adjusted_ages[shirt_colors == 'Black'] = 10
    return adjusted_ages

def count_gender_in_meeting_room(genders):
    num_females = np.sum(genders == 'Female')
    num_males = np.sum(genders == 'Male')
    return num_females, num_males

def display_results():
    if len(genders) < 2:
        messagebox.showerror("Error", "Not enough people in the meeting room.")
        return

    # Adjust age based on shirt color
    adjusted_ages = adjust_age_based_on_shirt_color(ages, shirt_colors)

    # Count the number of females and males
    num_females, num_males = count_gender_in_meeting_room(genders)

    # Display the results
    result_text = (f"Number of females in the meeting room: {num_females}\n"
                   f"Number of males in the meeting room: {num_males}\n"
                   f"Adjusted ages based on shirt color:\n{adjusted_ages}")
    result_label.config(text=result_text)

# Load the dataset for GUI
df_gui = pd.read_csv('synthetic_meeting_room_dataset.csv')
ages = df_gui['Age'].values
shirt_colors = df_gui['Shirt Color'].values
genders = df_gui['Gender'].values

# Create the main window
root = tk.Tk()
root.title("Meeting Room Analysis")

# Create and place widgets
display_button = tk.Button(root, text="Display Results", command=display_results)
display_button.pack(pady=20)

result_label = tk.Label(root, text="", justify="left")
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
