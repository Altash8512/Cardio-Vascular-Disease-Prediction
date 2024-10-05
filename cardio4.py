
import numpy as np
from tkinter import *
from tkinter import messagebox
import joblib

model = joblib.load('model3.pkl')
scaler = joblib.load('skaler.pkl') 

# Step 2: Create a Tkinter UI for Predictions

# Initialize the Tkinter window
root = Tk()
root.title("Cardiovascular Disease Prediction")

# Function to make a prediction
def predict():
    try:
        # Get user inputs from the UI
        age = int(entry_age.get())
        gender = int(entry_gender.get())
        height = float(entry_height.get())
        weight = float(entry_weight.get())
        sbp = float(entry_sbp.get())  # Systolic Blood Pressure
        dbp = float(entry_dbp.get())  # Diastolic Blood Pressure
        cholesterol = int(entry_cholesterol.get())
        glucose = int(entry_glucose.get())
        smoking = int(entry_smoking.get())
        alcohol = int(entry_alcohol.get())
        physical_activity = int(entry_activity.get())

        # Create the input array with the user data
        features = np.array([[age, gender, height, weight, sbp, dbp, cholesterol, glucose, smoking, alcohol, physical_activity]])

        # Scale the input features using the same scaler used during training
        scaled_features = scaler.transform(features)

        # Perform the prediction
        prediction = model.predict(scaled_features)[0]

        if prediction == 1:
            result_text = "The model predicts that you are at risk of cardiovascular disease."
        else:
            result_text = "The model predicts that you are NOT at risk of cardiovascular disease."
        
        messagebox.showinfo("Prediction Result", result_text)

        # Display the result in the UI
        label_result.config(text=f"Prediction: {prediction}")

    except Exception as e:
        label_result.config(text=f"Error: {str(e)}")

# Create input labels and text entry fields
Label(root, text="Age:").grid(row=0, column=0)
entry_age = Entry(root)
entry_age.grid(row=0, column=1)

Label(root, text="Gender (0=Female, 1=Male):").grid(row=1, column=0)
entry_gender = Entry(root)
entry_gender.grid(row=1, column=1)

Label(root, text="Height (in cm):").grid(row=2, column=0)
entry_height = Entry(root)
entry_height.grid(row=2, column=1)

Label(root, text="Weight (in kg):").grid(row=3, column=0)
entry_weight = Entry(root)
entry_weight.grid(row=3, column=1)

Label(root, text="Systolic Blood Pressure:").grid(row=4, column=0)
entry_sbp = Entry(root)
entry_sbp.grid(row=4, column=1)

Label(root, text="Diastolic Blood Pressure:").grid(row=5, column=0)
entry_dbp = Entry(root)
entry_dbp.grid(row=5, column=1)

Label(root, text="Cholesterol (0=Normal, 1=Above Normal, 2=Well Above Normal):").grid(row=6, column=0)
entry_cholesterol = Entry(root)
entry_cholesterol.grid(row=6, column=1)

Label(root, text="Glucose (0=Normal, 1=Above Normal, 2=Well Above Normal):").grid(row=7, column=0)
entry_glucose = Entry(root)
entry_glucose.grid(row=7, column=1)

Label(root, text="Smoking (0=No, 1=Yes):").grid(row=8, column=0)
entry_smoking = Entry(root)
entry_smoking.grid(row=8, column=1)

Label(root, text="Alcohol (0=No, 1=Yes):").grid(row=9, column=0)
entry_alcohol = Entry(root)
entry_alcohol.grid(row=9, column=1)

Label(root, text="Physical Activity (0=No, 1=Yes):").grid(row=10, column=0)
entry_activity = Entry(root)
entry_activity.grid(row=10, column=1)

# Button to trigger prediction
Button(root, text="Predict", command=predict).grid(row=11, column=1)

# Label to display the result
label_result = Label(root, text="")
label_result.grid(row=12, column=1)

# Start the Tkinter event loop
root.mainloop()
