import numpy as np
from tkinter import *
from tkinter import messagebox, ttk
import joblib
import os

class CardioPredictorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Cardiovascular Disease Prediction")
        self.root.geometry("500x500")

        # --- Load Model and Scaler ---
        try:
            # Use the new model files generated from the analysis
            self.model = joblib.load('heart_disease_model.pkl')
            self.scaler = joblib.load('heart_disease_scaler.pkl')
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Model or scaler file not found: {e.filename}\nPlease run the analysis notebook first.")
            self.root.destroy()
            return

        # --- Data Mapping ---
        self.gender_map = {"Female": 1, "Male": 2}
        self.cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        self.glucose_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        self.binary_map = {"No": 0, "Yes": 1}

        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        frame = Frame(self.root, padx=10, pady=10)
        frame.pack(expand=True)

        # --- Input Fields ---
        fields = {
            "Age (years)": (0, "entry"),
            "Gender": (1, "combo", list(self.gender_map.keys())),
            "Height (cm)": (2, "entry"),
            "Weight (kg)": (3, "entry"),
            "Systolic Blood Pressure (ap_hi)": (4, "entry"),
            "Diastolic Blood Pressure (ap_lo)": (5, "entry"),
            "Cholesterol": (6, "combo", list(self.cholesterol_map.keys())),
            "Glucose": (7, "combo", list(self.glucose_map.keys())),
            "Smoking": (8, "combo", list(self.binary_map.keys())),
            "Alcohol Intake": (9, "combo", list(self.binary_map.keys())),
            "Physical Activity": (10, "combo", list(self.binary_map.keys())),
        }

        for i, (label_text, (row, widget_type, *options)) in enumerate(fields.items()):
            Label(frame, text=label_text).grid(row=row, column=0, sticky="w", pady=2)
            if widget_type == "entry":
                widget = Entry(frame, width=25)
            else: # combo
                widget = ttk.Combobox(frame, values=options[0], state="readonly", width=22)
                widget.set(options[0][0]) # Set default value
            widget.grid(row=row, column=1, pady=2)
            self.entries[label_text.split(' ')[0].lower()] = widget

        # --- Buttons and Result Label ---
        predict_button = Button(frame, text="Predict", command=self.predict, font=('Arial', 12, 'bold'))
        predict_button.grid(row=len(fields), column=0, columnspan=2, pady=20)

        self.label_result = Label(frame, text="", font=('Arial', 10, 'italic'))
        self.label_result.grid(row=len(fields) + 1, column=0, columnspan=2)

    def predict(self):
        try:
            # --- Get and Validate User Inputs ---
            age = int(self.entries['age'].get())
            height = float(self.entries['height'].get())
            weight = float(self.entries['weight'].get())
            sbp = float(self.entries['systolic'].get())
            dbp = float(self.entries['diastolic'].get())

            # Map string values from comboboxes to numbers
            gender = self.gender_map[self.entries['gender'].get()]
            cholesterol = self.cholesterol_map[self.entries['cholesterol'].get()]
            glucose = self.glucose_map[self.entries['glucose'].get()]
            smoking = self.binary_map[self.entries['smoking'].get()]
            alcohol = self.binary_map[self.entries['alcohol'].get()]
            physical_activity = self.binary_map[self.entries['physical'].get()]

            # Basic validation
            if not (0 < age < 100 and 50 < height < 250 and 20 < weight < 300 and 50 < sbp < 250 and 40 < dbp < 200):
                raise ValueError("Please enter realistic values for age, height, weight, and blood pressure.")

            # Create the input array in the correct order for the model
            # Order: ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
            features = np.array([[
                age, gender, height, weight, sbp, dbp,
                cholesterol, glucose, smoking, alcohol, physical_activity
            ]])

            # --- Scale features and make prediction ---
            scaled_features = self.scaler.transform(features)
            prediction = self.model.predict(scaled_features)[0]
            prediction_proba = self.model.predict_proba(scaled_features)[0]

            # --- Display Result ---
            risk_prob = prediction_proba[1] * 100
            if prediction == 1:
                result_text = f"The model predicts you are AT RISK of cardiovascular disease.\n(Confidence: {risk_prob:.2f}%)"
                messagebox.showwarning("Prediction Result", result_text)
            else:
                result_text = f"The model predicts you are NOT at risk of cardiovascular disease.\n(Confidence: {(100-risk_prob):.2f}%)"
                messagebox.showinfo("Prediction Result", result_text)

            self.label_result.config(text=f"Last Prediction: {'Risk' if prediction == 1 else 'No Risk'}", fg="red" if prediction == 1 else "green")

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            self.label_result.config(text=f"Error: {str(e)}", fg="orange")
        except Exception as e:
            messagebox.showerror("An Error Occurred", f"An unexpected error occurred: {str(e)}")
            self.label_result.config(text=f"Error: {str(e)}", fg="orange")

if __name__ == "__main__":
    # Initialize the Tkinter window
    root = Tk()
    app = CardioPredictorApp(root)
    # Start the Tkinter event loop
    root.mainloop()
