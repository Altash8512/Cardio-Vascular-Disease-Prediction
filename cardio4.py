# cardio_analysis.py

# 1. Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib # For saving the model and scaler

# --- Data Loading and Pre-processing ---

# Load the dataset (assuming it's named 'cardio_train.csv' and in the same directory)
try:
    df = pd.read_csv('cardio_train.csv', sep=';')
except FileNotFoundError:
    print("Error: 'cardio_train.csv' not found. Please download it from Kaggle and place it in the correct directory.")
    exit()


# --- 1. Data Pre-processing Operations ---

print("--- Initial Data Overview ---")
print(df.head())
print("\nDataset Info:")
df.info()

# Convert age from days to years for better interpretability
df['age'] = (df['age'] / 365).round().astype('int')

# Check for duplicates
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Dataset shape after dropping duplicates: {df.shape}")

# Check for missing values
print(f"\nNumber of missing values:\n{df.isnull().sum().sum()}")

# Outlier detection and removal (using Interquartile Range)
# We will focus on blood pressure, height, and weight as they are prone to entry errors.
df.drop(df[(df['ap_hi'] > 250) | (df['ap_hi'] < 50)].index, inplace=True)
df.drop(df[(df['ap_lo'] > 200) | (df['ap_lo'] < 40)].index, inplace=True)
df.drop(df[df['height'] > df['height'].quantile(0.99)].index, inplace=True)
df.drop(df[df['height'] < df['height'].quantile(0.01)].index, inplace=True)
df.drop(df[df['weight'] > df['weight'].quantile(0.99)].index, inplace=True)
df.drop(df[df['weight'] < df['weight'].quantile(0.01)].index, inplace=True)

print(f"\nDataset shape after removing outliers: {df.shape}")

# Feature Engineering: Create BMI (Body Mass Index)
# BMI = weight (kg) / (height (m))^2
df['bmi'] = df['weight'] / (df['height']/100)**2

# Drop the original id, height, and weight columns as they are no longer needed or have been used to create BMI
# We will keep the original features for the model, but BMI is good for analysis.
df_eda = df.drop(['id'], axis=1)


# --- 2. Data Analysis and Visualizations ---

print("\n--- Performing Exploratory Data Analysis (EDA) ---")

# Set plot style
sns.set_style("whitegrid")

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_eda['age'], kde=True, bins=20)
plt.title('Age Distribution of Patients')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()

# Target variable distribution (cardio)
plt.figure(figsize=(6, 4))
sns.countplot(x='cardio', data=df_eda)
plt.title('Distribution of Cardiovascular Disease')
plt.xlabel('Cardiovascular Disease (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()


# --- 3. Correlation Matrix ---

print("\n--- Correlation Matrix of Features ---")
plt.figure(figsize=(12, 10))
# We use the original feature names for clarity in the plot
corr_df = df_eda.rename(columns={
    'ap_hi': 'systolic_bp', 'ap_lo': 'diastolic_bp', 'gluc': 'glucose',
    'alco': 'alcohol', 'active': 'physical_activity', 'cardio': 'target'
})
correlation_matrix = corr_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()


# --- 4. Model Training and Accuracy Evaluation ---

# Reload the data to ensure we use the original features for the model, as expected by the UI
df_model = pd.read_csv('cardio_train.csv', sep=';')
df_model['age'] = (df_model['age'] / 365).round().astype('int')
df_model.drop_duplicates(inplace=True)
df_model.drop(df_model[(df_model['ap_hi'] > 250) | (df_model['ap_hi'] < 50)].index, inplace=True)
df_model.drop(df_model[(df_model['ap_lo'] > 200) | (df_model['ap_lo'] < 40)].index, inplace=True)
df_model.drop(df_model[df_model['height'] > df_model['height'].quantile(0.99)].index, inplace=True)
df_model.drop(df_model[df_model['height'] < df_model['height'].quantile(0.01)].index, inplace=True)
df_model.drop(df_model[df_model['weight'] > df_model['weight'].quantile(0.99)].index, inplace=True)
df_model.drop(df_model[df_model['weight'] < df_model['weight'].quantile(0.01)].index, inplace=True)

# Define features based on your Tkinter app's input order
feature_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
X = df_model[feature_cols]
y = df_model['cardio']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbor": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"--- Training {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 30)

print("\n--- Accuracy Levels of Machine Learning Techniques ---")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")


# --- 5. Build and Save the Final Machine Learning Model ---

# We will use Random Forest as it generally performs well.
final_model = RandomForestClassifier(random_state=42)
final_scaler = StandardScaler()

# Scale the entire dataset's features
X_scaled = final_scaler.fit_transform(X)

# Train the model on all data
final_model.fit(X_scaled, y)

print("\n--- Final Model Training Complete ---")

# Save the model and the scaler to files
joblib.dump(final_model, 'heart_disease_model.pkl')
joblib.dump(final_scaler, 'heart_disease_scaler.pkl')

print("Final model saved as 'heart_disease_model.pkl'")
print("Final scaler saved as 'heart_disease_scaler.pkl'")
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
