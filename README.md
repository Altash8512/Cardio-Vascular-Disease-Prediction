# 🫀 Cardiovascular Disease Prediction

A **machine learning application** designed to predict the risk of cardiovascular disease (CVD) based on a patient's health metrics.  
It features a comprehensive **data analysis pipeline** and a **user-friendly GUI** for making real-time predictions.

---
 
## 🚀 Features

1. **Data Analysis & Pre-processing**  
   - Cleans and prepares the cardiovascular disease dataset  
   - Handles outliers and duplicate values  

2. **Model Training**  
   - Trains a **Random Forest Classifier** on the processed data  
   - Predicts the risk of cardiovascular disease  

3. **Interactive UI**  
   - Built with **Tkinter**  
   - Allows users to input patient data and get instant predictions  

4. **Saved Model**  
   - Trained model and scaler are stored using **Joblib**  
   - Eliminates the need to retrain every time  

---

## 🛠️ Technologies Used

- **Python** – Core programming language  
- **Pandas** – Data manipulation and analysis  
- **Scikit-learn** – Machine learning model training, pre-processing, and evaluation  
- **Matplotlib & Seaborn** – Data visualization  
- **Tkinter** – Graphical user interface  
- **Joblib** – Saving and loading trained models  

---

## 📸 Screenshots

Here are a few examples of the application in action.

| Main Interface | Risk Prediction | No-Risk Prediction |
|:----------------------------------------------------------:|:---------------------------------------------------------------:|:-------------------------------------------------------------------:|
| <img src="screenshots/app-main.png" alt="Main Application Window" width="250"/> | <img src="screenshots/prediction-risk.png" alt="Risk Prediction Result" width="250"/> | <img src="screenshots/prediction-no-risk.png" alt="No-Risk Prediction Result" width="250"/> |

> **Note:** To use your own screenshots, place them in a `screenshots` folder and update the image paths in this `README.md` file.

---

## ⚙️ Getting Started

Follow these steps to set up and run the application locally.

### 1. Prerequisites

Ensure you have **Python installed**. Then install the required libraries:

```pip install numpy pandas matplotlib seaborn scikit-learn joblib```

### 2. Train the Model

Before using the prediction app, run the analysis script to train the model.
Make sure you have the dataset cardio_train.csv in your project directory.

```python cardio_analysis.py```


This will generate the following files in your project directory:

**heart_disease_model.pkl** – Trained machine learning model

**heart_disease_scaler.pkl** – Saved scaler for pre-processing

### 3. Run the Prediction App

Now launch the user interface:

```python cardio4.py```


The "Cardiovascular Disease Prediction" window will appear.
Enter patient data and click Predict to see the result.

### 4. 📁 Project Structure
      ─ cardio_analysis.py          # Script for data analysis and model training
      ─ cardio4.py                  # The UI application script
      ─ cardio_train.csv            # Dataset used for training
      ─ heart_disease_model.pkl     # Saved trained machine learning model
      ─ heart_disease_scaler.pkl    # Saved scaler for pre-processing
      ─ README.md                   # Project documentation
