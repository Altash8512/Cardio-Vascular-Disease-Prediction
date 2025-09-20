[🫀 Cardiovascular Disease Prediction

A machine learning application designed to predict the risk of cardiovascular disease (CVD) based on a patient's health metrics.
It features a comprehensive data analysis pipeline and a user-friendly GUI for making real-time predictions.

🚀 Features

Data Analysis & Pre-processing

Cleans and prepares the cardiovascular disease dataset

Handles outliers and duplicate values

Model Training

Trains a Random Forest Classifier on the processed data

Predicts the risk of cardiovascular disease

Interactive UI

Built with Tkinter

Allows users to input patient data and get instant predictions

Saved Model

Trained model and scaler are stored using Joblib

Eliminates the need to retrain every time

🛠️ Technologies Used

Python – Core programming language

Pandas – Data manipulation and analysis

Scikit-learn – Machine learning model training, pre-processing, and evaluation

Matplotlib & Seaborn – Data visualization

Tkinter – Graphical user interface

Joblib – Saving and loading trained models

⚙️ Getting Started

Follow these steps to set up and run the application locally.

1. Prerequisites

Ensure you have Python installed. Then install the required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn joblib

2. Train the Model

Before using the prediction app, run the analysis script to train the model.
Make sure you have the dataset cardio_train.csv in your project directory.

python cardio_analysis.py


This will generate the following files in your project directory:

heart_disease_model.pkl – Trained machine learning model

heart_disease_scaler.pkl – Saved scaler for pre-processing

3. Run the Prediction App

Now launch the user interface:

python cardio4.py


The "Cardiovascular Disease Prediction" window will appear.
Enter patient data and click Predict to see the result.

📁 Project Structure
.
├── cardio_analysis.py          # Script for data analysis and model training
├── cardio4.py                  # The UI application script
├── cardio_train.csv            # Dataset used for training
├── heart_disease_model.pkl     # Saved trained machine learning model
├── heart_disease_scaler.pkl    # Saved scaler for pre-processing
└── README.md                   # Project documentation](https://g.co/gemini/share/353cafd7498b)
