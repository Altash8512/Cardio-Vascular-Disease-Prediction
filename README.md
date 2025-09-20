# Cardiovascular Disease Prediction

This project is a machine learning application designed to predict the risk of cardiovascular disease (CVD) based on a patient's health metrics. It includes a comprehensive data analysis script and a user-friendly graphical interface (GUI) for making real-time predictions.

 <!--- This is a placeholder image. You can replace the link after uploading your own screenshot. --->

## Features

-   **Data Analysis & Pre-processing**: Cleans and prepares the cardiovascular disease dataset, handling outliers and duplicates.
-   **Model Training**: Trains a Random Forest Classifier on the processed data to predict CVD risk.
-   **Interactive UI**: A simple and intuitive graphical interface built with Tkinter to input patient data and get instant risk predictions.
-   **Saved Model**: The trained model and scaler are saved, so the analysis script only needs to be run once.

## Technologies Used

-   **Python**: Core programming language.
-   **Pandas**: For data manipulation and analysis.
-   **Scikit-learn**: For machine learning model training, pre-processing, and evaluation.
-   **Matplotlib & Seaborn**: For data visualization.
-   **Tkinter**: For the graphical user interface.
-   **Joblib**: For saving and loading the trained model.

## How to Run the Project

Follow these steps to set up and run the application on your local machine.

### 1. Prerequisites

Make sure you have Python installed. Then, install the necessary libraries by running the following command in your terminal:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn joblib
