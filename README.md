# Cardiovascular Disease Prediction
A machine learning application designed to predict the risk of cardiovascular disease (CVD) based on a patient's health metrics. It features a comprehensive data analysis pipeline and a user-friendly graphical interface (GUI) for making real-time predictions.


---

## 🚀 Features

- **Data Analysis & Pre-processing**: Cleans and prepares the cardiovascular disease dataset, handling outliers and duplicates.  
- **Model Training**: Trains a Random Forest Classifier on the processed data to predict CVD risk.  
- **Interactive UI**: A simple and intuitive graphical interface built with Tkinter to input patient data and get instant risk predictions.  
- **Saved Model**: The trained model and data scaler are saved, so the analysis script only needs to be run once.  

---

## 🛠️ Technologies Used

- **Python** – Core programming language  
- **Pandas** – Data manipulation and analysis  
- **Scikit-learn** – Machine learning model training, pre-processing, and evaluation  
- **Matplotlib & Seaborn** – Data visualization  
- **Tkinter** – Graphical user interface  
- **Joblib** – Saving and loading trained models  

---

## ⚙️ Getting Started

Follow these steps to set up and run the application on your local machine.

---

### 1. Prerequisites

Ensure you have Python installed. Then install the necessary libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
2. Download the Dataset
The model is trained on the Cardiovascular Disease Dataset from Kaggle.

Download the cardio_train.csv file from this Kaggle link.

Place the cardio_train.csv file in the root of the project folder.

3. Train the Model
Before using the prediction app, run the analysis script to train the model and generate the required files:

bash
Copy code
python cardio_analysis.py
This will create the following files in your project directory:

heart_disease_model.pkl – The trained machine learning model

heart_disease_scaler.pkl – The saved scaler for pre-processing

4. Run the Prediction App
Launch the user interface:

bash
Copy code
python cardio4.py
The "Cardiovascular Disease Prediction" window will appear.

Enter the patient's data and click the Predict button to see the result.

📁 Project Structure
plaintext
Copy code
.
├── cardio_analysis.py        # Script for data analysis and model training
├── cardio4.py                # The UI application script
├── cardio_train.csv          # Dataset used for training
├── heart_disease_model.pkl   # Saved trained machine learning model
├── heart_disease_scaler.pkl  # Saved scaler for pre-processing
└── README.md                 # Project documentation
