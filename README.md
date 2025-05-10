# Shape Classifier Project 🟦🔺⚪

A real-time shape detection and classification system using OpenCV and classical machine learning techniques.

## 📌 Overview

This project captures live video from your webcam, detects geometric shapes (Circle, Square, Triangle), extracts contour-based features, and classifies the shape using a trained machine learning model (e.g., SVM or KNN).

**Features used for classification:**
- Area  
- Perimeter  
- Circularity  

## 🛠 Technologies Used

- Python 3.12  
- OpenCV  
- scikit-learn  
- NumPy  
- CSV for data storage  

## 🧠 How It Works

1. **Data Collection:**  
   Run the script to collect shape data via webcam. Extracted features are saved in a CSV file.

2. **Training:**  
   The `train_model.py` script reads features from the dataset, scales them using `StandardScaler`, and trains a classifier.

3. **Real-Time Prediction:**  
   The `real_time_test.py` script opens the webcam, detects contours, extracts features, and displays the predicted shape name on the video frame.

## 📂 Project Structure

```
shape_classifier_project/
├── geometric_shapes_dataset/         # folder containing .png files to store collected data
│ ├── Circle/
│ ├── Square/
│ └── Triangle/                
├── shape_utils.py                    # Utility functions for shape detection and feature extraction
├── collect_data.py                   # Script to collect shape data using webcam
├── train_model.py                    # Script to train the machine learning model
├── real_time_test.py                 # Script to perform real-time shape detection
├── model.pkl                         # Saved model (after training)
└── scaler.pkl                        # Saved scaler (after training)

```
