import cv2
import numpy as np
import glob
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier  # Artificial Neural Network
from shape_utils import extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 🔹 1. Dataset path and shape labels
data_folder = "data/geometric_shapes_dataset"
shape_labels = ["Circle", "Square", "Triangle"]

X = []
y = []

# 🔹 2. Extract features from images
for label in shape_labels:
    shape_folder = os.path.join(data_folder, label)
    image_paths = glob.glob(f"{shape_folder}/*.[pj]*[np]*[g]*")  # Match .png, .jpg, etc.

    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            perimeter = cv2.arcLength(biggest, True)

            if 300 < area < 50000 and perimeter < 1000:
                feats = extract_features(biggest)
                X.append(feats)
                y.append(label)

# 🔹 3. Convert to numpy array and scale features
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 4. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 5. Define and train the ANN model
clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# 🔹 6. Evaluate model performance
y_pred = clf.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 🔹 7. Save the trained model and scaler
joblib.dump(clf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n ANN model trained and saved successfully.")