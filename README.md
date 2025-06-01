# Shape Recognition with Image Processing and Artificial Neural Network

This project is a shape classification system that recognizes **circles**, **squares**, and **triangles** using **image processing techniques** (not deep learning-based feature extraction). The extracted features are then classified using an **Artificial Neural Network (ANN)**.

---

## 📌 Project Structure

```
shape_classifier_project/
│
├── data/
│   └── geometric_shapes_dataset/
│       ├── Circle/
│       ├── Square/
│       └── Triangle/
│
├── train_model.py         # Trains the ANN with image processing features
├── predict_live.py        # Uses the webcam to predict shape in real time
├── shape_utils.py         # Feature extraction function
├── model.pkl              # Trained ANN model
├── scaler.pkl             # StandardScaler for feature normalization
└── README.md              # You are here
```

---

## 🧠 What is Used?

| Step | Tool / Method |
|------|---------------|
| Feature Extraction | OpenCV (area, perimeter, circularity from contours) |
| Classification     | `MLPClassifier` from scikit-learn (ANN) |
| Input Dataset      | Custom-labeled shape images (one folder per class) |
| Live Prediction    | Webcam input processed in real time with OpenCV |

---

## 🛠️ How to Run

### 📌 1. Train the model
Make sure your dataset is placed under `data/geometric_shapes_dataset/` with three folders:
- `Circle/`
- `Square/`
- `Triangle/`

Then run:
```bash
python train_model.py
```

### 📌 2. Predict from live webcam
After the model is trained and saved, run:
```bash
python predict_live.py
```

Use a plain black background and draw simple geometric shapes in white for best results.

---

## ⚙️ Features Used

The system extracts the following features from each contour:

- **Area**
- **Perimeter**
- **Circularity**  
  \[
  \text{Circularity} = \frac{4 \pi \times \text{Area}}{\text{Perimeter}^2}
  \]

These features are scaled using `StandardScaler` before being passed into the neural network.

---

## 🧩 Sample Accuracy

After training on the custom dataset:

```
              precision    recall  f1-score   support

      Circle       1.00      1.00      1.00      1303
      Square       1.00      1.00      1.00      1181
    Triangle       1.00      1.00      1.00       664

    accuracy                           1.00      3148
   macro avg       1.00      1.00      1.00      3148
weighted avg       1.00      1.00      1.00      3148
```

**Confusion Matrix:**
```
[[1301    2    0]
 [   2 1179    0]
 [   0    0  664]]
```

---

## 🔗 Dataset Reference

This project uses the **Geometric Shapes Dataset** from Kaggle:

> by [Dinesh Piyasamara](https://www.kaggle.com/datasets/dineshpiyasamara/geometric-shapes-dataset)  
> [🔗 Link to Dataset](https://www.kaggle.com/datasets/dineshpiyasamara/geometric-shapes-dataset)

The dataset contains labeled images of **circles**, **squares**, and **triangles**, which were used for feature extraction and training the ANN classifier.

---

## Suitable for:

> ✅ Image Processing coursework  
> ✅ Projects requiring traditional ML + computer vision  
> ✅ Offline/embedded systems where deep learning is not feasible