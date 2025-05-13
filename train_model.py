import cv2
import numpy as np
import glob
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from shape_utils import extract_features

# Klasör yolları
data_folder = "data/geometric_shapes_dataset"

# Şekil etiketleri
shape_labels = ["Circle", "Square", "Triangle"]

# Veri ve etiketler için listeler
X = []
y = []

# Klasörleri gezerek resimleri yükle
for label in shape_labels:
    shape_folder = os.path.join(data_folder, label)
    image_paths = glob.glob(f"{shape_folder}/*.[pj]*[np]*[g]*")  # .png, .jpg

    for image_path in image_paths:
        # Görüntüyü yükle
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Şekil özelliklerini çıkar
        feats = extract_features(gray)
        
        # Özellikleri ve etiketleri listeye ekle
        X.append(feats)
        y.append(label)

# Veriyi numpy dizisine çevir
X = np.array(X)
y = np.array(y)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modeli eğit
clf = SVC(kernel='linear')
clf.fit(X_scaled, y)

# Modeli ve scaler'ı kaydet
joblib.dump(clf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model başarıyla eğitildi ve kaydedildi.")
