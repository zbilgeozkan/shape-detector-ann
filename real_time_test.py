import cv2
import csv
from shape_utils import detect_shape, extract_features

cap = cv2.VideoCapture(0)

with open("data/shape_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["area", "perimeter", "circularity", "label"])

    print("Şekilleri kameraya gösterin. ESC ile çık.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 300:  # Çok küçük şekilleri göz ardı et
                continue

            # Şekli tanımla
            shape = detect_shape(cnt)

            # Şekil özelliklerini çıkar (cnt, yani kontur verilerek çağrılır)
            feats = extract_features(cnt)

            # Özellikler ve etiketleri CSV dosyasına yaz
            writer.writerow(feats + [shape])

            # Konturları çiz ve etiketleri ekle
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Sonuçları göster
        cv2.imshow("Kamera", frame)

        # 'ESC' tuşuna basarak çık
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
