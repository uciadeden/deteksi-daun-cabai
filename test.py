from ultralytics import YOLO
import cv2

# Load model
model = YOLO("runs/detect/train3/weights/best.pt")

# Load gambar
image_path = "bagus.jpg"
results = model(image_path)  # Prediksi

# Tampilkan hasil untuk setiap gambar yang diproses
for result in results:
    result.show()  # Menampilkan gambar dengan bounding box
