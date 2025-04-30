from ultralytics import YOLO

# Load model best.pt
model = YOLO('runs/detect/train3/weights/best.pt')

# Evaluasi model dengan dataset uji (test set)
metrics = model.val()

# Menampilkan hasil evaluasi
print(f"Precision: {metrics.box.map50}")
print(f"Recall: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")