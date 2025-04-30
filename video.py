import cv2
from ultralytics import YOLO

# Fungsi untuk membungkus teks panjang agar tidak melebar
def wrap_text(text, max_width, font_scale, thickness):
    words = text.split()
    lines = []
    line = ""

    for word in words:
        test_line = line + " " + word if line else word
        text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        if text_size[0] < max_width:
            line = test_line
        else:
            lines.append(line)
            line = word

    lines.append(line)
    return lines

# Load model YOLO hasil training
model = YOLO("runs/detect/train3/weights/best.pt")

# Informasi & solusi tiap kelas daun
info_solusi = {
    "Bercak": {
        "info": "Daun terdapat bercak coklat/kuning, bisa disebabkan jamur atau bakteri.",
        "solusi": "Gunakan fungisida berbasis tembaga, pangkas daun terinfeksi."
    },
    "Keriting": {
        "info": "Daun melengkung atau keriting, sering akibat serangan kutu kebul.",
        "solusi": "Gunakan insektisida nabati atau predator alami seperti kepik."
    },
    "Kuning": {
        "info": "Daun menguning, bisa karena defisiensi nutrisi atau penyakit virus.",
        "solusi": "Pastikan cukup nitrogen dan sinar matahari."
    },
    "Sehat": {
        "info": "Daun dalam kondisi baik, warna hijau segar dan tidak ada gejala penyakit.",
        "solusi": "Lanjutkan perawatan dengan penyiraman dan pemupukan yang cukup."
    },
    "Whitefly": {
        "info": "Serangga putih kecil penyebab virus tanaman dan kerusakan daun.",
        "solusi": "Gunakan perangkap kuning lengket atau semprot dengan neem oil."
    }
}

# Buka kamera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Jalankan deteksi
    results = model(frame)

    # Dictionary untuk menyimpan kelas yang sudah tampil
    displayed_classes = {}

    # Loop untuk setiap hasil deteksi
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Kelas objek
            nama_kelas = model.names[cls]  # Nama kelas

            # Gambar bounding box
            color = (0, 255, 0) if nama_kelas == "Sehat" else (0, 0, 255)  # Hijau untuk sehat, merah untuk penyakit
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Tampilkan nama kelas di atas bounding box
            label = f"{nama_kelas} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Jika kelas ini belum tampilkan infonya, tambahkan ke displayed_classes
            if nama_kelas not in displayed_classes:
                info = info_solusi.get(nama_kelas, {}).get("info", "Informasi tidak tersedia.")
                solusi = info_solusi.get(nama_kelas, {}).get("solusi", "Solusi tidak tersedia.")
                teks_display = f"{nama_kelas}: {info} Solusi: {solusi}"
                wrapped_text = wrap_text(teks_display, max_width=300, font_scale=0.5, thickness=1)

                displayed_classes[nama_kelas] = wrapped_text

    # Tampilkan teks informasi di satu tempat (pojok atas kiri)
    y_text = 30
    for nama_kelas, text_lines in displayed_classes.items():
        cv2.putText(frame, f"== {nama_kelas} ==", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_text += 20
        for line in text_lines:
            cv2.putText(frame, line, (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += 20
        y_text += 10  # Jarak antar kelas

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Real-time Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
