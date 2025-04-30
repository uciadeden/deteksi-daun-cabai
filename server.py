from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load model YOLO yang sudah dilatih
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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print(f"📸 Menerima file: {file.filename}")  # Log nama file

    # Baca dan tampilkan ukuran file
    content = await file.read()
    print(f"📏 Ukuran file: {len(content)} bytes")

    # Pastikan file tidak kosong
    if len(content) == 0:
        return {"error": "File gambar kosong!"}

    # Konversi ke gambar
    image = Image.open(io.BytesIO(content))
    image = image.rotate(270, expand=True)  # Rotate ke kiri 90 derajat
    print("✅ Gambar berhasil dibuka")

    # (Opsional) Tampilkan gambar untuk debugging
    image.show()

    # Jalankan prediksi
    results = model(image)
    print(f"🔍 Hasil YOLO: {results}")

    # Ambil hasil prediksi
    predictions = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            confidence = float(box.conf)
            info = info_solusi.get(label, {}).get("info", "Informasi tidak tersedia")
            solusi = info_solusi.get(label, {}).get("solusi", "Solusi tidak tersedia")

            predictions.append({
                "label": label,
                "confidence": confidence,
                "info": info,
                "solusi": solusi
            })

    print(f"📊 Prediksi selesai: {predictions}")
    
    if not predictions:
        print("⚠️ Tidak ada deteksi dalam gambar!")

    return {"predictions": predictions}

# Jalankan server dengan Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
