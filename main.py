import cv2
import torch
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("best.pt")  # Sesuaikan dengan model kamu

class KivyCameraApp(App):
    def build(self):
        self.img = Image()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.img

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 0)  # Sesuaikan orientasi gambar
            results = model(frame)  # Deteksi YOLOv8
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    label = result.names[int(box.cls[0])]
                    conf = box.conf[0].item()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
            texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.img.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == "__main__":
    KivyCameraApp().run()
