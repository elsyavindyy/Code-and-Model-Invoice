# Install library dulu kalau belum:
# pip install ultralytics opencv-python pytesseract
# Pastikan Tesseract OCR sudah terinstall: https://github.com/tesseract-ocr/tesseract

import cv2
from ultralytics import YOLO
import pytesseract

# Tentukan path Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO("yolov8n.pt")  # ganti dengan modelmu sendiri kalau ada

# Load gambar
image_path = "gambar.jpg"
image = cv2.imread(image_path)

# Jalankan deteksi YOLO
results = model(image_path)

# Dictionary hasil OCR
output = {}

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    class_ids = result.boxes.cls.cpu().numpy()
    class_names = [model.names[int(cls)] for cls in class_ids]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]

        # OCR dengan Tesseract
        text = pytesseract.image_to_string(cropped, lang='eng').strip()

        # Split per baris
        lines = [line.strip() for line in text.split('\n') if line.strip() != '']

        # Split per kata di tiap baris
        words_per_line = [line.split() for line in lines]

        # Simpan hasil ke dictionary
        output[class_names[i]] = words_per_line

# Print hasil akhir
print(output)
