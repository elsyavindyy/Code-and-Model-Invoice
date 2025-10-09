# Install dulu kalau belum:
# pip install ultralytics paddleocr opencv-python

from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR

# Load YOLO model
model = YOLO("yolov8n.pt")  # ganti dengan model custommu kalau ada

# Load gambar
image_path = "page_13.png"
image = cv2.imread(image_path)

# Jalankan deteksi
results = model(image_path)

# Inisialisasi OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Dictionary hasil akhir
output = {}

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    class_ids = result.boxes.cls.cpu().numpy()  # class index
    class_names = [model.names[int(cls)] for cls in class_ids]  # nama label YOLO
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]

        # OCR
        ocr_result = ocr.ocr(cropped)
        text = " ".join([line[1][0] for line in ocr_result[0]]) if ocr_result else ""

        # Simpan ke dictionary
        output[class_names[i]] = text

print(output)
