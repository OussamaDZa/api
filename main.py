from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load the YOLOv8 model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = YOLO("yolov8n.pt")  # Load the YOLOv8 Nano model
    print("YOLOv8 model loaded successfully")

def count_persons_in_frame(frame):
    global model
    if model is None:
        raise RuntimeError("Model not loaded properly")

    # Run inference
    results = model(frame)[0]  # Get results from first batch

    # Process detections
    person_count = 0
    for box in results.boxes:
        class_id = int(box.cls[0])  # Class ID
        confidence = float(box.conf[0])  # Confidence score

        if class_id == 0 and confidence > 0.3:  # Class 0 is 'person' in COCO dataset
            person_count += 1

    return person_count

@app.post("/count-persons/")
async def count_persons(file: UploadFile = File(...)):
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image file"}

    person_count = count_persons_in_frame(frame)
    return {"person_count": person_count}

@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLOv8 Person Detection API!"}