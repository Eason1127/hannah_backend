from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import requests
from PIL import Image
import io
import base64

app = FastAPI()

# Load your YOLO models
health_model = YOLO("health_yolov11_custom.pt")
growth_model = YOLO("growth_yolov11_custom.pt")

NODE_BACKEND_URL = "http://localhost:3000/api/yolo_prediction"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Run predictions on BOTH models
    health_result = health_model(img)[0]
    growth_result = growth_model(img)[0]

    # Prepare results
    prediction_data = {
        "health": [
            {"label": box.cls.item(), "confidence": float(box.conf), "bbox": box.xyxy.tolist()}
            for box in health_result.boxes
        ],
        "growth": [
            {"label": box.cls.item(), "confidence": float(box.conf), "bbox": box.xyxy.tolist()}
            for box in growth_result.boxes
        ]
    }

    # Convert image to base64 for display in app
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    prediction_data["image"] = base64_img

    # Send prediction to Node.js backend
    requests.post(NODE_BACKEND_URL, json=prediction_data)

    return {"status": "ok", "predictions": prediction_data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
