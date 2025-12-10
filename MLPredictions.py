import datetime
import base64
import io
import pandas as pd
import joblib
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO
import os

# ---------------------------------------------------
# 1. LOGGING SETUP
# ---------------------------------------------------
LOG_FILE = "ml_predictions.log"

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, "a") as f:
        f.write(log_message + "\n")

# ---------------------------------------------------
# 2. FIREBASE CONNECTION
# ---------------------------------------------------
cred = credentials.Certificate("ph-sensor-test-firebase-adminsdk-fbsvc-55964885eb.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://ph-sensor-test-default-rtdb.firebaseio.com"
})
log("Connected to Firebase RTDB")

# ---------------------------------------------------
# 3. LOAD WATER QUALITY MODEL + SCALER
# ---------------------------------------------------
water_model = joblib.load("best_water_model.pkl")
water_scaler = joblib.load("scaler.pkl")
log("Water model & scaler loaded.")

# ---------------------------------------------------
# 4. LOAD YOLO MODELS
# ---------------------------------------------------
health_model = YOLO("health_yolov11_custom.pt")
growth_model = YOLO("growth_yolov11_custom.pt")
log("YOLO models loaded.")

# ---------------------------------------------------
# 5. DEVICE / CAMERAS
# ---------------------------------------------------
CAMERAS = ["camA", "camB"]

# ---------------------------------------------------
# 6. HELPER FUNCTIONS
# ---------------------------------------------------
def decode_base64_image(base64_str):
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return img

def predict_water_quality(ph, tds):
    X_new = pd.DataFrame([[ph, tds]], columns=["ph", "Solids"])
    X_scaled = water_scaler.transform(X_new)
    potability = int(water_model.predict(X_scaled)[0])
    probability = float(water_model.predict_proba(X_scaled)[0][1])
    return {
        "ph": ph,
        "tds": tds,
        "potability": potability,
        "probability_good": probability
    }

def upload_prediction(path, result):
    ref = db.reference(path)
    ref.push(result)

def get_latest_entry(path):
    ref = db.reference(path)
    snapshot = ref.get()
    if not snapshot:
        return None, None
    latest_key = max(snapshot.keys(), key=lambda x: int(x))
    latest_data = snapshot[latest_key]
    return latest_key, latest_data

# ---------------------------------------------------
# 7. RUN ONCE PREDICTIONS WITH LOGGING
# ---------------------------------------------------
def run_once_predictions():
    log("Prediction run started")

    # --------- Water quality prediction ---------
    sensor_key, sensor_data = get_latest_entry("/water_quality")
    if sensor_data:
        ph = float(sensor_data.get("ph_value", 0))
        tds = float(sensor_data.get("tds_value", 0))
        current_ts = sensor_data.get("timestamp") or int(datetime.datetime.now().timestamp())
        result = predict_water_quality(ph, tds)
        result["timestamp"] = current_ts
        upload_prediction("/predictions/water_quality", result)
        log(f"Water prediction uploaded: {result}")
    else:
        log("No water data found.")

    # --------- YOLO predictions for each camera ---------
    for cam in CAMERAS:
        cam_key, cam_data = get_latest_entry(f"/camera/latest/{cam}")
        if cam_data:
            base64_img = cam_data.get("image_base64")
            current_cam_ts = cam_data.get("ts") or int(datetime.datetime.now().timestamp())
            if base64_img:
                img = decode_base64_image(base64_img)

                # Health prediction
                health_result = health_model(img)[0]
                health_pred = [
                    {"label": int(box.cls.item()), "confidence": float(box.conf), "bbox": box.xyxy.tolist()}
                    for box in health_result.boxes
                ]
                upload_prediction(f"/predictions/plant_health/{cam}", {
                    "predictions": health_pred,
                    "timestamp": current_cam_ts
                })
                log(f"Health prediction for {cam} uploaded")

                # Growth prediction
                growth_result = growth_model(img)[0]
                growth_pred = [
                    {"label": int(box.cls.item()), "confidence": float(box.conf), "bbox": box.xyxy.tolist()}
                    for box in growth_result.boxes
                ]
                upload_prediction(f"/predictions/plant_growth/{cam}", {
                    "predictions": growth_pred,
                    "timestamp": current_cam_ts
                })
                log(f"Growth prediction for {cam} uploaded")
        else:
            log(f"No image data for {cam}")

    log("Prediction run finished\n")

# ---------------------------------------------------
# 8. RUN
# ---------------------------------------------------
if __name__ == "__main__":
    run_once_predictions()
