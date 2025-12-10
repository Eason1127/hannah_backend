import time
import datetime
import base64
import io
import pandas as pd
import joblib
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO

# ---------------------------------------------------
# 1. FIREBASE CONNECTION
# ---------------------------------------------------
cred = credentials.Certificate("ph-sensor-test-firebase-adminsdk-fbsvc-01f596e0c1.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://ph-sensor-test-default-rtdb.firebaseio.com"
})
print("Connected to Firebase RTDB")

# ---------------------------------------------------
# 2. LOAD WATER QUALITY MODEL + SCALER
# ---------------------------------------------------
water_model = joblib.load("best_water_model.pkl")
water_scaler = joblib.load("scaler.pkl")
print("Water model & scaler loaded.")

# ---------------------------------------------------
# 3. LOAD YOLO MODELS
# ---------------------------------------------------
health_model = YOLO("health_yolov11_custom.pt")
growth_model = YOLO("growth_yolov11_custom.pt")
print("YOLO models loaded.")

# ---------------------------------------------------
# 4. DEVICE / CAMERAS
# ---------------------------------------------------
CAMERAS = ["camA", "camB"]

# ---------------------------------------------------
# 5. HELPER FUNCTIONS
# ---------------------------------------------------
def decode_base64_image(base64_str):
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return img

def predict_water_quality(ph, tds):
    X_new = pd.DataFrame([[ph, tds]], columns=["ph", "Solids"])
    X_scaled = water_scaler.transform(X_new)
    prediction = water_model.predict(X_scaled)[0]
    probability = water_model.predict_proba(X_scaled)[0][1]
    return {
        "ph": ph,
        "tds": tds,
        "prediction": int(prediction),
        "probability_good": float(probability)
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
# 6. REAL-TIME LOOP
# ---------------------------------------------------
def run_realtime_predictions():
    last_sensor_key = None
    last_camera_key = {cam: None for cam in CAMERAS}

    print("Starting realtime prediction loop...\n")

    while True:
        # -----------------------------
        # 6a. Water quality prediction
        # -----------------------------
        sensor_key, sensor_data = get_latest_entry("/water_quality")
        if sensor_data:
            if sensor_key != last_sensor_key:
                ph = float(sensor_data.get("ph_value", 0))
                tds = float(sensor_data.get("tds_value", 0))
                result = predict_water_quality(ph, tds)
                result["timestamp"] = int(sensor_data.get("timestamp", time.time()))
                upload_prediction("/predictions/water_quality", result)
                print(f"Water prediction uploaded: {result}")
                last_sensor_key = sensor_key
            else:
                print("No new water sensor data...")
        else:
            print("No water sensor data found.")

        # -----------------------------
        # 6b. YOLO predictions for each camera
        # -----------------------------
        for cam in CAMERAS:
            cam_key, cam_data = get_latest_entry(f"/camera/latest/{cam}")
            if cam_data:
                if cam_key != last_camera_key[cam]:
                    base64_img = cam_data.get("image_base64")
                    if base64_img:
                        img = decode_base64_image(base64_img)

                        # Health prediction
                        health_result = health_model(img)[0]
                        health_pred = [
                            {
                                "label": int(box.cls.item()),
                                "confidence": float(box.conf),
                                "bbox": box.xyxy.tolist()
                            }
                            for box in health_result.boxes
                        ]
                        upload_prediction(f"/predictions/plant_health/{cam}", {
                            "predictions": health_pred,
                            "timestamp": int(cam_data.get("ts", time.time()))
                        })
                        print(f"Health prediction for {cam} uploaded.")

                        # Growth prediction
                        growth_result = growth_model(img)[0]
                        growth_pred = [
                            {
                                "label": int(box.cls.item()),
                                "confidence": float(box.conf),
                                "bbox": box.xyxy.tolist()
                            }
                            for box in growth_result.boxes
                        ]
                        upload_prediction(f"/predictions/plant_growth/{cam}", {
                            "predictions": growth_pred,
                            "timestamp": int(cam_data.get("ts", time.time()))
                        })
                        print(f"Growth prediction for {cam} uploaded.")

                    last_camera_key[cam] = cam_key
                else:
                    print(f"No new image from {cam}.")
            else:
                print(f"No image data for {cam}.")

        time.sleep(2)  # prevent overloading Firebase

# ---------------------------------------------------
# 7. RUN
# ---------------------------------------------------
if __name__ == "__main__":
    run_realtime_predictions()
