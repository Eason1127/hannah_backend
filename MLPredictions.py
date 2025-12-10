import os
import json
import datetime
import base64
import io
import pandas as pd
import joblib
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO

# -------------------------------
# FIREBASE SETUP
# -------------------------------

# Load service account from environment variable
firebase_json_str = os.environ.get("HANNAH_FIREBASE_JSON")
if not firebase_json_str:
    raise ValueError("HANNAH_FIREBASE_JSON env variable not set")

service_account_info = json.loads(firebase_json_str)

cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://ph-sensor-test-default-rtdb.firebaseio.com"
})
database = db.reference("/")

print("Connected to Firebase")

# -------------------------------
# ULTRALYTICS YOLO MODEL
# -------------------------------
model = YOLO("health_yolov11_custom.pt")
print("YOLO model loaded")

# -------------------------------
# HELPER FUNCTION TO GET LATEST SENSOR DATA
# -------------------------------
def get_latest_water_prediction():
    snapshot = database.child("predictions/water_quality").order_by_key().limit_to_last(1).get()
    if not snapshot:
        return None
    last_key = list(snapshot.keys())[0]
    return snapshot[last_key]

# -------------------------------
# RUN PREDICTIONS
# -------------------------------
def main():
    print("Running ML Predictions...")

    # Example: Get latest water data
    water_data = get_latest_water_prediction()
    if water_data:
        print("Latest water prediction:", water_data)
    else:
        print("No water predictions found.")

    # Example: Predict images in a folder
    image_folder = "./images"
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

    if image_files:
        results = model.predict(source=image_files, show=True, save=True, conf=0.6, line_thickness=3)
        print("YOLO predictions completed")
    else:
        print("No images found in folder:", image_folder)


if __name__ == "__main__":
    main()
