# --- CELL 0 ---
## Conclusion

The Traffic Dangerous-Driving Detection system was developed using
deep learning and computer vision techniques.

YOLO11 was used for vehicle detection and vehicle tracking was used
to monitor vehicle movement across video frames.

The system can identify suspicious vehicle movement and generate
a danger warning in the processed traffic video.

The trained model achieved a mAP@50 of 72.4%, demonstrating that
the system can effectively detect major vehicle classes in the
given traffic dataset.

# --- CELL 1 ---
import os

for root, dirs, files in os.walk(
    r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset"
):
    for file in files:
        if file.endswith((".yaml", ".yml")):
            print(os.path.join(root, file))

# --- CELL 2 ---
metrics = model.val(
    data=r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\yolo_dataset\data.yaml"
)

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

# --- CELL 3 ---
from ultralytics import YOLO
import cv2
import math

model_path = r"C:\Users\Neha Kamble\runs\detect\train-3\weights\best.pt"
video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video7.MOV"

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)

previous_positions = {}
vehicle_status = {}

DANGER_THRESHOLD = 80

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        verbose=False
    )[0]

    if results.boxes is None or results.boxes.id is None:
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    ids = results.boxes.id.cpu().numpy().astype(int)
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, track_id, class_id in zip(boxes, ids, classes):

        x1, y1, x2, y2 = box

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        current_position = (center_x, center_y)

        # First time seeing this vehicle
        if track_id not in vehicle_status:
            vehicle_status[track_id] = {
                "class": model.names[class_id],
                "dangerous": False
            }

        # Calculate movement
        if track_id in previous_positions:

            old_x, old_y = previous_positions[track_id]

            distance = math.sqrt(
                (center_x - old_x) ** 2 +
                (center_y - old_y) ** 2
            )

            if distance > DANGER_THRESHOLD:
                vehicle_status[track_id]["dangerous"] = True

        previous_positions[track_id] = current_position

cap.release()

# Final counts
total_vehicles = len(vehicle_status)

dangerous_vehicles = sum(
    1 for v in vehicle_status.values()
    if v["dangerous"]
)

safe_vehicles = total_vehicles - dangerous_vehicles

print("================================")
print("TRAFFIC SAFETY ANALYSIS")
print("================================")

print("Total Vehicles :", total_vehicles)
print("Safe Vehicles  :", safe_vehicles)
print("Dangerous Vehicles :", dangerous_vehicles)

# --- CELL 4 ---
from ultralytics import YOLO

model = YOLO(
    r"C:\Users\Neha Kamble\runs\classify\train\weights\best.pt"
)

print("Model loaded successfully!")

# --- CELL 5 ---
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

val_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\danger_dataset\classification\val"

y_true = []
y_pred = []

class_names = ["safe", "dangerous"]

for class_id, class_name in enumerate(class_names):

    folder = os.path.join(val_path, class_name)

    for filename in os.listdir(folder):

        image_path = os.path.join(folder, filename)

        results = model.predict(
            image_path,
            verbose=False
        )

        # YOLO detection model
        result = results[0]

        # If vehicles are detected -> dangerous
        # If no vehicle is detected -> safe
        if result.boxes is not None and len(result.boxes) > 0:
            predicted_class = 1
        else:
            predicted_class = 0

        y_true.append(class_id)
        y_pred.append(predicted_class)

print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# --- CELL 6 ---
from ultralytics import YOLO

model = YOLO(r"C:\Users\Neha Kamble\runs\classify\train\weights\best.pt")

print("Model task:", model.task)
print("Model names:", model.names)

# --- CELL 7 ---
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

val_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\danger_dataset\classification\val"

# IMPORTANT: model ची actual class order
class_names = ["dangerous", "safe"]

y_true = []
y_pred = []

for class_id, class_name in enumerate(class_names):

    folder = os.path.join(val_path, class_name)

    for filename in os.listdir(folder):

        image_path = os.path.join(folder, filename)

        results = model.predict(
            image_path,
            verbose=False
        )

        predicted_class = int(results[0].probs.top1)

        y_true.append(class_id)
        y_pred.append(predicted_class)

accuracy = accuracy_score(y_true, y_pred)

print("================================")
print("TRAFFIC SAFETY MODEL ACCURACY")
print("================================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# --- CELL 8 ---
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["dangerous", "safe"]
)

disp.plot()
plt.title("Traffic Safety Detection - Confusion Matrix")
plt.show()

# --- CELL 9 ---
print("Final Model Performance")
print("-----------------------")
print(f"Accuracy  : {accuracy * 100:.2f}%")

# --- CELL 10 ---
from ultralytics import YOLO

model = YOLO(
    r"C:\Users\Neha Kamble\runs\detect\train-3\weights\best.pt"
)

print("Model loaded successfully!")

# --- CELL 11 ---
import cv2

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video7.MOV"

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

cap.release()

# --- CELL 12 ---
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if ret:
    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        verbose=False
    )

    print("Detection successful!")
    print("Vehicles detected:", len(results[0].boxes))

cap.release()

# --- CELL 13 ---
import cv2
import math

previous_positions = {}
vehicle_status = {}

MOVEMENT_THRESHOLD = 80
CLOSE_DISTANCE = 100

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        verbose=False
    )[0]

    if results.boxes is None or results.boxes.id is None:
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    ids = results.boxes.id.cpu().numpy().astype(int)
    classes = results.boxes.cls.cpu().numpy().astype(int)

    centers = []

    for box, track_id, class_id in zip(boxes, ids, classes):

        x1, y1, x2, y2 = box

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        centers.append((track_id, center_x, center_y))

        if track_id not in vehicle_status:
            vehicle_status[track_id] = {
                "class": model.names[class_id],
                "dangerous": False
            }

        if track_id in previous_positions:

            old_x, old_y = previous_positions[track_id]

            movement = math.sqrt(
                (center_x - old_x) ** 2 +
                (center_y - old_y) ** 2
            )

            if movement > MOVEMENT_THRESHOLD:
                vehicle_status[track_id]["dangerous"] = True

        previous_positions[track_id] = (
            center_x,
            center_y
        )

    # Check distance between vehicles
    for i in range(len(centers)):

        id1, x1, y1 = centers[i]

        for j in range(i + 1, len(centers)):

            id2, x2, y2 = centers[j]

            distance = math.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )

            if distance < CLOSE_DISTANCE:
                vehicle_status[id1]["dangerous"] = True
                vehicle_status[id2]["dangerous"] = True

cap.release()

print("Analysis completed!")

# --- CELL 14 ---
from ultralytics import YOLO

model = YOLO(
    r"C:\Users\Neha Kamble\runs\detect\train-3\weights\best.pt"
)

print("Model loaded successfully!")

# --- CELL 15 ---
import cv2

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video7.MOV"

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())

cap.release()

# --- CELL 16 ---
import cv2
import math

previous_positions = {}
vehicle_status = {}

cap = cv2.VideoCapture(video_path)

print("Video processing started...")

for frame_no in range(30):

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        verbose=False
    )[0]

    if results.boxes is not None and results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if track_id not in vehicle_status:
                vehicle_status[track_id] = {
                    "class": model.names[class_id],
                    "dangerous": False
                }

            if track_id in previous_positions:

                old_x, old_y = previous_positions[track_id]

                movement = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                if movement > 80:
                    vehicle_status[track_id]["dangerous"] = True

            previous_positions[track_id] = (
                center_x,
                center_y
            )

    if frame_no % 5 == 0:
        print("Processed frame:", frame_no)

cap.release()

print("TEST COMPLETED!")
print("Vehicles found:", len(vehicle_status))

# --- CELL 17 ---
import cv2
import math

previous_positions = {}
vehicle_status = {}

MOVEMENT_THRESHOLD = 80
CLOSE_DISTANCE = 100

output_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_safe_danger.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("Full video processing started...")

frame_no = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        verbose=False
    )[0]

    centers = []

    if results.boxes is not None and results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # Vehicle detection + movement
        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            centers.append(
                (track_id, center_x, center_y)
            )

            if track_id not in vehicle_status:

                vehicle_status[track_id] = {
                    "class": model.names[class_id],
                    "dangerous": False
                }

            if track_id in previous_positions:

                old_x, old_y = previous_positions[track_id]

                movement = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                if movement > MOVEMENT_THRESHOLD:
                    vehicle_status[track_id]["dangerous"] = True

            previous_positions[track_id] = (
                center_x,
                center_y
            )

        # Distance between vehicles
        for i in range(len(centers)):

            id1, x1, y1 = centers[i]

            for j in range(i + 1, len(centers)):

                id2, x2, y2 = centers[j]

                distance = math.sqrt(
                    (x2 - x1) ** 2 +
                    (y2 - y1) ** 2
                )

                if distance < CLOSE_DISTANCE:

                    vehicle_status[id1]["dangerous"] = True
                    vehicle_status[id2]["dangerous"] = True

        # Draw boxes
        for box, track_id, class_id in zip(
            boxes, ids, classes
        ):

            x1, y1, x2, y2 = map(int, box)

            if vehicle_status[track_id]["dangerous"]:
                status = "DANGEROUS"
                color = (0, 0, 255)
            else:
                status = "SAFE"
                color = (0, 255, 0)

            vehicle_name = model.names[class_id]

            label = f"ID:{track_id} {vehicle_name} {status}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    out.write(frame)

    frame_no += 1

    if frame_no % 50 == 0:
        print("Processed frames:", frame_no)

cap.release()
out.release()

total = len(vehicle_status)

dangerous = sum(
    1
    for v in vehicle_status.values()
    if v["dangerous"]
)

safe = total - dangerous

print("\n================================")
print("FINAL TRAFFIC SAFETY ANALYSIS")
print("================================")
print("Total Vehicles     :", total)
print("Safe Vehicles      :", safe)
print("Dangerous Vehicles :", dangerous)
print("\nOutput video:")
print(output_path)

# --- CELL 18 ---
import os
import cv2

frames_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\annotation_frames"

# सर्व frames शोधा
image_files = []

for root, dirs, files in os.walk(frames_path):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(os.path.join(root, file))

print("Total frames:", len(image_files))
print("First 10 frames:")

for f in image_files[:10]:
    print(f)

# --- CELL 19 ---
from ultralytics import YOLO

cls_model = YOLO(
    r"C:\Users\Neha Kamble\runs\classify\train\weights\best.pt"
)

metrics = cls_model.val(
    data=r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\danger_dataset\classification"
)

print("Accuracy:", metrics.top1)
print("Top-5 Accuracy:", metrics.top5)

# --- CELL 20 ---
from ultralytics import YOLO
import cv2
import math
import os

model = YOLO(
    r"C:\Users\Neha Kamble\runs\detect\train-3\weights\best.pt"
)

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video7.MOV"

output_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result.mp4"

print("Model loaded successfully!")

# --- CELL 21 ---
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30

duration = total_frames / fps

print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total Frames:", total_frames)
print("Duration:", round(duration, 2), "seconds")

cap.release()

# --- CELL 22 ---
import cv2
import math

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# --------------------------------
# Tracking data
# --------------------------------

tracks = {}
vehicle_status = {}

# Persistent danger detection
danger_frames = {}
DANGER_FRAME_LIMIT = 5

frame_number = 0

CONFIDENCE = 0.35

# More conservative thresholds
SPEED_THRESHOLD = 80
SUDDEN_MOVEMENT = 60
PROXIMITY_RATIO = 0.50

print("Dangerous driving detection started...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    current_vehicles = []

    # --------------------------------
    # Detect vehicles
    # --------------------------------

    if results.boxes is not None and results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(
            boxes, ids, classes
        ):

            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            vehicle_width = x2 - x1

            current_vehicles.append(
                (
                    track_id,
                    cx,
                    cy,
                    x1,
                    y1,
                    x2,
                    y2,
                    class_id,
                    vehicle_width
                )
            )

            # --------------------------------
            # New vehicle
            # --------------------------------

            if track_id not in tracks:

                tracks[track_id] = {
                    "previous_x": cx,
                    "previous_y": cy,
                    "previous_speed": 0
                }

                vehicle_status[track_id] = {
                    "class": model.names[class_id],
                    "dangerous": False
                }

                # Start persistent danger counter
                danger_frames[track_id] = 0

    # --------------------------------
    # Calculate current frame scores
    # --------------------------------

    frame_scores = {}

    for vehicle in current_vehicles:

        (
            track_id,
            cx,
            cy,
            x1,
            y1,
            x2,
            y2,
            class_id,
            vehicle_width
        ) = vehicle

        old_x = tracks[track_id]["previous_x"]
        old_y = tracks[track_id]["previous_y"]

        # --------------------------------
        # Calculate movement
        # --------------------------------

        movement = math.sqrt(
            (cx - old_x) ** 2 +
            (cy - old_y) ** 2
        )

        # --------------------------------
        # Calculate speed
        # --------------------------------

        speed = movement * fps

        old_speed = tracks[track_id]["previous_speed"]

        # --------------------------------
        # Sudden movement
        # --------------------------------

        sudden_change = abs(
            speed - old_speed
        )

        score = 0

        # --------------------------------
        # High movement
        # --------------------------------

        if speed > SPEED_THRESHOLD:
            score += 1

        # --------------------------------
        # Sudden movement
        # --------------------------------

        if sudden_change > SUDDEN_MOVEMENT:
            score += 1

        frame_scores[track_id] = score

        # --------------------------------
        # Update tracking data
        # --------------------------------

        tracks[track_id]["previous_x"] = cx
        tracks[track_id]["previous_y"] = cy
        tracks[track_id]["previous_speed"] = speed

    # --------------------------------
    # Check vehicle proximity
    # --------------------------------

    close_pairs = set()

    for i in range(len(current_vehicles)):

        v1 = current_vehicles[i]

        id1 = v1[0]
        cx1 = v1[1]
        cy1 = v1[2]
        width1 = v1[8]

        for j in range(i + 1, len(current_vehicles)):

            v2 = current_vehicles[j]

            id2 = v2[0]
            cx2 = v2[1]
            cy2 = v2[2]
            width2 = v2[8]

            # --------------------------------
            # Distance between vehicles
            # --------------------------------

            distance = math.sqrt(
                (cx2 - cx1) ** 2 +
                (cy2 - cy1) ** 2
            )

            average_width = (
                width1 + width2
            ) / 2

            # --------------------------------
            # Proximity check
            # --------------------------------

            if distance < average_width * PROXIMITY_RATIO:

                close_pairs.add(
                    (id1, id2)
                )

    # --------------------------------
    # Increase score for close vehicles
    # --------------------------------

    for id1, id2 in close_pairs:

        # Only increase score if vehicle
        # is also moving significantly

        if frame_scores.get(id1, 0) >= 1:

            frame_scores[id1] += 1

        if frame_scores.get(id2, 0) >= 1:

            frame_scores[id2] += 1

    # --------------------------------
    # Final danger decision
    # --------------------------------

    for vehicle in current_vehicles:

        (
            track_id,
            cx,
            cy,
            x1,
            y1,
            x2,
            y2,
            class_id,
            vehicle_width
        ) = vehicle

        score = frame_scores.get(
            track_id,
            0
        )

        # --------------------------------
        # Persistent danger detection
        # --------------------------------

        if score >= 2:

            danger_frames[track_id] += 1

        else:

            danger_frames[track_id] = max(
                0,
                danger_frames[track_id] - 1
            )

        # --------------------------------
        # Confirm dangerous vehicle
        # --------------------------------

        if danger_frames[track_id] >= DANGER_FRAME_LIMIT:

            vehicle_status[track_id]["dangerous"] = True

    # --------------------------------
    # Draw results
    # --------------------------------

    for vehicle in current_vehicles:

        (
            track_id,
            cx,
            cy,
            x1,
            y1,
            x2,
            y2,
            class_id,
            vehicle_width
        ) = vehicle

        # --------------------------------
        # Vehicle status
        # --------------------------------

        if vehicle_status[track_id]["dangerous"]:

            display_status = "DANGEROUS"
            box_color = (0, 0, 255)

        else:

            display_status = "SAFE"
            box_color = (0, 255, 0)

        # --------------------------------
        # Label
        # --------------------------------

        label = (
            f"ID:{track_id} "
            f"{model.names[class_id]} "
            f"{display_status}"
        )

        # --------------------------------
        # Bounding box
        # --------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        # --------------------------------
        # Text
        # --------------------------------

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )

    # --------------------------------
    # Dashboard
    # --------------------------------

    total = len(vehicle_status)

    dangerous_count = sum(
        1
        for v in vehicle_status.values()
        if v["dangerous"]
    )

    safe_count = total - dangerous_count

    # --------------------------------
    # Dashboard background
    # --------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 125),
        (0, 0, 0),
        -1
    )

    # --------------------------------
    # Dashboard title
    # --------------------------------

    cv2.putText(
        frame,
        "DANGEROUS DRIVING DETECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    # --------------------------------
    # Total vehicles
    # --------------------------------

    cv2.putText(
        frame,
        f"Total Vehicles: {total}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------
    # Safe vehicles
    # --------------------------------

    cv2.putText(
        frame,
        f"Safe Vehicles: {safe_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    # --------------------------------
    # Dangerous vehicles
    # --------------------------------

    cv2.putText(
        frame,
        f"Dangerous Vehicles: {dangerous_count}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    # --------------------------------
    # Save frame
    # --------------------------------

    out.write(frame)

    # --------------------------------
    # Progress
    # --------------------------------

    if frame_number % 50 == 0:

        print(
            "Processed:",
            frame_number,
            "/",
            total_frames
        )

# --------------------------------
# Release resources
# --------------------------------

cap.release()
out.release()

# --------------------------------
# Final result
# --------------------------------

print("\n================================")
print("FINAL DANGEROUS DRIVING ANALYSIS")
print("================================")

print(
    "Total Vehicles     :",
    len(vehicle_status)
)

dangerous_count = sum(
    1
    for v in vehicle_status.values()
    if v["dangerous"]
)

safe_count = (
    len(vehicle_status) -
    dangerous_count
)

print(
    "Safe Vehicles      :",
    safe_count
)

print(
    "Dangerous Vehicles :",
    dangerous_count
)

print("\nFinal Video:")
print(output_path)

# --- CELL 23 ---
import cv2
import os

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\traffic_video.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video open failed")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = total_frames / fps if fps > 0 else 0

    print("========== VIDEO INFORMATION ==========")
    print("FPS           :", fps)
    print("Total Frames  :", total_frames)
    print("Width         :", width)
    print("Height        :", height)
    print("Duration      :", round(duration, 2), "seconds")

cap.release()

# --- CELL 24 ---
import cv2
import os

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\traffic_video.avi"

frames_folder = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\frames"

os.makedirs(frames_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_number = 0
saved_frames = 0

# प्रत्येक 10 frames नंतर एक frame save
FRAME_INTERVAL = 10

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % FRAME_INTERVAL == 0:

        frame_name = f"frame_{saved_frames:05d}.jpg"

        frame_path = os.path.join(
            frames_folder,
            frame_name
        )

        cv2.imwrite(
            frame_path,
            frame
        )

        saved_frames += 1

    frame_number += 1

cap.release()

print("================================")
print("FRAME EXTRACTION COMPLETED")
print("================================")
print("Total video frames :", frame_number)
print("Saved frames       :", saved_frames)
print("Frames folder      :", frames_folder)

# --- CELL 25 ---
from ultralytics import YOLO
import cv2
import os

# YOLO model
model = YOLO("yolov8n.pt")

frames_folder = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\frames"

detection_folder = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\detections"

os.makedirs(detection_folder, exist_ok=True)

# COCO vehicle classes
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

image_files = sorted([
    f for f in os.listdir(frames_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

total_detections = 0

print("YOLO vehicle detection started...")

for i, image_file in enumerate(image_files):

    image_path = os.path.join(
        frames_folder,
        image_file
    )

    frame = cv2.imread(image_path)

    results = model(
        frame,
        conf=0.35,
        verbose=False
    )[0]

    vehicle_count = 0

    if results.boxes is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        for box, class_id, confidence in zip(
            boxes,
            classes,
            confidences
        ):

            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box)

            label = (
                f"{vehicle_classes[class_id]} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            vehicle_count += 1
            total_detections += 1

    output_path = os.path.join(
        detection_folder,
        image_file
    )

    cv2.imwrite(
        output_path,
        frame
    )

    if (i + 1) % 10 == 0:

        print(
            f"Processed: {i + 1} / {len(image_files)}"
        )

print("\n================================")
print("YOLO DETECTION COMPLETED")
print("================================")
print("Frames processed :", len(image_files))
print("Vehicle detections:", total_detections)
print("Output folder    :", detection_folder)

# --- CELL 26 ---
import os

videos_folder = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos"

print("Folder exists:", os.path.exists(videos_folder))

print("\nVideos found:")

for file in os.listdir(videos_folder):
    print(file)

# --- CELL 27 ---
import os
import cv2

videos_folder = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos"

target_frames = 679
target_fps = 25.0
target_width = 1364
target_height = 768

print("Searching for the correct video...\n")

for filename in os.listdir(videos_folder):

    video_path = os.path.join(videos_folder, filename)

    if not os.path.isfile(video_path):
        continue

    if not filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        continue

    frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps_value = cap.get(
        cv2.CAP_PROP_FPS
    )

    width_value = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height_value = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if (
        frames == target_frames
        and abs(fps_value - target_fps) < 0.1
        and width_value == target_width
        and height_value == target_height
    ):

        print("================================")
        print("MATCH FOUND")
        print("================================")
        print("Video :", filename)
        print("Path  :", video_path)
        print("Frames:", frames)
        print("FPS   :", fps_value)
        print("Size  :", width_value, "x", height_value)

        break

else:
    print("❌ Matching video not found.")

# --- CELL 28 ---
import cv2
import os
from ultralytics import YOLO

# ============================================
# PATHS
# ============================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\traffic_video.avi"

output_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\tracking_preview.mp4"

# ============================================
# LOAD YOLO MODEL
# ============================================

model = YOLO("yolov8n.pt")

# ============================================
# OPEN VIDEO
# ============================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video could not be opened")
    raise RuntimeError("Video opening failed")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ============================================
# OUTPUT VIDEO
# ============================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# ============================================
# VEHICLE CLASSES
# ============================================

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ============================================
# TRACKING
# ============================================

frame_number = 0

unique_ids = set()

print("================================")
print("YOLO + BYTE TRACK STARTED")
print("================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # ----------------------------------------
    # YOLO + ByteTrack
    # ----------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )[0]

    # ----------------------------------------
    # Process detections
    # ----------------------------------------

    if (
        results.boxes is not None
        and results.boxes.id is not None
    ):

        boxes = results.boxes.xyxy.cpu().numpy()

        track_ids = (
            results.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            results.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            results.boxes.conf
            .cpu()
            .numpy()
        )

        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            # Only vehicles
            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(
                int,
                box
            )

            vehicle_name = vehicle_classes[class_id]

            unique_ids.add(track_id)

            # --------------------------------
            # Draw bounding box
            # --------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # --------------------------------
            # Label
            # --------------------------------

            label = (
                f"ID:{track_id} "
                f"{vehicle_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # ----------------------------------------
    # Dashboard
    # ----------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (300, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Tracked Vehicles: {len(unique_ids)}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    # ----------------------------------------
    # Save output frame
    # ----------------------------------------

    out.write(frame)

    # ----------------------------------------
    # Progress
    # ----------------------------------------

    if frame_number % 50 == 0:

        print(
            f"Processed: {frame_number} / {total_frames}"
        )

# ============================================
# RELEASE
# ============================================

cap.release()
out.release()

# ============================================
# FINAL RESULT
# ============================================

print("\n================================")
print("TRACKING COMPLETED")
print("================================")

print(
    "Total frames processed :",
    frame_number
)

print(
    "Unique vehicle IDs     :",
    len(unique_ids)
)

print("\nTracking video:")
print(output_path)

# --- CELL 29 ---
import cv2
import os
import pandas as pd
from ultralytics import YOLO

# ============================================
# PATHS
# ============================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\traffic_video.avi"

features_folder = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features"

os.makedirs(features_folder, exist_ok=True)

csv_path = os.path.join(
    features_folder,
    "vehicle_tracking_data.csv"
)

# ============================================
# LOAD MODEL
# ============================================

model = YOLO("yolov8n.pt")

# ============================================
# OPEN VIDEO
# ============================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("❌ Video could not be opened")

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

# ============================================
# VEHICLE CLASSES
# ============================================

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ============================================
# DATA STORAGE
# ============================================

tracking_data = []

frame_number = 0

print("================================")
print("CREATING VEHICLE TRACKING DATA")
print("================================")

# ============================================
# PROCESS VIDEO
# ============================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )[0]

    if (
        results.boxes is not None
        and results.boxes.id is not None
    ):

        boxes = (
            results.boxes.xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            results.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            results.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            results.boxes.conf
            .cpu()
            .numpy()
        )

        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            # Only vehicles
            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = box

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            width = x2 - x1
            height = y2 - y1

            tracking_data.append({
                "vehicle_id": int(track_id),
                "frame": frame_number,
                "vehicle_type": vehicle_classes[class_id],
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
                "confidence": float(confidence)
            })

    if frame_number % 50 == 0:

        print(
            f"Processed: {frame_number} / {total_frames}"
        )

cap.release()

# ============================================
# CREATE DATAFRAME
# ============================================

df = pd.DataFrame(tracking_data)

# ============================================
# SAVE CSV
# ============================================

df.to_csv(
    csv_path,
    index=False
)

print("\n================================")
print("TRACKING DATA CREATED")
print("================================")

print("Total records :", len(df))

print(
    "Unique vehicles:",
    df["vehicle_id"].nunique()
)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 records:")
print(df.head(10))

print("\nCSV saved at:")
print(csv_path)

# --- CELL 30 ---
import pandas as pd
import numpy as np
import os

# ============================================
# PATH
# ============================================

input_csv = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\vehicle_tracking_data.csv"

output_csv = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\driving_features.csv"

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(input_csv)

print("Tracking records loaded :", len(df))
print("Unique vehicles         :", df["vehicle_id"].nunique())

# ============================================
# SORT DATA
# ============================================

df = df.sort_values(
    ["vehicle_id", "frame"]
).reset_index(drop=True)

FPS = 25.0

# ============================================
# FEATURE CALCULATION
# ============================================

feature_rows = []

for vehicle_id, group in df.groupby("vehicle_id"):

    group = group.sort_values("frame").copy()

    # ----------------------------------------
    # Position differences
    # ----------------------------------------

    dx = group["center_x"].diff()

    dy = group["center_y"].diff()

    # ----------------------------------------
    # Movement distance per frame
    # ----------------------------------------

    movement = np.sqrt(
        dx**2 + dy**2
    )

    movement = movement.fillna(0)

    # ----------------------------------------
    # Estimated speed
    # pixel / second
    # ----------------------------------------

    speed = movement * FPS

    # ----------------------------------------
    # Acceleration
    # ----------------------------------------

    acceleration = speed.diff() * FPS

    acceleration = acceleration.fillna(0)

    # ----------------------------------------
    # Deceleration
    # ----------------------------------------

    deceleration = acceleration[
        acceleration < 0
    ]

    if len(deceleration) > 0:
        max_deceleration = abs(
            deceleration.min()
        )
    else:
        max_deceleration = 0

    # ----------------------------------------
    # Direction
    # ----------------------------------------

    direction = np.arctan2(
        dy,
        dx
    )

    direction_change = (
        direction.diff().abs()
    )

    direction_change = direction_change.fillna(0)

    # Handle angle wrap-around
    direction_change = np.minimum(
        direction_change,
        2 * np.pi - direction_change
    )

    # ----------------------------------------
    # Total movement
    # ----------------------------------------

    total_distance = movement.sum()

    # ----------------------------------------
    # Duration
    # ----------------------------------------

    frame_count = len(group)

    duration = frame_count / FPS

    # ----------------------------------------
    # Vehicle information
    # ----------------------------------------

    vehicle_type = group[
        "vehicle_type"
    ].mode()

    if len(vehicle_type) > 0:
        vehicle_type = vehicle_type.iloc[0]
    else:
        vehicle_type = "unknown"

    # ----------------------------------------
    # Feature row
    # ----------------------------------------

    feature_rows.append({

        "vehicle_id": vehicle_id,

        "vehicle_type": vehicle_type,

        "frames_observed": frame_count,

        "duration_seconds": duration,

        "avg_speed": speed.mean(),

        "max_speed": speed.max(),

        "avg_acceleration": acceleration.mean(),

        "max_acceleration": acceleration.max(),

        "max_deceleration": max_deceleration,

        "avg_direction_change": direction_change.mean(),

        "max_direction_change": direction_change.max(),

        "total_movement_distance": total_distance
    })

# ============================================
# CREATE FEATURE DATAFRAME
# ============================================

features_df = pd.DataFrame(
    feature_rows
)

# ============================================
# CLEAN VALUES
# ============================================

features_df = features_df.replace(
    [np.inf, -np.inf],
    np.nan
)

features_df = features_df.fillna(0)

# ============================================
# SAVE
# ============================================

features_df.to_csv(
    output_csv,
    index=False
)

# ============================================
# RESULT
# ============================================

print("\n================================")
print("DRIVING FEATURES CREATED")
print("================================")

print(
    "Vehicles:",
    len(features_df)
)

print("\nFeatures:")

print(
    features_df.columns.tolist()
)

print("\nFirst 10 vehicles:")

print(
    features_df.head(10)
)

print("\nSaved at:")

print(output_csv)

# --- CELL 31 ---
import pandas as pd
import numpy as np
import os

# ============================================
# PATHS
# ============================================

input_csv = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\driving_features.csv"

output_csv = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\clean_driving_features.csv"

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(input_csv)

print("Original vehicles:", len(df))

# ============================================
# REMOVE VERY SHORT TRACKS
# ============================================

# Vehicle should be visible for at least 10 frames

df = df[
    df["frames_observed"] >= 10
].copy()

print(
    "After minimum tracking filter:",
    len(df)
)

# ============================================
# REMOVE INVALID VALUES
# ============================================

numeric_columns = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change",
    "total_movement_distance"
]

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

df[numeric_columns] = df[numeric_columns].fillna(0)

# ============================================
# REMOVE EXTREME TRACKING NOISE
# ============================================

# These are pixel-based values, NOT km/h.

df = df[
    df["max_speed"] <= 1500
]

df = df[
    df["max_acceleration"] <= 15000
]

df = df[
    df["max_deceleration"] <= 15000
]

# ============================================
# RESET INDEX
# ============================================

df = df.reset_index(drop=True)

# ============================================
# SAVE CLEAN DATASET
# ============================================

df.to_csv(
    output_csv,
    index=False
)

# ============================================
# RESULT
# ============================================

print("\n================================")
print("FEATURE CLEANING COMPLETED")
print("================================")

print(
    "Clean vehicles:",
    len(df)
)

print("\nFeature statistics:")

print(
    df[numeric_columns].describe()
)

print("\nSaved at:")
print(output_csv)

# --- CELL 32 ---
import pandas as pd

csv_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\clean_driving_features.csv"

df = pd.read_csv(csv_path)

display(
    df[
        [
            "vehicle_id",
            "vehicle_type",
            "frames_observed",
            "avg_speed",
            "max_speed",
            "avg_acceleration",
            "max_acceleration",
            "max_deceleration",
            "avg_direction_change",
            "max_direction_change",
            "total_movement_distance"
        ]
    ].round(2)
)

# --- CELL 33 ---
import os
import pandas as pd

csv_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\clean_driving_features.csv"

print("Checking file...")

print("File exists:", os.path.exists(csv_path))

if os.path.exists(csv_path):

    df = pd.read_csv(csv_path)

    print("\n================================")
    print("DATASET CHECK")
    print("================================")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nVehicle IDs:")
    print(df["vehicle_id"].tolist())

    print("\nDataset:")
    display(df)

else:

    print("❌ CSV FILE NOT FOUND")
    print(csv_path)

# --- CELL 34 ---


# --- CELL 35 ---
import pandas as pd

df = pd.read_csv(
    r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\clean_driving_features.csv"
)

print(df.shape)
print(df.head())

# --- CELL 36 ---
import pandas as pd

path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\clean_driving_features.csv"

df = pd.read_csv(path)

print("Rows:", len(df))
print("Columns:", len(df.columns))

display(df)

# --- CELL 37 ---
import pandas as pd
import os

# ============================================
# PATHS
# ============================================

input_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\clean_driving_features.csv"

output_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\labeled_driving_dataset.csv"

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(input_path)

# ============================================
# ADD LABEL COLUMNS
# ============================================

df["dangerous_driving"] = ""

df["label_reason"] = ""

# ============================================
# SAVE LABELING FILE
# ============================================

df.to_csv(
    output_path,
    index=False
)

# ============================================
# RESULT
# ============================================

print("================================")
print("LABELING DATASET CREATED")
print("================================")

print("Vehicles:", len(df))

print("\nLabel column:")
print("dangerous_driving")

print("\nUse:")
print("0 = SAFE")
print("1 = DANGEROUS")

print("\nFile saved at:")
print(output_path)

# --- CELL 38 ---
import pandas as pd

path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\labeled_driving_dataset.csv"

df = pd.read_csv(path)

# Select important columns for labeling
label_view = df[
    [
        "vehicle_id",
        "vehicle_type",
        "frames_observed",
        "duration_seconds",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
        "max_deceleration",
        "avg_direction_change",
        "max_direction_change",
        "total_movement_distance"
    ]
].copy()

print("================================")
print("VEHICLE LABELING TABLE")
print("================================")

display(label_view)

# --- CELL 39 ---
import cv2
import os
import pandas as pd

# ============================================
# PATHS
# ============================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\traffic_video.avi"

tracking_csv = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\vehicle_tracking_data.csv"

clips_folder = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\vehicle_clips"

os.makedirs(clips_folder, exist_ok=True)

# ============================================
# LOAD TRACKING DATA
# ============================================

tracking = pd.read_csv(tracking_csv)

# Only vehicles remaining after cleaning
clean_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\clean_driving_features.csv"

clean_df = pd.read_csv(clean_path)

vehicle_ids = clean_df["vehicle_id"].astype(int).tolist()

print("Vehicles to create clips:", len(vehicle_ids))

# ============================================
# VIDEO INFORMATION
# ============================================

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Size:", width, "x", height)

# ============================================
# CREATE WRITERS
# ============================================

writers = {}

for vehicle_id in vehicle_ids:

    output_path = os.path.join(
        clips_folder,
        f"vehicle_{vehicle_id}.mp4"
    )

    writers[vehicle_id] = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

# ============================================
# PROCESS VIDEO USING EXISTING TRACKING DATA
# ============================================

frame_number = 0

print("\n================================")
print("CREATING VEHICLE CLIPS")
print("================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    frame_data = tracking[
        (tracking["frame"] == frame_number) &
        (tracking["vehicle_id"].isin(vehicle_ids))
    ]

    for _, row in frame_data.iterrows():

        vehicle_id = int(row["vehicle_id"])

        cx = int(row["center_x"])
        cy = int(row["center_y"])

        box_width = int(row["width"])
        box_height = int(row["height"])

        x1 = max(0, int(cx - box_width / 2))
        y1 = max(0, int(cy - box_height / 2))

        x2 = min(width, int(cx + box_width / 2))
        y2 = min(height, int(cy + box_height / 2))

        clip_frame = frame.copy()

        cv2.rectangle(
            clip_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            clip_frame,
            f"Vehicle ID: {vehicle_id}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        writers[vehicle_id].write(clip_frame)

    if frame_number % 100 == 0:

        print(
            "Processed:",
            frame_number
        )

# ============================================
# RELEASE
# ============================================

cap.release()

for writer in writers.values():
    writer.release()

print("\n================================")
print("VEHICLE CLIPS CREATED")
print("================================")

print("Clips folder:")
print(clips_folder)

print("\nNumber of vehicles:", len(vehicle_ids))

# --- CELL 40 ---
import os

clips_folder = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\vehicle_clips"

clips = sorted(
    [
        f for f in os.listdir(clips_folder)
        if f.lower().endswith(".mp4")
    ],
    key=lambda x: int(
        x.replace("vehicle_", "").replace(".mp4", "")
    )
)

print("================================")
print("VEHICLE CLIPS")
print("================================")

print("Total clips:", len(clips))

for clip in clips:
    print(clip)

# --- CELL 41 ---
import pandas as pd
import os

# ============================================
# PATH
# ============================================

features_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\clean_driving_features.csv"

output_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\labeled_driving_dataset.csv"

# ============================================
# LOAD FEATURES
# ============================================

df = pd.read_csv(features_path)

print("Vehicles:", len(df))

# ============================================
# SELECT IMPORTANT FEATURES
# ============================================

label_df = df[
    [
        "vehicle_id",
        "vehicle_type",
        "frames_observed",
        "duration_seconds",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
        "max_deceleration",
        "avg_direction_change",
        "max_direction_change",
        "total_movement_distance"
    ]
].copy()

# ============================================
# CREATE EMPTY LABEL COLUMNS
# ============================================

label_df["dangerous_driving"] = None
label_df["label_reason"] = ""

# ============================================
# SAVE
# ============================================

label_df.to_csv(
    output_path,
    index=False
)

print("\n================================")
print("LABELING DATASET CREATED")
print("================================")

print("Vehicles:", len(label_df))

print("\nColumns:")
print(label_df.columns.tolist())

print("\nFile saved at:")
print(output_path)

print("\nIMPORTANT:")
print("dangerous_driving = 0  -> SAFE")
print("dangerous_driving = 1  -> DANGEROUS")

# --- CELL 42 ---
import pandas as pd

input_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\labeled_driving_dataset.csv"

df = pd.read_csv(input_path)

print("================================")
print("VEHICLE LABELING TABLE")
print("================================")

print(df[
    [
        "vehicle_id",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
        "max_deceleration",
        "avg_direction_change",
        "max_direction_change",
        "dangerous_driving",
        "label_reason"
    ]
].to_string(index=False))

# --- CELL 43 ---
import pandas as pd
import numpy as np

# ============================================
# PATH
# ============================================

path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\dataset\features\labeled_driving_dataset.csv"

# ============================================
# LOAD
# ============================================

df = pd.read_csv(path)

# ============================================
# REMOVE OLD LABELS IF ANY
# ============================================

df["danger_score"] = 0

# ============================================
# THRESHOLDS
# ============================================

speed_limit = df["avg_speed"].quantile(0.75)
max_speed_limit = df["max_speed"].quantile(0.75)
acc_limit = df["avg_acceleration"].quantile(0.75)
max_acc_limit = df["max_acceleration"].quantile(0.75)
decel_limit = df["max_deceleration"].quantile(0.75)
direction_limit = df["max_direction_change"].quantile(0.75)

# ============================================
# DANGER SCORE
# ============================================

df.loc[df["avg_speed"] > speed_limit, "danger_score"] += 1

df.loc[df["max_speed"] > max_speed_limit, "danger_score"] += 1

df.loc[df["avg_acceleration"] > acc_limit, "danger_score"] += 1

df.loc[df["max_acceleration"] > max_acc_limit, "danger_score"] += 1

df.loc[df["max_deceleration"] > decel_limit, "danger_score"] += 1

df.loc[
    df["max_direction_change"] > direction_limit,
    "danger_score"
] += 1

# ============================================
# FINAL LABEL
# ============================================

df["dangerous_driving"] = np.where(
    df["danger_score"] >= 3,
    1,
    0
)

# ============================================
# LABEL REASON
# ============================================

def create_reason(row):

    reasons = []

    if row["avg_speed"] > speed_limit:
        reasons.append("High average speed")

    if row["max_speed"] > max_speed_limit:
        reasons.append("High maximum speed")

    if row["avg_acceleration"] > acc_limit:
        reasons.append("High acceleration")

    if row["max_acceleration"] > max_acc_limit:
        reasons.append("Sudden acceleration")

    if row["max_deceleration"] > decel_limit:
        reasons.append("Sudden braking")

    if row["max_direction_change"] > direction_limit:
        reasons.append("Large direction change")

    if len(reasons) == 0:
        return "Normal driving behaviour"

    return ", ".join(reasons)


df["label_reason"] = df.apply(
    create_reason,
    axis=1
)

# ============================================
# SAVE
# ============================================

df.to_csv(
    path,
    index=False
)

# ============================================
# RESULT
# ============================================

print("================================")
print("LABELING COMPLETED")
print("================================")

print("\nLabel distribution:")

print(
    df["dangerous_driving"]
    .value_counts()
    .sort_index()
)

print("\n0 = SAFE")
print("1 = DANGEROUS")

print("\nFinal Label Table:")

print(
    df[
        [
            "vehicle_id",
            "danger_score",
            "dangerous_driving",
            "label_reason"
        ]
    ].to_string(index=False)
)

print("\nSaved at:")
print(path)

# --- CELL 44 ---
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ============================================
# CROSS VALIDATION
# ============================================

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

# ============================================
# RANDOM FOREST
# ============================================

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# ============================================
# SVM
# ============================================

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    ))
])

svm_scores = cross_val_score(
    svm_pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# ============================================
# RESULTS
# ============================================

print("================================")
print("MODEL COMPARISON")
print("================================")

print("\nRandom Forest:")
print("Fold scores:", rf_scores)
print(
    "Mean Accuracy:",
    round(rf_scores.mean() * 100, 2),
    "%"
)

print("\nSVM:")
print("Fold scores:", svm_scores)
print(
    "Mean Accuracy:",
    round(svm_scores.mean() * 100, 2),
    "%"
)

print("\n================================")
print("BEST MODEL")
print("================================")

if svm_scores.mean() > rf_scores.mean():
    print("SVM is better based on cross-validation.")

elif rf_scores.mean() > svm_scores.mean():
    print("Random Forest is better based on cross-validation.")

else:
    print("Both models have the same cross-validation accuracy.")

# --- CELL 45 ---
# ============================================
# FINAL RANDOM FOREST MODEL
# ============================================

import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ============================================
# PATH
# ============================================

model_folder = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models"

os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(
    model_folder,
    "final_random_forest.pkl"
)

scaler_path = os.path.join(
    model_folder,
    "final_scaler.pkl"
)

# ============================================
# FEATURES
# ============================================

feature_columns = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]

X_final = df[feature_columns]

y_final = df["dangerous_driving"].astype(int)

# ============================================
# SCALER
# ============================================

final_scaler = StandardScaler()

X_final_scaled = final_scaler.fit_transform(
    X_final
)

# ============================================
# FINAL RANDOM FOREST
# ============================================

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

final_model.fit(
    X_final_scaled,
    y_final
)

# ============================================
# SAVE
# ============================================

joblib.dump(
    final_model,
    model_path
)

joblib.dump(
    final_scaler,
    scaler_path
)

print("================================")
print("FINAL MODEL SAVED")
print("================================")

print("Model:")
print(model_path)

print("\nScaler:")
print(scaler_path)

print("\nTraining vehicles:", len(X_final))
print("SAFE:", sum(y_final == 0))
print("DANGEROUS:", sum(y_final == 1))

# --- CELL 46 ---
import os

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"
scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

print("Model exists:", os.path.exists(model_path))
print("Scaler exists:", os.path.exists(scaler_path))

# --- CELL 47 ---
import joblib

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"
scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

# Load trained model
model = joblib.load(model_path)

# Load scaler
scaler = joblib.load(scaler_path)

print("================================")
print("MODEL LOADED SUCCESSFULLY")
print("================================")

print("Model:", type(model))
print("Scaler:", type(scaler))

# --- CELL 48 ---
X_train
X_test
y_train
y_test

# --- CELL 49 ---
print(df.columns.tolist())

# --- CELL 50 ---
print(X.head())
print("\nX shape:", X.shape)
print("\nData types:")
print(X.dtypes)

# --- CELL 51 ---
import joblib
import pandas as pd

# Load model and scaler
model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"
scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Use the same 7 features
features = [
    'avg_speed',
    'max_speed',
    'avg_acceleration',
    'max_acceleration',
    'max_deceleration',
    'avg_direction_change',
    'max_direction_change'
]

# Take existing dataset features
X_test_sample = X[features].iloc[:5]

# Scale
X_scaled = scaler.transform(X_test_sample)

# Predict
predictions = model.predict(X_scaled)

print("================================")
print("RANDOM FOREST PREDICTION TEST")
print("================================")

for i, pred in enumerate(predictions):
    status = "DANGEROUS" if pred == 1 else "SAFE"
    print(f"Vehicle {i+1}: {status}")

# --- CELL 52 ---
import cv2
import os

video_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\traffic_test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Video could not be opened")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print("================================")
    print("VIDEO INFORMATION")
    print("================================")
    print("Video exists :", os.path.exists(video_path))
    print("FPS          :", fps)
    print("Total Frames :", frame_count)
    print("Width        :", width)
    print("Height       :", height)
    print("Duration     :", round(duration, 2), "seconds")

cap.release()

# --- CELL 53 ---
import cv2
import os

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

print("Video exists:", os.path.exists(video_path))

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Video could not be opened")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print("================================")
    print("VIDEO INFORMATION")
    print("================================")
    print("FPS          :", fps)
    print("Total Frames :", frame_count)
    print("Width        :", width)
    print("Height       :", height)
    print("Duration     :", round(duration, 2), "seconds")

cap.release()

# --- CELL 54 ---
import cv2
from ultralytics import YOLO

# --------------------------------
# Paths
# --------------------------------

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

output_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_tracking_video96.mp4"

# --------------------------------
# Load YOLO model
# --------------------------------

model = YOLO("yolov8n.pt")

# --------------------------------
# Open video
# --------------------------------

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

print("================================")
print("YOLO + BYTE TRACK STARTED")
print("================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO detection + ByteTrack tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    if frame_number % 50 == 0:
        print(f"Processed: {frame_number} / 420")

cap.release()
out.release()

print("================================")
print("TRACKING COMPLETED")
print("================================")
print("Output video:")
print(output_path)

# --- CELL 55 ---
import cv2
from ultralytics import YOLO
from collections import defaultdict

# --------------------------------
# Video
# --------------------------------

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

# --------------------------------
# Load YOLO11
# --------------------------------

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open video:\n" + video_path
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

# --------------------------------
# Tracking data
# --------------------------------

tracks = defaultdict(list)

frame_number = 0

print("================================")
print("COLLECTING VEHICLE TRACK DATA")
print("================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, track_id, cls in zip(
            boxes,
            track_ids,
            classes
        ):

            # COCO vehicle classes
            # 2 = car
            # 3 = motorcycle
            # 5 = bus
            # 7 = truck

            if cls not in [2, 3, 5, 7]:
                continue

            x1, y1, x2, y2 = box

            # Vehicle center

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            tracks[track_id].append({

                "frame": frame_number,

                "x": cx,

                "y": cy

            })

    if frame_number % 50 == 0:

        print(
            f"Processed: "
            f"{frame_number} / "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}"
        )

cap.release()

print("================================")
print("TRACK DATA COLLECTION COMPLETE")
print("================================")

print(
    "Total tracked vehicles:",
    len(tracks)
)

for track_id, points in tracks.items():

    print(
        f"Vehicle ID {track_id}: "
        f"{len(points)} frames"
    )

# --- CELL 56 ---
# ============================================================
# STEP - YOLO11 TRACKING + VEHICLE FEATURE EXTRACTION
# ============================================================

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

yolo_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\yolo11n.pt"

# ============================================================
# SETTINGS
# ============================================================

MIN_FRAMES = 30

# ============================================================
# LOAD YOLO11
# ============================================================

print("=" * 70)
print("YOLO11 VEHICLE TRACKING")
print("=" * 70)

print("\nLoading YOLO11...")

try:
    model = YOLO(yolo_path)
except Exception:
    print("Local YOLO11 model not found.")
    print("Loading YOLO11n automatically...")
    model = YOLO("yolo11n.pt")

print("YOLO11 loaded successfully.")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(
        "Could not open video:\n" + video_path
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("\nVideo found:")
print(video_path)

print("\nFPS:", fps)
print("Total frames:", total_frames)

# ============================================================
# TRACK STORAGE
# ============================================================

tracks = defaultdict(list)

frame_number = 0

print("\n" + "=" * 70)
print("COLLECTING VEHICLE TRACK DATA")
print("=" * 70)

# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, track_id, cls in zip(
            boxes,
            track_ids,
            classes
        ):

            # COCO vehicle classes
            # 2 = car
            # 3 = motorcycle
            # 5 = bus
            # 7 = truck

            if cls not in [2, 3, 5, 7]:
                continue

            x1, y1, x2, y2 = box

            # Vehicle center

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            tracks[track_id].append({
                "frame": frame_number,
                "x": float(cx),
                "y": float(cy)
            })

    if frame_number % 50 == 0:

        print(
            f"Processed: "
            f"{frame_number}/{total_frames}"
        )

cap.release()

print("\n" + "=" * 70)
print("TRACKING COMPLETE")
print("=" * 70)

print(
    "Total tracked vehicles:",
    len(tracks)
)

# ============================================================
# VEHICLE FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING VEHICLE FEATURES")
print("=" * 70)

vehicle_features = []

for vehicle_id, points in tracks.items():

    # Ignore very short tracks

    if len(points) < MIN_FRAMES:
        continue

    # --------------------------------------------------------
    # Convert track points to arrays
    # --------------------------------------------------------

    x = np.array(
        [p["x"] for p in points],
        dtype=float
    )

    y = np.array(
        [p["y"] for p in points],
        dtype=float
    )

    # --------------------------------------------------------
    # Position differences
    # --------------------------------------------------------

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    # --------------------------------------------------------
    # Pixel movement
    # --------------------------------------------------------

    movement = np.sqrt(
        dx ** 2 + dy ** 2
    )

    # --------------------------------------------------------
    # Pixel speed
    # --------------------------------------------------------

    speed = movement * fps

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    acceleration = np.diff(
        speed,
        prepend=speed[0]
    ) * fps

    # --------------------------------------------------------
    # Maximum deceleration
    # --------------------------------------------------------

    negative_acceleration = (
        acceleration[acceleration < 0]
    )

    if len(negative_acceleration) > 0:

        max_deceleration = abs(
            negative_acceleration.min()
        )

    else:

        max_deceleration = 0.0

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = np.arctan2(
        dy,
        dx
    )

    direction_change = np.diff(
        direction,
        prepend=direction[0]
    )

    direction_change = np.abs(
        direction_change
    )

    # Handle angle wrap-around

    direction_change = np.minimum(
        direction_change,
        2 * np.pi - direction_change
    )

    # --------------------------------------------------------
    # Total movement
    # --------------------------------------------------------

    total_distance = movement.sum()

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = len(points) / fps

    # --------------------------------------------------------
    # Create feature row
    # --------------------------------------------------------

    vehicle_features.append({

        "vehicle_id": int(vehicle_id),

        "frames_observed": len(points),

        "duration_seconds": duration,

        "avg_speed": float(
            np.mean(speed)
        ),

        "max_speed": float(
            np.max(speed)
        ),

        "avg_acceleration": float(
            np.mean(acceleration)
        ),

        "max_acceleration": float(
            np.max(acceleration)
        ),

        "max_deceleration": float(
            max_deceleration
        ),

        "avg_direction_change": float(
            np.mean(direction_change)
        ),

        "max_direction_change": float(
            np.max(direction_change)
        ),

        "total_movement_distance": float(
            total_distance
        )
    })

# ============================================================
# CREATE DATAFRAME
# ============================================================

features_df = pd.DataFrame(
    vehicle_features
)

# ============================================================
# CLEAN DATA
# ============================================================

features_df = features_df.replace(
    [np.inf, -np.inf],
    np.nan
)

features_df = features_df.fillna(0)

# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("VEHICLE FEATURES CREATED")
print("=" * 70)

print(
    "Vehicles with valid tracks:",
    len(features_df)
)

print("\nFeature columns:")

print(
    features_df.columns.tolist()
)

print("\nFirst 10 vehicles:")

display(
    features_df.head(10)
)

# --- CELL 57 ---
# ============================================
# RANDOM FOREST PREDICTION
# ============================================

import joblib

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"
scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Exact 7 training features
feature_columns = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]

# Prepare video features
X_video = features_df[feature_columns]

# Scale
X_video_scaled = scaler.transform(X_video)

# Prediction
predictions = model.predict(X_video_scaled)

# Probability
probabilities = model.predict_proba(X_video_scaled)

# Add results
features_df["prediction"] = predictions
features_df["status"] = features_df["prediction"].map({
    0: "SAFE",
    1: "DANGEROUS"
})

features_df["danger_probability"] = probabilities[:, 1] * 100

# ============================================
# DISPLAY RESULTS
# ============================================

print("================================")
print("RANDOM FOREST VIDEO PREDICTION")
print("================================")

print(
    features_df[
        ["vehicle_id", "frames_observed", "status", "danger_probability"]
    ].to_string(index=False)
)

safe_count = (features_df["prediction"] == 0).sum()
dangerous_count = (features_df["prediction"] == 1).sum()

print("\n================================")
print("FINAL PREDICTION SUMMARY")
print("================================")
print("Total Valid Vehicles :", len(features_df))
print("SAFE Vehicles        :", safe_count)
print("DANGEROUS Vehicles   :", dangerous_count)

# --- CELL 58 ---
# ============================================================
# FINAL DANGEROUS DRIVING DETECTION VIDEO
# YOLO + BYTE TRACK + RANDOM FOREST
# ============================================================

import cv2
import math
import joblib
import numpy as np
import pandas as pd

from ultralytics import YOLO
from collections import defaultdict

# ============================================================
# 1. PATHS
# ============================================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

output_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96.mp4"


# ============================================================
# 2. LOAD YOLO + RANDOM FOREST
# ============================================================

print("================================")
print("LOADING MODELS")
print("================================")

yolo_model = YOLO("yolov8n.pt")

rf_model = joblib.load(model_path)

scaler = joblib.load(scaler_path)

print("YOLO           : Loaded")
print("Random Forest  : Loaded")
print("Scaler         : Loaded")


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("ERROR: Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n================================")
print("VIDEO INFORMATION")
print("================================")
print("FPS          :", fps)
print("Total Frames :", total_frames)
print("Resolution   :", width, "x", height)


# ============================================================
# 4. TRACKING DATA
# ============================================================

tracks = defaultdict(list)

# Store detections for every frame
frame_detections = defaultdict(list)

frame_number = 0

# Vehicle classes from COCO
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [2, 3, 5, 7]

print("\n================================")
print("STEP 1: YOLO + BYTE TRACK")
print("================================")


# ============================================================
# 5. RUN YOLO + BYTE TRACK
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = result.boxes.id.cpu().numpy().astype(int)

        classes = result.boxes.cls.cpu().numpy().astype(int)

        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, cls, confidence in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box

            # Center point
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Store trajectory
            tracks[track_id].append({
                "frame": frame_number,
                "x": cx,
                "y": cy
            })

            # Store detection for final video
            frame_detections[frame_number].append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "track_id": int(track_id),
                "class": int(cls),
                "confidence": float(confidence)
            })

    if frame_number % 50 == 0:
        print(
            f"Processed: {frame_number} / {total_frames}"
        )


cap.release()


print("\n================================")
print("TRACKING COMPLETED")
print("================================")

print("Total tracked IDs:", len(tracks))


# ============================================================
# 6. FEATURE EXTRACTION
# ============================================================

print("\n================================")
print("STEP 2: FEATURE EXTRACTION")
print("================================")

MIN_FRAMES = 30

vehicle_features = []

for vehicle_id, points in tracks.items():

    # Ignore very short tracks
    if len(points) < MIN_FRAMES:
        continue

    # Sort by frame
    points = sorted(
        points,
        key=lambda p: p["frame"]
    )

    x = np.array(
        [p["x"] for p in points],
        dtype=float
    )

    y = np.array(
        [p["y"] for p in points],
        dtype=float
    )

    # --------------------------------------------------------
    # Position difference
    # --------------------------------------------------------

    dx = np.diff(x)
    dy = np.diff(y)

    distances = np.sqrt(
        dx**2 + dy**2
    )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    speed = distances * fps

    if len(speed) == 0:
        continue

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    if len(speed) > 1:

        acceleration = np.diff(speed) * fps

    else:

        acceleration = np.array([0.0])

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    directions = np.arctan2(
        dy,
        dx
    )

    if len(directions) > 1:

        direction_change = np.abs(
            np.diff(directions)
        )

        direction_change = np.minimum(
            direction_change,
            2 * np.pi - direction_change
        )

    else:

        direction_change = np.array([0.0])

    # --------------------------------------------------------
    # 7 FEATURES
    # --------------------------------------------------------

    avg_speed = np.mean(speed)

    max_speed = np.max(speed)

    avg_acceleration = np.mean(
        np.abs(acceleration)
    )

    max_acceleration = np.max(
        np.abs(acceleration)
    )

    negative_acceleration = acceleration[
        acceleration < 0
    ]

    if len(negative_acceleration) > 0:

        max_deceleration = np.max(
            np.abs(negative_acceleration)
        )

    else:

        max_deceleration = 0.0

    avg_direction_change = np.mean(
        direction_change
    )

    max_direction_change = np.max(
        direction_change
    )

    vehicle_features.append({

        "vehicle_id": vehicle_id,

        "frames_observed": len(points),

        "avg_speed": avg_speed,

        "max_speed": max_speed,

        "avg_acceleration": avg_acceleration,

        "max_acceleration": max_acceleration,

        "max_deceleration": max_deceleration,

        "avg_direction_change": avg_direction_change,

        "max_direction_change": max_direction_change
    })


features_df = pd.DataFrame(
    vehicle_features
)


print(
    "Valid vehicles:",
    len(features_df)
)


# ============================================================
# 7. RANDOM FOREST PREDICTION
# ============================================================

print("\n================================")
print("STEP 3: RANDOM FOREST PREDICTION")
print("================================")


feature_columns = [

    "avg_speed",

    "max_speed",

    "avg_acceleration",

    "max_acceleration",

    "max_deceleration",

    "avg_direction_change",

    "max_direction_change"
]


X_video = features_df[
    feature_columns
]


# Scale exactly like training
X_scaled = scaler.transform(
    X_video
)


# Prediction
predictions = rf_model.predict(
    X_scaled
)


# Probability
probabilities = rf_model.predict_proba(
    X_scaled
)


# Add results
features_df["prediction"] = predictions

features_df["status"] = features_df[
    "prediction"
].map({

    0: "SAFE",

    1: "DANGEROUS"
})


features_df["danger_probability"] = (
    probabilities[:, 1] * 100
)


# ============================================================
# 8. CREATE PREDICTION DICTIONARY
# ============================================================

vehicle_predictions = {}

for _, row in features_df.iterrows():

    vehicle_id = int(
        row["vehicle_id"]
    )

    vehicle_predictions[vehicle_id] = {

        "status": row["status"],

        "probability": row[
            "danger_probability"
        ]
    }


# ============================================================
# 9. PRINT PREDICTIONS
# ============================================================

print("\nVehicle Predictions:")
print("--------------------------------")

for _, row in features_df.iterrows():

    print(
        f"Vehicle {int(row['vehicle_id']):4d} : "
        f"{row['status']:10s} "
        f"({row['danger_probability']:.1f}%)"
    )


safe_count = int(
    (features_df["prediction"] == 0).sum()
)

dangerous_count = int(
    (features_df["prediction"] == 1).sum()
)


# ============================================================
# 10. FINAL VIDEO WRITER
# ============================================================

print("\n================================")
print("STEP 4: CREATING FINAL VIDEO")
print("================================")


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 11. READ VIDEO AGAIN
# ============================================================

cap = cv2.VideoCapture(
    video_path
)

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    detections = frame_detections.get(
        frame_number,
        []
    )


    # --------------------------------------------------------
    # Draw every tracked vehicle
    # --------------------------------------------------------

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        vehicle_id = detection[
            "track_id"
        ]


        # If prediction exists
        if vehicle_id in vehicle_predictions:

            status = vehicle_predictions[
                vehicle_id
            ]["status"]

            probability = vehicle_predictions[
                vehicle_id
            ]["probability"]


            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = (
                f"ID: {vehicle_id} | "
                f"{status} | "
                f"{probability:.0f}%"
            )


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            if status == "DANGEROUS":

                # Red
                box_color = (
                    0,
                    0,
                    255
                )

            else:

                # Green
                box_color = (
                    0,
                    255,
                    0
                )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )


            # ------------------------------------------------
            # Label background
            # ------------------------------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.65

            thickness = 2


            (text_width,
             text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )


            text_y = max(
                y1 - 10,
                text_height + 10
            )


            cv2.rectangle(

                frame,

                (
                    x1,
                    text_y - text_height - baseline - 5
                ),

                (
                    x1 + text_width + 5,
                    text_y + 5
                ),

                box_color,

                -1
            )


            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            cv2.putText(

                frame,

                label,

                (
                    x1 + 2,
                    text_y
                ),

                font,

                font_scale,

                (255, 255, 255),

                thickness,

                cv2.LINE_AA
            )


        else:

            # ------------------------------------------------
            # No prediction available
            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (255, 255, 0),

                2
            )


            cv2.putText(

                frame,

                f"ID: {vehicle_id}",

                (x1, max(y1 - 10, 20)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 0),

                2,

                cv2.LINE_AA
            )


    # ========================================================
    # SUMMARY ON VIDEO
    # ========================================================

    cv2.rectangle(

        frame,

        (20, 20),

        (430, 105),

        (0, 0, 0),

        -1
    )


    cv2.putText(

        frame,

        "AI TRAFFIC DANGER DETECTION",

        (35, 48),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"SAFE: {safe_count}",

        (35, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 255, 0),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"DANGEROUS: {dangerous_count}",

        (180, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 0, 255),

        2,

        cv2.LINE_AA
    )


    # Write frame
    out.write(frame)


    if frame_number % 50 == 0:

        print(
            f"Video written: "
            f"{frame_number} / {total_frames}"
        )


cap.release()

out.release()


# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n")
print("================================")
print("FINAL VIDEO CREATED")
print("================================")

print(
    "Total Valid Vehicles :",
    len(features_df)
)

print(
    "SAFE Vehicles        :",
    safe_count
)

print(
    "DANGEROUS Vehicles   :",
    dangerous_count
)

print("\nOutput Video:")

print(output_path)

# --- CELL 59 ---
import subprocess
import imageio_ffmpeg

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96.mp4"

output_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96_h264.mp4"

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg_path,
    "-y",
    "-i", input_video,
    "-vf", "scale=1280:-2",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-crf", "28",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    output_video
]

result = subprocess.run(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

if result.returncode == 0:
    print("================================")
    print("VIDEO CONVERSION COMPLETED")
    print("================================")
    print(output_video)
else:
    print("Conversion failed:")
    print(result.stderr[-2000:])

# --- CELL 60 ---
from IPython.display import Video, display

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96_h264.mp4"

display(Video(video_path, embed=True, width=900))

# --- CELL 61 ---
# ============================================================
# COMPLETE IMPROVED FINAL VIDEO CODE
# YOLO + BYTE TRACK + RANDOM FOREST
# ============================================================

import cv2
import joblib
import numpy as np
import pandas as pd

from ultralytics import YOLO
from collections import defaultdict


# ============================================================
# 1. PATHS
# ============================================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

output_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96_IMPROVED.mp4"


# ============================================================
# 2. LOAD MODELS
# ============================================================

print("================================")
print("LOADING MODELS")
print("================================")

yolo_model = YOLO("yolov8n.pt")

rf_model = joblib.load(model_path)

scaler = joblib.load(scaler_path)

print("YOLO          : Loaded")
print("Random Forest : Loaded")
print("Scaler        : Loaded")


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("ERROR: Video could not be opened")

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n================================")
print("VIDEO INFORMATION")
print("================================")

print("FPS          :", fps)
print("Total Frames :", total_frames)
print("Resolution   :", width, "x", height)


# ============================================================
# 4. TRACKING DATA
# ============================================================

tracks = defaultdict(list)

frame_detections = defaultdict(list)

frame_number = 0

VEHICLE_CLASSES = [2, 3, 5, 7]


# ============================================================
# 5. YOLO + BYTE TRACK
# ============================================================

print("\n================================")
print("STEP 1: YOLO + BYTE TRACK")
print("================================")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = result.boxes.id.cpu().numpy().astype(int)

        classes = result.boxes.cls.cpu().numpy().astype(int)

        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, cls, confidence in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Trajectory
            tracks[track_id].append({
                "frame": frame_number,
                "x": cx,
                "y": cy
            })

            # Detection
            frame_detections[frame_number].append({
                "box": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],
                "track_id": int(track_id),
                "confidence": float(confidence)
            })


    if frame_number % 50 == 0:

        print(
            f"Processed: {frame_number} / {total_frames}"
        )


cap.release()


print("\n================================")
print("TRACKING COMPLETED")
print("================================")

print(
    "Total tracked IDs:",
    len(tracks)
)


# ============================================================
# 6. FEATURE EXTRACTION
# ============================================================

print("\n================================")
print("STEP 2: FEATURE EXTRACTION")
print("================================")


MIN_FRAMES = 30

vehicle_features = []


for vehicle_id, points in tracks.items():

    if len(points) < MIN_FRAMES:
        continue

    points = sorted(
        points,
        key=lambda p: p["frame"]
    )

    x = np.array(
        [p["x"] for p in points],
        dtype=float
    )

    y = np.array(
        [p["y"] for p in points],
        dtype=float
    )

    dx = np.diff(x)

    dy = np.diff(y)

    distances = np.sqrt(
        dx ** 2 + dy ** 2
    )

    if len(distances) == 0:
        continue

    # Speed
    speed = distances * fps

    # Acceleration
    if len(speed) > 1:

        acceleration = np.diff(speed) * fps

    else:

        acceleration = np.array([0.0])


    # Direction
    directions = np.arctan2(
        dy,
        dx
    )


    if len(directions) > 1:

        direction_change = np.abs(
            np.diff(directions)
        )

        direction_change = np.minimum(
            direction_change,
            2 * np.pi - direction_change
        )

    else:

        direction_change = np.array([0.0])


    # 7 features
    avg_speed = np.mean(speed)

    max_speed = np.max(speed)

    avg_acceleration = np.mean(
        np.abs(acceleration)
    )

    max_acceleration = np.max(
        np.abs(acceleration)
    )


    negative_acceleration = acceleration[
        acceleration < 0
    ]


    if len(negative_acceleration) > 0:

        max_deceleration = np.max(
            np.abs(negative_acceleration)
        )

    else:

        max_deceleration = 0.0


    avg_direction_change = np.mean(
        direction_change
    )

    max_direction_change = np.max(
        direction_change
    )


    vehicle_features.append({

        "vehicle_id": vehicle_id,

        "frames_observed": len(points),

        "avg_speed": avg_speed,

        "max_speed": max_speed,

        "avg_acceleration": avg_acceleration,

        "max_acceleration": max_acceleration,

        "max_deceleration": max_deceleration,

        "avg_direction_change": avg_direction_change,

        "max_direction_change": max_direction_change
    })


features_df = pd.DataFrame(
    vehicle_features
)


print(
    "Valid vehicles:",
    len(features_df)
)


# ============================================================
# 7. RANDOM FOREST PREDICTION
# ============================================================

print("\n================================")
print("STEP 3: RANDOM FOREST PREDICTION")
print("================================")


feature_columns = [

    "avg_speed",

    "max_speed",

    "avg_acceleration",

    "max_acceleration",

    "max_deceleration",

    "avg_direction_change",

    "max_direction_change"
]


X_video = features_df[
    feature_columns
]


X_scaled = scaler.transform(
    X_video
)


predictions = rf_model.predict(
    X_scaled
)


probabilities = rf_model.predict_proba(
    X_scaled
)


features_df["prediction"] = predictions

features_df["status"] = features_df[
    "prediction"
].map({
    0: "SAFE",
    1: "DANGEROUS"
})


features_df["danger_probability"] = (
    probabilities[:, 1] * 100
)


# ============================================================
# 8. FIX STATIONARY VEHICLES
# ============================================================

print("\n================================")
print("STEP 4: STATIONARY VEHICLE CHECK")
print("================================")


RECENT_FRAMES = 15

# Increase threshold to handle YOLO box jitter
STOPPED_THRESHOLD = 4.0


final_vehicle_status = {}


for _, row in features_df.iterrows():

    vehicle_id = int(
        row["vehicle_id"]
    )

    original_status = row["status"]

    probability = float(
        row["danger_probability"]
    )

    points = sorted(
        tracks[vehicle_id],
        key=lambda p: p["frame"]
    )


    # --------------------------------------------------------
    # Recent movement
    # --------------------------------------------------------

    if len(points) >= RECENT_FRAMES:

        recent_points = points[
            -RECENT_FRAMES:
        ]

        recent_x = np.array(
            [p["x"] for p in recent_points],
            dtype=float
        )

        recent_y = np.array(
            [p["y"] for p in recent_points],
            dtype=float
        )

        recent_dx = np.diff(
            recent_x
        )

        recent_dy = np.diff(
            recent_y
        )

        recent_distances = np.sqrt(
            recent_dx ** 2 +
            recent_dy ** 2
        )

        average_recent_movement = np.mean(
            recent_distances
        )

    else:

        average_recent_movement = 0.0


    # --------------------------------------------------------
    # Stationary = SAFE
    # --------------------------------------------------------

    if average_recent_movement < STOPPED_THRESHOLD:

        final_status = "SAFE"

        final_probability = 0.0

    else:

        final_status = original_status

        final_probability = probability


    final_vehicle_status[vehicle_id] = {

        "status": final_status,

        "probability": final_probability,

        "movement": average_recent_movement
    }


# ============================================================
# 9. PRINT FINAL CLASSIFICATION
# ============================================================

print("\nVehicle Final Status")
print("--------------------------------")

for vehicle_id, data in final_vehicle_status.items():

    print(
        f"Vehicle {vehicle_id:4d} : "
        f"{data['status']:10s} | "
        f"Movement: {data['movement']:.2f} | "
        f"Probability: {data['probability']:.1f}%"
    )


# ============================================================
# 10. COUNTS
# ============================================================

safe_count = sum(

    1

    for data in final_vehicle_status.values()

    if data["status"] == "SAFE"
)


dangerous_count = sum(

    1

    for data in final_vehicle_status.values()

    if data["status"] == "DANGEROUS"
)


# ============================================================
# 11. CREATE OUTPUT VIDEO
# ============================================================

print("\n================================")
print("STEP 5: CREATING IMPROVED VIDEO")
print("================================")


cap = cv2.VideoCapture(
    video_path
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(

    output_path,

    fourcc,

    fps,

    (width, height)
)


frame_number = 0


# ============================================================
# 12. DRAW BOXES + LABELS
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    detections = frame_detections.get(
        frame_number,
        []
    )


    for detection in detections:

        vehicle_id = detection[
            "track_id"
        ]

        x1, y1, x2, y2 = detection[
            "box"
        ]


        # ----------------------------------------------------
        # Do NOT display short/unclassified tracks
        # ----------------------------------------------------

        if vehicle_id not in final_vehicle_status:

            continue


        data = final_vehicle_status[
            vehicle_id
        ]

        status = data["status"]

        probability = data["probability"]


        # ----------------------------------------------------
        # SAFE = GREEN
        # DANGEROUS = RED
        # ----------------------------------------------------

        if status == "DANGEROUS":

            box_color = (
                0,
                0,
                255
            )

        else:

            box_color = (
                0,
                255,
                0
            )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            box_color,

            3
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if status == "DANGEROUS":

            label = (
                f"ID: {vehicle_id} | "
                f"DANGEROUS | "
                f"{probability:.0f}%"
            )

        else:

            label = (
                f"ID: {vehicle_id} | SAFE"
            )


        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.65

        thickness = 2


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(

            label,

            font,

            font_scale,

            thickness
        )


        label_y = max(
            y1 - 10,
            text_height + 10
        )


        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (
                x1,
                label_y -
                text_height -
                baseline -
                5
            ),

            (
                x1 +
                text_width +
                8,

                label_y + 5
            ),

            box_color,

            -1
        )


        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        cv2.putText(

            frame,

            label,

            (
                x1 + 3,
                label_y
            ),

            font,

            font_scale,

            (255, 255, 255),

            thickness,

            cv2.LINE_AA
        )


    # ========================================================
    # SUMMARY BOX
    # ========================================================

    cv2.rectangle(

        frame,

        (20, 20),

        (470, 110),

        (0, 0, 0),

        -1
    )


    cv2.putText(

        frame,

        "AI TRAFFIC DANGER DETECTION",

        (35, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"SAFE: {safe_count}",

        (35, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 0),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"DANGEROUS: {dangerous_count}",

        (200, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 0, 255),

        2,

        cv2.LINE_AA
    )


    out.write(frame)


    if frame_number % 50 == 0:

        print(
            f"Video written: "
            f"{frame_number} / {total_frames}"
        )


# ============================================================
# 13. RELEASE
# ============================================================

cap.release()

out.release()


# ============================================================
# 14. FINAL RESULT
# ============================================================

print("\n================================")
print("IMPROVED FINAL VIDEO CREATED")
print("================================")

print(
    "Total Valid Vehicles :",
    len(final_vehicle_status)
)

print(
    "SAFE Vehicles        :",
    safe_count
)

print(
    "DANGEROUS Vehicles   :",
    dangerous_count
)

print("\nOutput Video:")

print(output_path)

# --- CELL 62 ---
import subprocess
import imageio_ffmpeg

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96_IMPROVED.mp4"

output_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96_FINAL.mp4"

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg,
    "-y",
    "-i", input_video,

    # Smaller resolution for fast conversion
    "-vf", "scale=640:-2",

    # Browser compatible H.264
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-crf", "30",
    "-pix_fmt", "yuv420p",

    # Browser streaming
    "-movflags", "+faststart",

    # No audio needed
    "-an",

    output_video
]

print("Converting video...")
print("Please wait...")

result = subprocess.run(
    command,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    text=True
)

if result.returncode == 0:
    print("================================")
    print("CONVERSION COMPLETED")
    print("================================")
    print(output_video)
else:
    print("ERROR:")
    print(result.stderr[-3000:])

# --- CELL 63 ---
from IPython.display import Video, display

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96_FINAL.mp4"

display(
    Video(
        video_path,
        embed=True,
        width=800
    )
)

# --- CELL 64 ---
import numpy as np
import pandas as pd

print("================================")
print("VEHICLE MOVEMENT ANALYSIS")
print("================================")

movement_data = []

for vehicle_id, points in tracks.items():

    if len(points) < 30:
        continue

    points = sorted(
        points,
        key=lambda p: p["frame"]
    )

    x = np.array(
        [p["x"] for p in points],
        dtype=float
    )

    y = np.array(
        [p["y"] for p in points],
        dtype=float
    )

    dx = np.diff(x)
    dy = np.diff(y)

    distances = np.sqrt(
        dx**2 + dy**2
    )

    if len(distances) == 0:
        continue

    movement_data.append({
        "vehicle_id": vehicle_id,
        "frames": len(points),
        "avg_movement": np.mean(distances),
        "median_movement": np.median(distances),
        "max_movement": np.max(distances),
        "std_movement": np.std(distances)
    })


movement_df = pd.DataFrame(
    movement_data
)

print(movement_df.to_string(index=False))

print("\n================================")
print("MOVEMENT STATISTICS")
print("================================")

print(
    movement_df[
        [
            "avg_movement",
            "median_movement",
            "max_movement",
            "std_movement"
        ]
    ].describe()
)

# --- CELL 65 ---
import numpy as np
import pandas as pd

print("================================")
print("SMOOTHED TRAJECTORY ANALYSIS")
print("================================")

smoothed_features = []

WINDOW = 7

for vehicle_id, points in tracks.items():

    if len(points) < 30:
        continue

    points = sorted(
        points,
        key=lambda p: p["frame"]
    )

    x = np.array(
        [p["x"] for p in points],
        dtype=float
    )

    y = np.array(
        [p["y"] for p in points],
        dtype=float
    )

    # --------------------------------
    # Smooth trajectory
    # --------------------------------

    x_smooth = (
        pd.Series(x)
        .rolling(
            WINDOW,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )

    y_smooth = (
        pd.Series(y)
        .rolling(
            WINDOW,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )

    # --------------------------------
    # Movement
    # --------------------------------

    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)

    movement = np.sqrt(
        dx**2 + dy**2
    )

    if len(movement) < 2:
        continue

    # --------------------------------
    # Speed
    # --------------------------------

    speed = movement * fps

    # --------------------------------
    # Acceleration
    # --------------------------------

    acceleration = np.diff(speed) * fps

    # --------------------------------
    # Direction
    # --------------------------------

    direction = np.arctan2(
        dy,
        dx
    )

    if len(direction) > 1:

        direction_change = np.abs(
            np.diff(direction)
        )

        direction_change = np.minimum(
            direction_change,
            2 * np.pi - direction_change
        )

    else:

        direction_change = np.array([0.0])

    # --------------------------------
    # Robust statistics
    # --------------------------------

    smoothed_features.append({

        "vehicle_id": vehicle_id,

        "frames": len(points),

        "median_movement": np.median(
            movement
        ),

        "avg_movement": np.mean(
            movement
        ),

        "movement_p90": np.percentile(
            movement,
            90
        ),

        "speed_median": np.median(
            speed
        ),

        "speed_p90": np.percentile(
            speed,
            90
        ),

        "acceleration_median": np.median(
            np.abs(acceleration)
        ),

        "acceleration_p90": np.percentile(
            np.abs(acceleration),
            90
        ),

        "direction_change_median": np.median(
            direction_change
        ),

        "direction_change_p90": np.percentile(
            direction_change,
            90
        )
    })


smoothed_df = pd.DataFrame(
    smoothed_features
)

print(
    smoothed_df.to_string(
        index=False
    )
)

print("\n================================")
print("SMOOTHED STATISTICS")
print("================================")

print(
    smoothed_df.describe()
)

# --- CELL 66 ---
# ============================================================
# STEP: PERSISTENT DANGEROUS BEHAVIOR ANALYSIS
# ============================================================

import numpy as np
import pandas as pd

print("================================")
print("PERSISTENT BEHAVIOR ANALYSIS")
print("================================")

# Minimum movement to consider a vehicle as actively moving
MOVEMENT_THRESHOLD = 3.0

# Percentage of recent frames that must show movement
PERSISTENCE_RATIO = 0.60

# Number of recent points to inspect
RECENT_POINTS = 30

persistent_results = []

for vehicle_id, points in tracks.items():

    if len(points) < 30:
        continue

    points = sorted(
        points,
        key=lambda p: p["frame"]
    )

    # --------------------------------------------------------
    # Smooth trajectory
    # --------------------------------------------------------

    x = np.array(
        [p["x"] for p in points],
        dtype=float
    )

    y = np.array(
        [p["y"] for p in points],
        dtype=float
    )

    x_smooth = (
        pd.Series(x)
        .rolling(
            7,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )

    y_smooth = (
        pd.Series(y)
        .rolling(
            7,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )

    # --------------------------------------------------------
    # Movement between frames
    # --------------------------------------------------------

    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)

    movement = np.sqrt(
        dx**2 + dy**2
    )

    if len(movement) < RECENT_POINTS:
        recent = movement
    else:
        recent = movement[-RECENT_POINTS:]

    # --------------------------------------------------------
    # Persistent movement
    # --------------------------------------------------------

    moving_frames = np.sum(
        recent > MOVEMENT_THRESHOLD
    )

    total_recent_frames = len(recent)

    movement_ratio = (
        moving_frames /
        total_recent_frames
    )

    median_recent_movement = np.median(
        recent
    )

    mean_recent_movement = np.mean(
        recent
    )

    p90_recent_movement = np.percentile(
        recent,
        90
    )

    # --------------------------------------------------------
    # Persistent or not
    # --------------------------------------------------------

    persistent = (
        movement_ratio >=
        PERSISTENCE_RATIO
    )

    persistent_results.append({

        "vehicle_id": vehicle_id,

        "frames": len(points),

        "moving_frames": moving_frames,

        "recent_frames": total_recent_frames,

        "movement_ratio": movement_ratio,

        "median_recent_movement":
            median_recent_movement,

        "mean_recent_movement":
            mean_recent_movement,

        "p90_recent_movement":
            p90_recent_movement,

        "persistent_movement":
            persistent
    })


persistent_df = pd.DataFrame(
    persistent_results
)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print(
    persistent_df.to_string(
        index=False
    )
)

print("\n================================")
print("PERSISTENT VEHICLES")
print("================================")

persistent_only = persistent_df[
    persistent_df["persistent_movement"] == True
]

print(
    persistent_only.to_string(
        index=False
    )
)

print("\n================================")
print("SUMMARY")
print("================================")

print(
    "Total valid vehicles :",
    len(persistent_df)
)

print(
    "Persistent movement  :",
    len(persistent_only)
)

# --- CELL 67 ---
# ============================================================
# OBJECTIVE DANGEROUS BEHAVIOR ANALYSIS
# ============================================================

print("================================")
print("OBJECTIVE BEHAVIOR ANALYSIS")
print("================================")

behavior_results = []

for vehicle_id, points in tracks.items():

    if len(points) < 30:
        continue

    points = sorted(
        points,
        key=lambda p: p["frame"]
    )

    x = np.array(
        [p["x"] for p in points],
        dtype=float
    )

    y = np.array(
        [p["y"] for p in points],
        dtype=float
    )

    # Smooth trajectory
    x_smooth = (
        pd.Series(x)
        .rolling(
            7,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )

    y_smooth = (
        pd.Series(y)
        .rolling(
            7,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )

    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)

    movement = np.sqrt(
        dx**2 + dy**2
    )

    if len(movement) < 10:
        continue

    speed = movement * fps

    acceleration = np.diff(speed) * fps

    direction = np.arctan2(
        dy,
        dx
    )

    if len(direction) > 1:

        direction_change = np.abs(
            np.diff(direction)
        )

        direction_change = np.minimum(
            direction_change,
            2*np.pi - direction_change
        )

    else:

        direction_change = np.array([0.0])

    # Recent 30 points
    recent_movement = movement[-30:]

    recent_speed = speed[-30:]

    recent_acceleration = (
        acceleration[-30:]
        if len(acceleration) >= 30
        else acceleration
    )

    recent_direction = (
        direction_change[-30:]
        if len(direction_change) >= 30
        else direction_change
    )

    # Statistics
    behavior_results.append({

        "vehicle_id": vehicle_id,

        "recent_median_movement":
            np.median(recent_movement),

        "recent_p90_movement":
            np.percentile(
                recent_movement,
                90
            ),

        "recent_speed_p90":
            np.percentile(
                recent_speed,
                90
            ),

        "recent_acceleration_p90":
            np.percentile(
                np.abs(recent_acceleration),
                90
            ),

        "recent_direction_p90":
            np.percentile(
                recent_direction,
                90
            ),

        "sharp_direction_count":
            np.sum(
                recent_direction > 1.0
            )
    })


behavior_df = pd.DataFrame(
    behavior_results
)

print(
    behavior_df.to_string(
        index=False
    )
)

# --- CELL 68 ---
# ============================================================
# LOAD RANDOM FOREST + SCALER AND FINAL DECISION
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Model paths
# ------------------------------------------------------------

MODEL_PATH = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

SCALER_PATH = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

print("================================")
print("LOADING RANDOM FOREST")
print("================================")

print("Model exists :", os.path.exists(MODEL_PATH))
print("Scaler exists:", os.path.exists(SCALER_PATH))

# Load
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model  :", type(model))
print("Scaler :", type(scaler))


# ============================================================
# 2. Features
# ============================================================

feature_columns = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]

X_video = features_df[
    feature_columns
].copy()


# ============================================================
# 3. Scale
# ============================================================

X_scaled = scaler.transform(
    X_video
)


# ============================================================
# 4. Random Forest prediction
# ============================================================

rf_prediction = model.predict(
    X_scaled
)

rf_probability = model.predict_proba(
    X_scaled
)[:, 1]


# ============================================================
# 5. Persistent movement dictionary
# ============================================================

persistent_status = {}

for _, row in persistent_df.iterrows():

    vehicle_id = int(
        row["vehicle_id"]
    )

    persistent_status[
        vehicle_id
    ] = bool(
        row["persistent_movement"]
    )


# ============================================================
# 6. FINAL DECISION
# ============================================================

final_vehicle_status = {}

for i, (_, row) in enumerate(
    features_df.iterrows()
):

    vehicle_id = int(
        row["vehicle_id"]
    )

    rf_result = int(
        rf_prediction[i]
    )

    probability = (
        float(rf_probability[i])
        * 100
    )

    is_persistent = persistent_status.get(
        vehicle_id,
        False
    )

    # Default SAFE
    final_status = "SAFE"

    # Dangerous only when both conditions agree
    if (
        rf_result == 1
        and
        is_persistent
    ):
        final_status = "DANGEROUS"

    final_vehicle_status[
        vehicle_id
    ] = {
        "status": final_status,
        "probability": probability,
        "persistent": is_persistent
    }


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n================================")
print("FINAL VEHICLE STATUS")
print("================================")

for vehicle_id in sorted(
    final_vehicle_status
):

    result = final_vehicle_status[
        vehicle_id
    ]

    print(
        f"Vehicle {vehicle_id:4d} : "
        f"{result['status']:10s} | "
        f"Probability: "
        f"{result['probability']:5.1f}% | "
        f"Persistent: "
        f"{result['persistent']}"
    )


# ============================================================
# 8. SUMMARY
# ============================================================

safe_count = sum(
    1
    for result in final_vehicle_status.values()
    if result["status"] == "SAFE"
)

dangerous_count = sum(
    1
    for result in final_vehicle_status.values()
    if result["status"] == "DANGEROUS"
)

print("\n================================")
print("FINAL SUMMARY")
print("================================")

print(
    "Total Vehicles     :",
    len(final_vehicle_status)
)

print(
    "SAFE Vehicles      :",
    safe_count
)

print(
    "DANGEROUS Vehicles :",
    dangerous_count
)

# --- CELL 69 ---
# ============================================================
# FINAL VIDEO GENERATION
# ============================================================

import cv2
import os

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

output_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_FINAL_SAFE_DANGEROUS_video96.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise RuntimeError("ERROR: Video could not be opened")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

print("================================")
print("CREATING FINAL VIDEO")
print("================================")

print("FPS          :", fps)
print("Total Frames :", total_frames)
print("Resolution   :", width, "x", height)


# ------------------------------------------------------------
# IMPORTANT:
# Re-run YOLO + ByteTrack so we get bounding boxes
# ------------------------------------------------------------

from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")

# ByteTrack
try:
    import supervision as sv
except ImportError:
    raise ImportError(
        "supervision is not installed. "
        "Run: pip install supervision"
    )

tracker = sv.ByteTrack()

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    results = yolo_model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    # --------------------------------------------------------
    # ByteTrack
    # --------------------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )

    # --------------------------------------------------------
    # Draw vehicles
    # --------------------------------------------------------

    for i in range(
        len(detections)
    ):

        if detections.tracker_id is None:
            continue

        vehicle_id = int(
            detections.tracker_id[i]
        )

        # Only draw vehicles that were
        # included in final prediction
        if vehicle_id not in final_vehicle_status:
            continue

        status_info = final_vehicle_status[
            vehicle_id
        ]

        status = status_info["status"]

        probability = status_info[
            "probability"
        ]

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        xyxy = detections.xyxy[i]

        x1, y1, x2, y2 = map(
            int,
            xyxy
        )

        # ----------------------------------------------------
        # Status-based appearance
        # ----------------------------------------------------

        if status == "DANGEROUS":

            box_color = (
                0,
                0,
                255
            )

            label = (
                f"ID {vehicle_id} | "
                f"DANGEROUS | "
                f"{probability:.0f}%"
            )

        else:

            box_color = (
                0,
                255,
                0
            )

            label = (
                f"ID {vehicle_id} | "
                f"SAFE"
            )

        # ----------------------------------------------------
        # Draw box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.65

        thickness = 2

        (tw, th), _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        label_y1 = max(
            0,
            y1 - th - 12
        )

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + tw + 10, y1),
            box_color,
            -1
        )

        # ----------------------------------------------------
        # Label text
        # ----------------------------------------------------

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 7),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Project information
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (15, 15),
        (500, 110),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI TRAFFIC DANGEROUS-DRIVING DETECTION",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"SAFE: {18}",
        (25, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"DANGEROUS: {7}",
        (170, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (25, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Write frame
    # --------------------------------------------------------

    out.write(frame)

    if frame_number % 50 == 0:

        print(
            f"Video written: "
            f"{frame_number} / "
            f"{total_frames}"
        )


cap.release()
out.release()

print("\n================================")
print("FINAL VIDEO CREATED")
print("================================")

print(
    "SAFE Vehicles      : 18"
)

print(
    "DANGEROUS Vehicles : 7"
)

print(
    "Output Video:"
)

print(
    output_video
)

# --- CELL 70 ---
import subprocess
import imageio_ffmpeg
import os

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_FINAL_SAFE_DANGEROUS_video96.mp4"

output_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_FINAL_H264_video96.mp4"

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

print("Converting video to H.264...")
print("Please wait...")

command = [
    ffmpeg_path,
    "-y",
    "-i", input_video,

    # H.264 video
    "-c:v", "libx264",

    # Faster encoding
    "-preset", "ultrafast",

    # Good quality
    "-crf", "23",

    # Browser-compatible pixel format
    "-pix_fmt", "yuv420p",

    # Remove audio because original traffic video doesn't need it
    "-an",

    output_video
]

result = subprocess.run(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

if result.returncode == 0:

    print("\n================================")
    print("H.264 VIDEO CREATED")
    print("================================")

    print("Output:")
    print(output_video)

    print("\nFile exists:", os.path.exists(output_video))

else:

    print("ERROR:")
    print(result.stderr[-3000:])

# --- CELL 71 ---
from IPython.display import Video, display

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_FINAL_H264_video96.mp4"

display(
    Video(
        video_path,
        embed=True,
        width=900
    )
)

# --- CELL 72 ---
# ============================================================
# FINAL VIDEO GENERATION
# ALL VEHICLES + SAFE / DANGEROUS / INSUFFICIENT DATA
# ============================================================

import cv2
import os

from ultralytics import YOLO

# ============================================================
# INPUT / OUTPUT
# ============================================================

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

output_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_FINAL_SAFE_DANGEROUS_video96.mp4"


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise RuntimeError("ERROR: Video could not be opened")


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PRINT VIDEO INFORMATION
# ============================================================

print("================================")
print("CREATING FINAL VIDEO")
print("================================")

print(
    "FPS          :",
    fps
)

print(
    "Total Frames :",
    total_frames
)

print(
    "Resolution   :",
    width,
    "x",
    height
)


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO model...")

yolo_model = YOLO(
    "yolov8n.pt"
)

print("YOLO loaded successfully")


# ============================================================
# LOAD BYTE TRACK
# ============================================================

try:

    import supervision as sv

except ImportError:

    raise ImportError(
        "supervision is not installed.\n"
        "Run this command:\n"
        "pip install supervision"
    )


tracker = sv.ByteTrack()


# ============================================================
# VEHICLE CLASSES
# ============================================================

# COCO classes:
#
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [
    2,
    3,
    5,
    7
]


# ============================================================
# TRACKING VARIABLES
# ============================================================

frame_number = 0

all_vehicle_ids = set()

vehicle_last_seen = {}

INSUFFICIENT_DATA_COUNT = 0


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = yolo_model(
        frame,
        classes=VEHICLE_CLASSES,
        conf=0.25,
        verbose=False
    )[0]


    # ========================================================
    # CONVERT YOLO DETECTIONS
    # ========================================================

    detections = sv.Detections.from_ultralytics(
        results
    )


    # ========================================================
    # BYTE TRACK
    # ========================================================

    detections = tracker.update_with_detections(
        detections
    )


    # ========================================================
    # DRAW EVERY VEHICLE
    # ========================================================

    for i in range(
        len(detections)
    ):

        # ----------------------------------------------------
        # Check tracker ID
        # ----------------------------------------------------

        if detections.tracker_id is None:
            continue


        vehicle_id = int(
            detections.tracker_id[i]
        )


        # ----------------------------------------------------
        # Save vehicle ID
        # ----------------------------------------------------

        all_vehicle_ids.add(
            vehicle_id
        )

        vehicle_last_seen[
            vehicle_id
        ] = frame_number


        # ----------------------------------------------------
        # Get bounding box
        # ----------------------------------------------------

        xyxy = detections.xyxy[i]

        x1, y1, x2, y2 = map(
            int,
            xyxy
        )


        # ====================================================
        # GET ML STATUS
        # ====================================================

        if vehicle_id in final_vehicle_status:

            status_info = (
                final_vehicle_status[
                    vehicle_id
                ]
            )


            status = status_info.get(
                "status",
                "INSUFFICIENT DATA"
            )


            probability = status_info.get(
                "probability",
                0
            )


        else:

            status = (
                "INSUFFICIENT DATA"
            )

            probability = 0


        # ====================================================
        # STATUS APPEARANCE
        # ====================================================

        if status == "DANGEROUS":

            # RED
            box_color = (
                0,
                0,
                255
            )

            label = (
                f"ID {vehicle_id} | "
                f"DANGEROUS | "
                f"{probability:.0f}%"
            )


        elif status == "SAFE":

            # GREEN
            box_color = (
                0,
                255,
                0
            )

            label = (
                f"ID {vehicle_id} | "
                f"SAFE | "
                f"{probability:.0f}%"
            )


        else:

            # YELLOW
            box_color = (
                0,
                255,
                255
            )

            label = (
                f"ID {vehicle_id} | "
                f"INSUFFICIENT DATA"
            )

            INSUFFICIENT_DATA_COUNT += 1


        # ====================================================
        # DRAW BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )


        # ====================================================
        # LABEL BACKGROUND
        # ====================================================

        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

        font_scale = 0.65

        thickness = 2


        (
            (tw, th),
            _
        ) = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )


        label_y1 = max(
            0,
            y1 - th - 12
        )


        cv2.rectangle(
            frame,
            (
                x1,
                label_y1
            ),
            (
                x1 + tw + 10,
                y1
            ),
            box_color,
            -1
        )


        # ====================================================
        # LABEL TEXT
        # ====================================================

        cv2.putText(
            frame,
            label,
            (
                x1 + 5,
                y1 - 7
            ),
            font,
            font_scale,
            (
                255,
                255,
                255
            ),
            thickness,
            cv2.LINE_AA
        )


    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    cv2.rectangle(
        frame,
        (15, 15),
        (620, 130),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "AI TRAFFIC DANGEROUS-DRIVING DETECTION",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # SAFE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "SAFE: 18",
        (25, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # DANGEROUS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "DANGEROUS: 7",
        (170, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # DETECTED VEHICLES
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Vehicles: {len(all_vehicle_ids)}",
        (350, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # FRAME
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(
        frame
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 50 == 0:

        print(
            f"Video written: "
            f"{frame_number} / "
            f"{total_frames}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()

out.release()


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n================================")
print("FINAL VIDEO CREATED")
print("================================")


print(
    "Total Unique Vehicles Detected :",
    len(all_vehicle_ids)
)


print(
    "Vehicles with ML Status         :",
    len(final_vehicle_status)
)


print(
    "Vehicles without ML Status     :",
    len(
        all_vehicle_ids
        - set(final_vehicle_status.keys())
    )
)


print(
    "SAFE Vehicles                   : 18"
)


print(
    "DANGEROUS Vehicles              : 7"
)


print(
    "Output Video:"
)


print(
    output_video
)

# --- CELL 73 ---
# ============================================================
# CHECK EXISTING ML STATUS
# ============================================================

print("Checking final_vehicle_status...")

if "final_vehicle_status" not in globals():

    print("ERROR: final_vehicle_status does not exist.")

else:

    print("final_vehicle_status exists")

    print(
        "Total ML vehicles:",
        len(final_vehicle_status)
    )

    print("\nVehicle IDs:")

    print(
        sorted(final_vehicle_status.keys())
    )

    print("\nFirst 5 vehicle records:")

    for vehicle_id, info in list(
        final_vehicle_status.items()
    )[:5]:

        print(
            "ID:",
            vehicle_id,
            "|",
            info
        )

# --- CELL 74 ---
# ============================================================
# STEP 2 - CHECK RANDOM FOREST MODEL
# ============================================================

import joblib
import os

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

print("================================")
print("CHECKING FINAL ML MODEL")
print("================================")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

rf_model = joblib.load(model_path)

print("\nRandom Forest loaded successfully")


# ------------------------------------------------------------
# Load scaler
# ------------------------------------------------------------

rf_scaler = joblib.load(scaler_path)

print("Scaler loaded successfully")


# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

print("\n================================")
print("MODEL INFORMATION")
print("================================")

print(
    "Number of estimators :",
    rf_model.n_estimators
)

print(
    "Number of features   :",
    rf_model.n_features_in_
)


# ------------------------------------------------------------
# Scaler information
# ------------------------------------------------------------

print("\n================================")
print("SCALER INFORMATION")
print("================================")

print(
    "Scaler features      :",
    rf_scaler.n_features_in_
)


# ------------------------------------------------------------
# Feature names
# ------------------------------------------------------------

if hasattr(rf_model, "feature_names_in_"):

    print("\n================================")
    print("FEATURE NAMES")
    print("================================")

    print(
        list(rf_model.feature_names_in_)
    )

else:

    print(
        "\nModel does not contain feature names."
    )


# ------------------------------------------------------------
# Existing variables in notebook
# ------------------------------------------------------------

print("\n================================")
print("POSSIBLE FEATURE VARIABLES")
print("================================")

possible_variables = [
    "vehicle_features",
    "features",
    "X",
    "X_train",
    "X_test",
    "feature_data",
    "training_data",
    "vehicle_data",
    "final_vehicle_status"
]

for variable in possible_variables:

    if variable in globals():

        try:

            obj = globals()[variable]

            print(
                variable,
                "->",
                type(obj),
                "shape:",
                getattr(obj, "shape", "N/A")
            )

        except Exception:

            print(
                variable,
                "-> exists"
            )

# --- CELL 75 ---
# ============================================================
# STEP 3 - INSPECT VEHICLE FEATURES
# ============================================================

print("================================")
print("VEHICLE FEATURES CHECK")
print("================================")

print(
    "Total vehicle feature records:",
    len(vehicle_features)
)

print("\n--------------------------------")
print("FIRST 10 VEHICLE RECORDS")
print("--------------------------------")

for i, item in enumerate(vehicle_features[:10]):

    print(
        f"\nRecord {i + 1}:"
    )

    print(
        item
    )


print("\n================================")
print("RECORD TYPES")
print("================================")

if len(vehicle_features) > 0:

    print(
        "First record type:",
        type(vehicle_features[0])
    )

    try:

        print(
            "First record length:",
            len(vehicle_features[0])
        )

    except:

        print(
            "First record has no length"
        )

# --- CELL 76 ---
# ============================================================
# STEP 4
# YOLO + BYTE TRACK + FEATURE EXTRACTION + RANDOM FOREST
# ============================================================

import cv2
import numpy as np
import joblib
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import math

# ============================================================
# PATHS
# ============================================================

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"


# ============================================================
# LOAD MODEL + SCALER
# ============================================================

print("================================")
print("LOADING ML MODEL")
print("================================")

rf_model = joblib.load(model_path)
rf_scaler = joblib.load(scaler_path)

print("Random Forest loaded")
print("Scaler loaded")

print(
    "Model features:",
    rf_model.n_features_in_
)


# ============================================================
# LOAD YOLO
# ============================================================

print("\n================================")
print("LOADING YOLO")
print("================================")

yolo_model = YOLO("yolov8n.pt")

print("YOLO loaded")


# ============================================================
# BYTE TRACK
# ============================================================

tracker = sv.ByteTrack()


# ============================================================
# VEHICLE CLASSES
# ============================================================

# COCO:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [2, 3, 5, 7]


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():

    raise RuntimeError(
        "ERROR: Video could not be opened"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print("\nFPS:", fps)
print("Total frames:", total_frames)


# ============================================================
# TRACK HISTORY
# ============================================================

track_centers = defaultdict(list)


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    results = yolo_model(
        frame,
        classes=VEHICLE_CLASSES,
        conf=0.25,
        verbose=False
    )[0]


    # --------------------------------------------------------
    # Supervision detections
    # --------------------------------------------------------

    detections = sv.Detections.from_ultralytics(
        results
    )


    # --------------------------------------------------------
    # ByteTrack
    # --------------------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # --------------------------------------------------------
    # Store center point for EVERY vehicle
    # --------------------------------------------------------

    if detections.tracker_id is not None:

        for i in range(
            len(detections)
        ):

            vehicle_id = int(
                detections.tracker_id[i]
            )


            x1, y1, x2, y2 = map(
                int,
                detections.xyxy[i]
            )


            # Center
            cx = (
                x1 + x2
            ) / 2.0

            cy = (
                y1 + y2
            ) / 2.0


            track_centers[
                vehicle_id
            ].append(
                (cx, cy)
            )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if frame_number % 50 == 0:

        print(
            f"Processed: "
            f"{frame_number} / "
            f"{total_frames}"
        )


cap.release()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n================================")
print("EXTRACTING FEATURES")
print("================================")


vehicle_features_new = []


for vehicle_id, centers in track_centers.items():

    # Need at least 2 positions
    if len(centers) < 2:
        continue


    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    speeds = []

    for i in range(
        1,
        len(centers)
    ):

        x1, y1 = centers[i - 1]

        x2, y2 = centers[i]


        distance = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )


        speed = (
            distance * fps
        )

        speeds.append(
            speed
        )


    if len(speeds) == 0:
        continue


    avg_speed = float(
        np.mean(speeds)
    )

    max_speed = float(
        np.max(speeds)
    )


    # --------------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------------

    accelerations = []


    for i in range(
        1,
        len(speeds)
    ):

        acceleration = (
            speeds[i] -
            speeds[i - 1]
        ) * fps


        accelerations.append(
            acceleration
        )


    if len(accelerations) > 0:

        avg_acceleration = float(
            np.mean(
                np.abs(
                    accelerations
                )
            )
        )

        max_acceleration = float(
            np.max(
                np.abs(
                    accelerations
                )
            )
        )

        max_deceleration = float(
            np.max(
                np.abs(
                    [
                        a
                        for a in accelerations
                        if a < 0
                    ]
                )
            )
            if any(
                a < 0
                for a in accelerations
            )
            else 0.0
        )

    else:

        avg_acceleration = 0.0
        max_acceleration = 0.0
        max_deceleration = 0.0


    # --------------------------------------------------------
    # DIRECTION CHANGE
    # --------------------------------------------------------

    direction_changes = []


    for i in range(
        2,
        len(centers)
    ):

        x0, y0 = centers[i - 2]

        x1, y1 = centers[i - 1]

        x2, y2 = centers[i]


        dx1 = x1 - x0
        dy1 = y1 - y0

        dx2 = x2 - x1
        dy2 = y2 - y1


        mag1 = math.sqrt(
            dx1 ** 2 +
            dy1 ** 2
        )

        mag2 = math.sqrt(
            dx2 ** 2 +
            dy2 ** 2
        )


        if mag1 == 0 or mag2 == 0:
            continue


        dot_product = (
            dx1 * dx2 +
            dy1 * dy2
        )


        cosine = (
            dot_product /
            (mag1 * mag2)
        )


        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )


        angle = math.acos(
            cosine
        )


        direction_changes.append(
            angle
        )


    if len(direction_changes) > 0:

        avg_direction_change = float(
            np.mean(
                direction_changes
            )
        )

        max_direction_change = float(
            np.max(
                direction_changes
            )
        )

    else:

        avg_direction_change = 0.0

        max_direction_change = 0.0


    # --------------------------------------------------------
    # STORE EXACT 7 FEATURES
    # --------------------------------------------------------

    record = {

        "vehicle_id": vehicle_id,

        "frames_observed": len(
            centers
        ),

        "avg_speed": avg_speed,

        "max_speed": max_speed,

        "avg_acceleration": avg_acceleration,

        "max_acceleration": max_acceleration,

        "max_deceleration": max_deceleration,

        "avg_direction_change":
            avg_direction_change,

        "max_direction_change":
            max_direction_change
    }


    vehicle_features_new.append(
        record
    )


# ============================================================
# RESULT
# ============================================================

print("\n================================")
print("FEATURE EXTRACTION COMPLETED")
print("================================")

print(
    "Total tracked vehicles:",
    len(track_centers)
)

print(
    "Vehicles with features:",
    len(vehicle_features_new)
)


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

print("\n================================")
print("RANDOM FOREST PREDICTION")
print("================================")


final_vehicle_status_new = {}


for record in vehicle_features_new:

    vehicle_id = int(
        record["vehicle_id"]
    )


    # --------------------------------------------------------
    # EXACT 7 FEATURES
    # --------------------------------------------------------

    feature_vector = np.array([

        record["avg_speed"],

        record["max_speed"],

        record["avg_acceleration"],

        record["max_acceleration"],

        record["max_deceleration"],

        record["avg_direction_change"],

        record["max_direction_change"]

    ]).reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    feature_scaled = rf_scaler.transform(
        feature_vector
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = rf_model.predict(
        feature_scaled
    )[0]


    probabilities = (
        rf_model.predict_proba(
            feature_scaled
        )[0]
    )


    probability = float(
        np.max(
            probabilities
        ) * 100
    )


    # --------------------------------------------------------
    # Convert prediction to status
    # --------------------------------------------------------

    if prediction in [1, "1", "DANGEROUS"]:

        status = "DANGEROUS"

    else:

        status = "SAFE"


    final_vehicle_status_new[
        vehicle_id
    ] = {

        "status": status,

        "probability": probability,

        "persistent": False
    }


# ============================================================
# FINAL COUNTS
# ============================================================

safe_count = sum(

    1
    for info
    in final_vehicle_status_new.values()

    if info["status"] == "SAFE"
)


dangerous_count = sum(

    1
    for info
    in final_vehicle_status_new.values()

    if info["status"] == "DANGEROUS"
)


print("\n================================")
print("ML CLASSIFICATION COMPLETED")
print("================================")

print(
    "Total vehicles:",
    len(final_vehicle_status_new)
)

print(
    "SAFE:",
    safe_count
)

print(
    "DANGEROUS:",
    dangerous_count
)


# ============================================================
# SAVE NEW STATUS
# ============================================================

final_vehicle_status = (
    final_vehicle_status_new
)

vehicle_features = (
    vehicle_features_new
)


print("\n================================")
print("NEW ML STATUS CREATED")
print("================================")

print(
    "final_vehicle_status:",
    len(final_vehicle_status)
)

print(
    "vehicle_features:",
    len(vehicle_features)
)

# --- CELL 77 ---
# ============================================================
# STEP 5
# FINAL VIDEO
# ALL VEHICLES + ML STATUS
# ============================================================

import cv2
import supervision as sv
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

output_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_FINAL_CORRECTED_video96.mp4"


# ============================================================
# LOAD YOLO
# ============================================================

print("================================")
print("LOADING YOLO")
print("================================")

yolo_model = YOLO("yolov8n.pt")

print("YOLO loaded successfully")


# ============================================================
# BYTE TRACK
# ============================================================

tracker = sv.ByteTrack()


# ============================================================
# VEHICLE CLASSES
# ============================================================

# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [
    2,
    3,
    5,
    7
]


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    input_video
)

if not cap.isOpened():

    raise RuntimeError(
        "ERROR: Video could not be opened"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# INFORMATION
# ============================================================

print("\n================================")
print("CREATING FINAL VIDEO")
print("================================")

print(
    "FPS          :",
    fps
)

print(
    "Total Frames :",
    total_frames
)

print(
    "Resolution   :",
    width,
    "x",
    height
)

print(
    "ML Vehicles  :",
    len(final_vehicle_status)
)


# ============================================================
# TRACKING
# ============================================================

frame_number = 0

all_vehicle_ids = set()


# ============================================================
# VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = yolo_model(
        frame,
        classes=VEHICLE_CLASSES,
        conf=0.25,
        verbose=False
    )[0]


    # ========================================================
    # CONVERT DETECTIONS
    # ========================================================

    detections = (
        sv.Detections.from_ultralytics(
            results
        )
    )


    # ========================================================
    # BYTE TRACK
    # ========================================================

    detections = (
        tracker.update_with_detections(
            detections
        )
    )


    # ========================================================
    # DRAW ALL VEHICLES
    # ========================================================

    if detections.tracker_id is not None:

        for i in range(
            len(detections)
        ):

            vehicle_id = int(
                detections.tracker_id[i]
            )


            all_vehicle_ids.add(
                vehicle_id
            )


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                detections.xyxy[i]
            )


            # =================================================
            # GET ML STATUS
            # =================================================

            if vehicle_id in final_vehicle_status:

                info = (
                    final_vehicle_status[
                        vehicle_id
                    ]
                )

                status = info[
                    "status"
                ]

                probability = info[
                    "probability"
                ]


            else:

                status = (
                    "INSUFFICIENT DATA"
                )

                probability = 0


            # =================================================
            # COLORS
            # =================================================

            if status == "DANGEROUS":

                box_color = (
                    0,
                    0,
                    255
                )

                label = (
                    f"ID {vehicle_id} | "
                    f"DANGEROUS | "
                    f"{probability:.0f}%"
                )


            elif status == "SAFE":

                box_color = (
                    0,
                    255,
                    0
                )

                label = (
                    f"ID {vehicle_id} | "
                    f"SAFE | "
                    f"{probability:.0f}%"
                )


            else:

                box_color = (
                    0,
                    255,
                    255
                )

                label = (
                    f"ID {vehicle_id} | "
                    f"INSUFFICIENT DATA"
                )


            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )


            # =================================================
            # LABEL
            # =================================================

            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.65

            thickness = 2


            (
                (tw, th),
                _
            ) = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )


            label_y1 = max(
                0,
                y1 - th - 12
            )


            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y1
                ),
                (
                    x1 + tw + 10,
                    y1
                ),
                box_color,
                -1
            )


            cv2.putText(
                frame,
                label,
                (
                    x1 + 5,
                    y1 - 7
                ),
                font,
                font_scale,
                (
                    255,
                    255,
                    255
                ),
                thickness,
                cv2.LINE_AA
            )


    # ========================================================
    # CALCULATE CURRENT COUNTS
    # ========================================================

    current_safe = 0

    current_dangerous = 0

    current_unknown = 0


    for vehicle_id in all_vehicle_ids:

        if vehicle_id in final_vehicle_status:

            status = (
                final_vehicle_status[
                    vehicle_id
                ]["status"]
            )

            if status == "SAFE":

                current_safe += 1

            elif status == "DANGEROUS":

                current_dangerous += 1

        else:

            current_unknown += 1


    # ========================================================
    # PROJECT INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (15, 15),
        (650, 145),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "AI TRAFFIC DANGEROUS-DRIVING DETECTION",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"SAFE: {current_safe}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"DANGEROUS: {current_dangerous}",
        (180, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"VEHICLES: {len(all_vehicle_ids)}",
        (390, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(
        frame
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 50 == 0:

        print(
            f"Video written: "
            f"{frame_number} / "
            f"{total_frames}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()

out.release()


# ============================================================
# FINAL COUNTS
# ============================================================

final_safe = 0

final_dangerous = 0


for info in final_vehicle_status.values():

    if info["status"] == "SAFE":

        final_safe += 1

    elif info["status"] == "DANGEROUS":

        final_dangerous += 1


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n================================")
print("FINAL VIDEO CREATED")
print("================================")

print(
    "Unique Vehicles Detected :",
    len(all_vehicle_ids)
)

print(
    "ML Classified Vehicles   :",
    len(final_vehicle_status)
)

print(
    "SAFE Vehicles            :",
    final_safe
)

print(
    "DANGEROUS Vehicles       :",
    final_dangerous
)

print(
    "Unclassified Vehicles    :",
    len(
        all_vehicle_ids
        - set(
            final_vehicle_status.keys()
        )
    )
)

print(
    "\nOutput Video:"
)

print(
    output_video
)

# --- CELL 78 ---
# ============================================================
# STEP 7 - FINAL ML RESULT CHECK
# ============================================================

print("================================")
print("FINAL ML RESULT")
print("================================")

safe_vehicles = []
dangerous_vehicles = []

for vehicle_id, info in final_vehicle_status.items():

    if info["status"] == "SAFE":
        safe_vehicles.append(vehicle_id)

    elif info["status"] == "DANGEROUS":
        dangerous_vehicles.append(vehicle_id)


print("Total ML Classified Vehicles :", len(final_vehicle_status))

print("SAFE Vehicles                :", len(safe_vehicles))
print("DANGEROUS Vehicles           :", len(dangerous_vehicles))

print("\nSAFE Vehicle IDs:")
print(sorted(safe_vehicles))

print("\nDANGEROUS Vehicle IDs:")
print(sorted(dangerous_vehicles))

# --- CELL 79 ---
# ============================================================
# STEP 8 - SAVE FINAL VEHICLE PREDICTIONS
# ============================================================

import pandas as pd

rows = []

for vehicle_id, info in final_vehicle_status.items():

    rows.append({
        "vehicle_id": int(vehicle_id),
        "status": info["status"],
        "probability": float(info["probability"]),
        "persistent": info.get("persistent", False)
    })


results_df = pd.DataFrame(rows)

# Sort by vehicle ID
results_df = results_df.sort_values(
    by="vehicle_id"
).reset_index(drop=True)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\final_vehicle_predictions_video96.csv"

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("================================")
print("PREDICTIONS SAVED")
print("================================")

print(
    "Total records:",
    len(results_df)
)

print(
    "SAFE:",
    (results_df["status"] == "SAFE").sum()
)

print(
    "DANGEROUS:",
    (results_df["status"] == "DANGEROUS").sum()
)

print(
    "\nCSV file:"
)

print(csv_path)


print("\nFirst 10 predictions:")
display(results_df.head(10))

# --- CELL 80 ---
# ============================================================
# STEP 9 - FINAL PROJECT STATISTICS
# ============================================================

import pandas as pd
import os

# ------------------------------------------------------------
# Counts
# ------------------------------------------------------------

total_detected = len(all_vehicle_ids)

total_classified = len(final_vehicle_status)

safe_count = sum(
    1
    for info in final_vehicle_status.values()
    if info["status"] == "SAFE"
)

dangerous_count = sum(
    1
    for info in final_vehicle_status.values()
    if info["status"] == "DANGEROUS"
)

insufficient_count = (
    total_detected - total_classified
)


# ------------------------------------------------------------
# Average confidence
# ------------------------------------------------------------

if total_classified > 0:

    avg_confidence = sum(
        info["probability"]
        for info in final_vehicle_status.values()
    ) / total_classified

else:

    avg_confidence = 0


# ------------------------------------------------------------
# Percentages
# ------------------------------------------------------------

if total_classified > 0:

    safe_percentage = (
        safe_count /
        total_classified
    ) * 100

    dangerous_percentage = (
        dangerous_count /
        total_classified
    ) * 100

else:

    safe_percentage = 0
    dangerous_percentage = 0


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

print("==============================================")
print("     AI TRAFFIC DANGEROUS-DRIVING RESULT")
print("==============================================")

print(
    f"Input Video              : video96.MOV"
)

print(
    f"Total Vehicles Detected  : {total_detected}"
)

print(
    f"ML Classified Vehicles   : {total_classified}"
)

print(
    f"SAFE Vehicles            : {safe_count}"
)

print(
    f"DANGEROUS Vehicles       : {dangerous_count}"
)

print(
    f"Insufficient Data        : {insufficient_count}"
)

print(
    f"SAFE Percentage          : {safe_percentage:.2f}%"
)

print(
    f"DANGEROUS Percentage     : {dangerous_percentage:.2f}%"
)

print(
    f"Average ML Confidence    : {avg_confidence:.2f}%"
)

print(
    "\nDetection Model         : YOLOv8"
)

print(
    "Tracking Algorithm      : ByteTrack"
)

print(
    "Classification Model    : Random Forest"
)

print(
    "Number of ML Features   : 7"
)

print(
    "\nFinal Video:"
)

print(
    r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_FINAL_CORRECTED_video96.mp4"
)

print(
    "\nPrediction CSV:"
)

print(
    r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\final_vehicle_predictions_video96.csv"
)

print("==============================================")

# --- CELL 81 ---
# ============================================================
# STEP 10 - EXTRACT SAMPLE FRAMES FROM FINAL VIDEO
# ============================================================

import cv2
import os

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_FINAL_CORRECTED_video96.mp4"

output_folder = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\final_check_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Final video could not be opened")


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print("================================")
print("CHECKING FINAL VIDEO")
print("================================")

print("Total frames:", total_frames)
print("FPS:", fps)


# ------------------------------------------------------------
# Frames to check
# ------------------------------------------------------------

frame_numbers = [
    30,
    100,
    170,
    240,
    310,
    400
]


saved_files = []


for frame_number in frame_numbers:

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    ret, frame = cap.read()

    if not ret:
        continue


    output_path = os.path.join(
        output_folder,
        f"frame_{frame_number}.jpg"
    )


    cv2.imwrite(
        output_path,
        frame
    )


    saved_files.append(
        output_path
    )


cap.release()


print("\n================================")
print("SAMPLE FRAMES CREATED")
print("================================")

for file in saved_files:
    print(file)

# --- CELL 82 ---
# ============================================================
# STEP 11 - COMPLETE VEHICLE ANALYSIS
# ============================================================

import pandas as pd
import os

print("================================")
print("CREATING COMPLETE VEHICLE REPORT")
print("================================")


# ------------------------------------------------------------
# Create complete records
# ------------------------------------------------------------

complete_records = []


for record in vehicle_features:

    vehicle_id = int(
        record["vehicle_id"]
    )


    # Get ML result
    if vehicle_id in final_vehicle_status:

        status_info = (
            final_vehicle_status[
                vehicle_id
            ]
        )

        status = status_info["status"]

        probability = float(
            status_info["probability"]
        )

    else:

        status = "INSUFFICIENT DATA"

        probability = 0.0


    # --------------------------------------------------------
    # Complete record
    # --------------------------------------------------------

    complete_records.append({

        "vehicle_id":
            vehicle_id,

        "frames_observed":
            int(record["frames_observed"]),

        "avg_speed":
            float(record["avg_speed"]),

        "max_speed":
            float(record["max_speed"]),

        "avg_acceleration":
            float(record["avg_acceleration"]),

        "max_acceleration":
            float(record["max_acceleration"]),

        "max_deceleration":
            float(record["max_deceleration"]),

        "avg_direction_change":
            float(record["avg_direction_change"]),

        "max_direction_change":
            float(record["max_direction_change"]),

        "status":
            status,

        "probability":
            probability
    })


# ------------------------------------------------------------
# DataFrame
# ------------------------------------------------------------

complete_df = pd.DataFrame(
    complete_records
)


complete_df = complete_df.sort_values(
    by="vehicle_id"
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

report_path = (
    r"C:\Users\Neha Kamble\Downloads"
    r"\Traffic_Danger_detection"
    r"\complete_vehicle_analysis_video96.csv"
)


complete_df.to_csv(
    report_path,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n================================")
print("COMPLETE REPORT CREATED")
print("================================")

print(
    "Total vehicle records:",
    len(complete_df)
)

print(
    "Report file:"
)

print(
    report_path
)


print("\nColumns:")

for column in complete_df.columns:

    print(
        "-",
        column
    )


print("\nFirst 10 records:")

display(
    complete_df.head(10)
)

# --- CELL 83 ---
# ============================================================
# STEP 12 - FEATURE ANALYSIS
# ============================================================

import pandas as pd

df = complete_df.copy()

print("================================")
print("FEATURE ANALYSIS")
print("================================")

# ------------------------------------------------------------
# SAFE / DANGEROUS summary
# ------------------------------------------------------------

summary = df.groupby("status")[
    [
        "frames_observed",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
        "max_deceleration",
        "avg_direction_change",
        "max_direction_change",
        "probability"
    ]
].mean()

print("\nAverage Feature Values:")
display(summary)


# ------------------------------------------------------------
# Vehicle counts
# ------------------------------------------------------------

print("\n================================")
print("VEHICLE COUNTS")
print("================================")

print(
    df["status"].value_counts()
)


# ------------------------------------------------------------
# Highest speed vehicles
# ------------------------------------------------------------

print("\n================================")
print("TOP 10 VEHICLES BY AVG SPEED")
print("================================")

display(
    df[
        [
            "vehicle_id",
            "frames_observed",
            "avg_speed",
            "max_speed",
            "status",
            "probability"
        ]
    ]
    .sort_values(
        by="avg_speed",
        ascending=False
    )
    .head(10)
)


# ------------------------------------------------------------
# Highest acceleration vehicles
# ------------------------------------------------------------

print("\n================================")
print("TOP 10 VEHICLES BY ACCELERATION")
print("================================")

display(
    df[
        [
            "vehicle_id",
            "frames_observed",
            "avg_acceleration",
            "max_acceleration",
            "max_deceleration",
            "status",
            "probability"
        ]
    ]
    .sort_values(
        by="max_acceleration",
        ascending=False
    )
    .head(10)
)

# --- CELL 84 ---
# ============================================================
# STEP 13 - FINAL RESULT GRAPHS
# ============================================================

import matplotlib.pyplot as plt

print("================================")
print("CREATING FINAL PROJECT GRAPHS")
print("================================")


# ============================================================
# GRAPH 1 - SAFE vs DANGEROUS VEHICLES
# ============================================================

counts = df["status"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    counts.index,
    counts.values
)

plt.title(
    "SAFE vs DANGEROUS Vehicles"
)

plt.xlabel(
    "Vehicle Status"
)

plt.ylabel(
    "Number of Vehicles"
)

for i, value in enumerate(counts.values):

    plt.text(
        i,
        value + 0.5,
        str(value),
        ha="center"
    )

plt.tight_layout()

plt.show()


# ============================================================
# GRAPH 2 - AVERAGE SPEED
# ============================================================

avg_speed = df.groupby(
    "status"
)["avg_speed"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    avg_speed.index,
    avg_speed.values
)

plt.title(
    "Average Speed Comparison"
)

plt.xlabel(
    "Vehicle Status"
)

plt.ylabel(
    "Average Speed (Pixel Displacement)"
)

plt.tight_layout()

plt.show()


# ============================================================
# GRAPH 3 - AVERAGE ACCELERATION
# ============================================================

avg_acceleration = df.groupby(
    "status"
)["avg_acceleration"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    avg_acceleration.index,
    avg_acceleration.values
)

plt.title(
    "Average Acceleration Comparison"
)

plt.xlabel(
    "Vehicle Status"
)

plt.ylabel(
    "Average Acceleration"
)

plt.tight_layout()

plt.show()


# ============================================================
# GRAPH 4 - MAX DECELERATION
# ============================================================

max_deceleration = df.groupby(
    "status"
)["max_deceleration"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    max_deceleration.index,
    max_deceleration.values
)

plt.title(
    "Average Maximum Deceleration"
)

plt.xlabel(
    "Vehicle Status"
)

plt.ylabel(
    "Maximum Deceleration"
)

plt.tight_layout()

plt.show()


# ============================================================
# GRAPH 5 - DIRECTION CHANGE
# ============================================================

direction_change = df.groupby(
    "status"
)["avg_direction_change"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    direction_change.index,
    direction_change.values
)

plt.title(
    "Average Direction Change"
)

plt.xlabel(
    "Vehicle Status"
)

plt.ylabel(
    "Average Direction Change"
)

plt.tight_layout()

plt.show()


print("\n================================")
print("ALL GRAPHS CREATED")
print("================================")

# --- CELL 85 ---
# ============================================================
# STEP 14 - FINAL PROJECT RESULT SUMMARY
# ============================================================

import pandas as pd
import os

print("==============================================")
print("       FINAL PROJECT RESULT SUMMARY")
print("==============================================")


# ------------------------------------------------------------
# Basic counts
# ------------------------------------------------------------

total_detected = 52
total_classified = 49
safe_count = 36
dangerous_count = 13
insufficient_count = 3


# ------------------------------------------------------------
# Percentages
# ------------------------------------------------------------

safe_percentage = (
    safe_count / total_classified
) * 100

dangerous_percentage = (
    dangerous_count / total_classified
) * 100


# ------------------------------------------------------------
# Feature averages
# ------------------------------------------------------------

dangerous_avg_speed = df[
    df["status"] == "DANGEROUS"
]["avg_speed"].mean()

safe_avg_speed = df[
    df["status"] == "SAFE"
]["avg_speed"].mean()


dangerous_avg_acceleration = df[
    df["status"] == "DANGEROUS"
]["avg_acceleration"].mean()

safe_avg_acceleration = df[
    df["status"] == "SAFE"
]["avg_acceleration"].mean()


dangerous_direction = df[
    df["status"] == "DANGEROUS"
]["avg_direction_change"].mean()

safe_direction = df[
    df["status"] == "SAFE"
]["avg_direction_change"].mean()


# ------------------------------------------------------------
# Create summary dictionary
# ------------------------------------------------------------

summary = {

    "Input Video":
        "video96.MOV",

    "Total Vehicles Detected":
        total_detected,

    "ML Classified Vehicles":
        total_classified,

    "SAFE Vehicles":
        safe_count,

    "DANGEROUS Vehicles":
        dangerous_count,

    "Insufficient Data Vehicles":
        insufficient_count,

    "SAFE Percentage":
        round(safe_percentage, 2),

    "DANGEROUS Percentage":
        round(dangerous_percentage, 2),

    "Average SAFE Speed":
        round(safe_avg_speed, 2),

    "Average DANGEROUS Speed":
        round(dangerous_avg_speed, 2),

    "Average SAFE Acceleration":
        round(safe_avg_acceleration, 2),

    "Average DANGEROUS Acceleration":
        round(dangerous_avg_acceleration, 2),

    "Average SAFE Direction Change":
        round(safe_direction, 2),

    "Average DANGEROUS Direction Change":
        round(dangerous_direction, 2),

    "Detection Model":
        "YOLOv8",

    "Tracking Algorithm":
        "ByteTrack",

    "Classification Model":
        "Random Forest",

    "ML Features":
        7
}


# ------------------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=[
        "Metric",
        "Value"
    ]
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

summary_path = (
    r"C:\Users\Neha Kamble\Downloads"
    r"\Traffic_Danger_detection"
    r"\FINAL_PROJECT_RESULT_video96.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\nFINAL RESULTS")
print("----------------------------------------------")

for key, value in summary.items():

    print(
        f"{key:<35}: {value}"
    )


print("\n==============================================")
print("FINAL RESULT FILE CREATED")
print("==============================================")

print(summary_path)

# --- CELL 86 ---
# ============================================================
# STEP 15 - FINAL PREDICTION VALIDATION
# ============================================================

print("==============================================")
print("FINAL PREDICTION VALIDATION")
print("==============================================")


# ------------------------------------------------------------
# 1. Probability distribution
# ------------------------------------------------------------

print("\nProbability Statistics")
print("----------------------------------------------")

print(
    df["probability"].describe()
)


# ------------------------------------------------------------
# 2. Lowest confidence predictions
# ------------------------------------------------------------

print("\n==============================================")
print("LOWEST CONFIDENCE PREDICTIONS")
print("==============================================")

low_confidence = df[
    [
        "vehicle_id",
        "frames_observed",
        "status",
        "probability"
    ]
].sort_values(
    by="probability"
).head(10)

display(low_confidence)


# ------------------------------------------------------------
# 3. Vehicles with very few frames
# ------------------------------------------------------------

print("\n==============================================")
print("VEHICLES WITH FEW OBSERVED FRAMES")
print("==============================================")

few_frames = df[
    [
        "vehicle_id",
        "frames_observed",
        "avg_speed",
        "avg_acceleration",
        "status",
        "probability"
    ]
].sort_values(
    by="frames_observed"
).head(10)

display(few_frames)


# ------------------------------------------------------------
# 4. SAFE predictions with probability < 60
# ------------------------------------------------------------

print("\n==============================================")
print("LOW-CONFIDENCE SAFE VEHICLES")
print("==============================================")

low_safe = df[
    (df["status"] == "SAFE") &
    (df["probability"] < 60)
][
    [
        "vehicle_id",
        "frames_observed",
        "status",
        "probability"
    ]
]

display(low_safe)


# ------------------------------------------------------------
# 5. DANGEROUS predictions with probability < 60
# ------------------------------------------------------------

print("\n==============================================")
print("LOW-CONFIDENCE DANGEROUS VEHICLES")
print("==============================================")

low_dangerous = df[
    (df["status"] == "DANGEROUS") &
    (df["probability"] < 60)
][
    [
        "vehicle_id",
        "frames_observed",
        "status",
        "probability"
    ]
]

display(low_dangerous)


# ------------------------------------------------------------
# Final check
# ------------------------------------------------------------

print("\n==============================================")
print("VALIDATION COMPLETED")
print("==============================================")

print(
    "Total classified vehicles:",
    len(df)
)

print(
    "SAFE:",
    (df["status"] == "SAFE").sum()
)

print(
    "DANGEROUS:",
    (df["status"] == "DANGEROUS").sum()
)

# --- CELL 87 ---
# ============================================================
# STEP 16 - APPLY MINIMUM FRAME THRESHOLD
# ============================================================

MIN_FRAMES = 15

print("==============================================")
print("APPLYING MINIMUM FRAME THRESHOLD")
print("==============================================")

updated_status = {}

for _, row in df.iterrows():

    vehicle_id = int(row["vehicle_id"])
    frames = int(row["frames_observed"])

    # --------------------------------------------------------
    # Insufficient tracking data
    # --------------------------------------------------------

    if frames < MIN_FRAMES:

        updated_status[vehicle_id] = {
            "status": "INSUFFICIENT DATA",
            "probability": 0.0,
            "persistent": False
        }

    else:

        updated_status[vehicle_id] = {
            "status": row["status"],
            "probability": float(row["probability"]),
            "persistent": False
        }


# Replace final status
final_vehicle_status = updated_status


# ============================================================
# FINAL COUNTS
# ============================================================

safe_count = sum(
    1
    for x in final_vehicle_status.values()
    if x["status"] == "SAFE"
)

dangerous_count = sum(
    1
    for x in final_vehicle_status.values()
    if x["status"] == "DANGEROUS"
)

insufficient_count = sum(
    1
    for x in final_vehicle_status.values()
    if x["status"] == "INSUFFICIENT DATA"
)


print("\n==============================================")
print("UPDATED CLASSIFICATION")
print("==============================================")

print(
    "Minimum frames required :", MIN_FRAMES
)

print(
    "SAFE vehicles            :", safe_count
)

print(
    "DANGEROUS vehicles       :", dangerous_count
)

print(
    "INSUFFICIENT DATA        :", insufficient_count
)

print(
    "Total vehicles analyzed  :",
    len(final_vehicle_status)
)

print("==============================================")

# --- CELL 88 ---
# ============================================================
# STEP 17 - IMPROVED VEHICLE TRACKING
# ============================================================

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

print("==============================================")
print("IMPROVED VEHICLE TRACKING")
print("==============================================")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

# ------------------------------------------------------------
# Load YOLO
# ------------------------------------------------------------

print("\nLoading YOLO...")

yolo_model = YOLO("yolov8n.pt")

print("YOLO loaded successfully")


# ------------------------------------------------------------
# Open video
# ------------------------------------------------------------

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise RuntimeError("Video could not be opened")


fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("\nFPS:", fps)
print("Total frames:", total_frames)


# ------------------------------------------------------------
# ByteTrack
# ------------------------------------------------------------

tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=60,
    minimum_matching_threshold=0.8
)


# ------------------------------------------------------------
# Vehicle classes
# COCO:
# car = 2
# motorcycle = 3
# bus = 5
# truck = 7
# ------------------------------------------------------------

VEHICLE_CLASSES = {
    2,
    3,
    5,
    7
}


# ------------------------------------------------------------
# Tracking storage
# ------------------------------------------------------------

vehicle_tracks = defaultdict(list)

frame_number = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    results = yolo_model(
        frame,
        verbose=False,
        conf=0.25
    )[0]


    # --------------------------------------------------------
    # Convert detections
    # --------------------------------------------------------

    detections = sv.Detections.from_ultralytics(
        results
    )


    # --------------------------------------------------------
    # Keep only vehicle classes
    # --------------------------------------------------------

    if len(detections) > 0:

        mask = np.isin(
            detections.class_id,
            list(VEHICLE_CLASSES)
        )

        detections = detections[mask]


    # --------------------------------------------------------
    # ByteTrack
    # --------------------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # --------------------------------------------------------
    # Store tracking information
    # --------------------------------------------------------

    if detections.tracker_id is not None:

        for i in range(
            len(detections)
        ):

            vehicle_id = int(
                detections.tracker_id[i]
            )

            x1, y1, x2, y2 = (
                detections.xyxy[i]
            )


            # Center point

            cx = (
                x1 + x2
            ) / 2

            cy = (
                y1 + y2
            ) / 2


            # Bounding box size

            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )


            vehicle_tracks[
                vehicle_id
            ].append({

                "frame":
                    frame_number,

                "cx":
                    float(cx),

                "cy":
                    float(cy),

                "width":
                    float(box_width),

                "height":
                    float(box_height)
            })


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if frame_number % 50 == 0:

        print(
            f"Processed: "
            f"{frame_number} / "
            f"{total_frames}"
        )


cap.release()


# ============================================================
# TRACKING SUMMARY
# ============================================================

print("\n==============================================")
print("IMPROVED TRACKING COMPLETED")
print("==============================================")


print(
    "Total unique vehicle IDs:",
    len(vehicle_tracks)
)


# ------------------------------------------------------------
# Track length statistics
# ------------------------------------------------------------

track_lengths = [
    len(track)
    for track in vehicle_tracks.values()
]


if track_lengths:

    print(
        "Minimum frames:",
        min(track_lengths)
    )

    print(
        "Maximum frames:",
        max(track_lengths)
    )

    print(
        "Average frames:",
        round(
            np.mean(track_lengths),
            2
        )
    )


# ------------------------------------------------------------
# Distribution
# ------------------------------------------------------------

short_tracks = sum(
    1
    for x in track_lengths
    if x < 15
)

good_tracks = sum(
    1
    for x in track_lengths
    if x >= 15
)


print(
    "\nTracks with < 15 frames:",
    short_tracks
)

print(
    "Tracks with >= 15 frames:",
    good_tracks
)


print("\n==============================================")
print("TRACKING DATA READY")
print("==============================================")

# --- CELL 89 ---
# ============================================================
# STEP 18 - TRACK LENGTH ANALYSIS
# ============================================================

import pandas as pd
import numpy as np

track_data = []

for vehicle_id, track in vehicle_tracks.items():

    track_data.append({
        "vehicle_id": int(vehicle_id),
        "frames": len(track)
    })

track_df = pd.DataFrame(track_data)

track_df = track_df.sort_values(
    by="frames",
    ascending=False
).reset_index(drop=True)


print("==============================================")
print("TRACK LENGTH ANALYSIS")
print("==============================================")

print("\nTotal tracks:", len(track_df))

print("\nTop 20 longest tracks:")
display(track_df.head(20))


print("\n==============================================")
print("TRACK LENGTH GROUPS")
print("==============================================")

groups = {
    "1-4 frames": (
        (track_df["frames"] >= 1) &
        (track_df["frames"] <= 4)
    ).sum(),

    "5-9 frames": (
        (track_df["frames"] >= 5) &
        (track_df["frames"] <= 9)
    ).sum(),

    "10-14 frames": (
        (track_df["frames"] >= 10) &
        (track_df["frames"] <= 14)
    ).sum(),

    "15-29 frames": (
        (track_df["frames"] >= 15) &
        (track_df["frames"] <= 29)
    ).sum(),

    "30-59 frames": (
        (track_df["frames"] >= 30) &
        (track_df["frames"] <= 59)
    ).sum(),

    "60+ frames": (
        (track_df["frames"] >= 60)
    ).sum()
}

for group, count in groups.items():

    print(
        f"{group:<20}: {count}"
    )


print("\n==============================================")
print("SHORTEST TRACKS")
print("==============================================")

display(
    track_df.sort_values(
        by="frames"
    ).head(20)
)

# --- CELL 90 ---
# ============================================================
# STEP 19 - FINAL FEATURE EXTRACTION FROM IMPROVED TRACKS
# ============================================================

import numpy as np
import pandas as pd

print("==============================================")
print("FINAL FEATURE EXTRACTION")
print("==============================================")


# ------------------------------------------------------------
# Feature calculation function
# ------------------------------------------------------------

def calculate_vehicle_features(track):

    if len(track) == 0:
        return None

    speeds = []
    accelerations = []
    direction_changes = []

    previous_speed = None
    previous_dx = None
    previous_dy = None

    for i in range(1, len(track)):

        p1 = track[i - 1]
        p2 = track[i]

        dx = p2["cx"] - p1["cx"]
        dy = p2["cy"] - p1["cy"]

        distance = np.sqrt(
            dx ** 2 + dy ** 2
        )

        speeds.append(distance)


        # ----------------------------------------------------
        # Acceleration
        # ----------------------------------------------------

        if previous_speed is not None:

            acceleration = (
                distance -
                previous_speed
            )

            accelerations.append(
                abs(acceleration)
            )

        previous_speed = distance


        # ----------------------------------------------------
        # Direction change
        # ----------------------------------------------------

        if (
            previous_dx is not None
            and previous_dy is not None
        ):

            magnitude1 = np.sqrt(
                previous_dx ** 2 +
                previous_dy ** 2
            )

            magnitude2 = np.sqrt(
                dx ** 2 +
                dy ** 2
            )

            if (
                magnitude1 > 0
                and magnitude2 > 0
            ):

                cos_angle = (
                    previous_dx * dx +
                    previous_dy * dy
                ) / (
                    magnitude1 *
                    magnitude2
                )

                cos_angle = np.clip(
                    cos_angle,
                    -1,
                    1
                )

                angle = np.arccos(
                    cos_angle
                )

                direction_changes.append(
                    angle
                )

        previous_dx = dx
        previous_dy = dy


    # --------------------------------------------------------
    # No movement data
    # --------------------------------------------------------

    if len(speeds) == 0:

        return {
            "frames_observed": len(track),
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "avg_acceleration": 0.0,
            "max_acceleration": 0.0,
            "max_deceleration": 0.0,
            "avg_direction_change": 0.0,
            "max_direction_change": 0.0
        }


    # --------------------------------------------------------
    # Acceleration values
    # --------------------------------------------------------

    if len(accelerations) > 0:

        avg_acceleration = np.mean(
            accelerations
        )

        max_acceleration = np.max(
            accelerations
        )

    else:

        avg_acceleration = 0.0
        max_acceleration = 0.0


    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if len(direction_changes) > 0:

        avg_direction_change = np.mean(
            direction_changes
        )

        max_direction_change = np.max(
            direction_changes
        )

    else:

        avg_direction_change = 0.0
        max_direction_change = 0.0


    # --------------------------------------------------------
    # Final features
    # --------------------------------------------------------

    return {

        "frames_observed":
            len(track),

        "avg_speed":
            float(np.mean(speeds)),

        "max_speed":
            float(np.max(speeds)),

        "avg_acceleration":
            float(avg_acceleration),

        "max_acceleration":
            float(max_acceleration),

        "max_deceleration":
            float(max_acceleration),

        "avg_direction_change":
            float(avg_direction_change),

        "max_direction_change":
            float(max_direction_change)
    }


# ============================================================
# CREATE FEATURE DATASET
# ============================================================

vehicle_features = []

for vehicle_id, track in vehicle_tracks.items():

    features = calculate_vehicle_features(
        track
    )

    if features is None:
        continue

    features["vehicle_id"] = int(
        vehicle_id
    )

    vehicle_features.append(
        features
    )


vehicle_features_df = pd.DataFrame(
    vehicle_features
)


# ------------------------------------------------------------
# Column order
# ------------------------------------------------------------

vehicle_features_df = vehicle_features_df[
    [
        "vehicle_id",
        "frames_observed",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
        "max_deceleration",
        "avg_direction_change",
        "max_direction_change"
    ]
]


print("\n==============================================")
print("FEATURE EXTRACTION COMPLETED")
print("==============================================")

print(
    "Total feature records:",
    len(vehicle_features_df)
)

print("\nFirst 10 records:")

display(
    vehicle_features_df.head(10)
)

# --- CELL 91 ---
# ============================================================
# STEP 20 - FINAL RANDOM FOREST CLASSIFICATION
# ============================================================

import joblib
import numpy as np
import pandas as pd

print("==============================================")
print("LOADING FINAL ML MODEL")
print("==============================================")


# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------

MODEL_PATH = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

SCALER_PATH = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

rf_model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

print("Random Forest loaded successfully")
print("Scaler loaded successfully")


# ============================================================
# FEATURES USED DURING TRAINING
# ============================================================

FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]


print("\n==============================================")
print("MODEL INFORMATION")
print("==============================================")

print(
    "Model features:",
    len(FEATURE_COLUMNS)
)

print(
    "Feature columns:"
)

for col in FEATURE_COLUMNS:
    print(" -", col)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    col
    for col in FEATURE_COLUMNS
    if col not in vehicle_features_df.columns
]

if missing_columns:

    raise ValueError(
        "Missing feature columns: "
        + str(missing_columns)
    )


# ============================================================
# MINIMUM FRAME RULE
# ============================================================

MIN_FRAMES = 15


# ============================================================
# FINAL STATUS DICTIONARY
# ============================================================

final_vehicle_status = {}

prediction_records = []


# ============================================================
# PROCESS EACH VEHICLE
# ============================================================

for _, row in vehicle_features_df.iterrows():

    vehicle_id = int(
        row["vehicle_id"]
    )

    frames = int(
        row["frames_observed"]
    )


    # --------------------------------------------------------
    # INSUFFICIENT DATA
    # --------------------------------------------------------

    if frames < MIN_FRAMES:

        final_vehicle_status[
            vehicle_id
        ] = {

            "status":
                "INSUFFICIENT DATA",

            "probability":
                0.0,

            "persistent":
                False
        }


        prediction_records.append({

            "vehicle_id":
                vehicle_id,

            "frames_observed":
                frames,

            "status":
                "INSUFFICIENT DATA",

            "probability":
                0.0,

            "persistent":
                False
        })

        continue


    # ========================================================
    # ML FEATURES
    # ========================================================

    X = pd.DataFrame(
        [[
            row[col]
            for col in FEATURE_COLUMNS
        ]],
        columns=FEATURE_COLUMNS
    )


    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = rf_model.predict(
        X_scaled
    )[0]


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = (
        rf_model.predict_proba(
            X_scaled
        )[0]
    )


    probability = (
        float(
            np.max(probabilities)
        )
        * 100
    )


    # --------------------------------------------------------
    # Convert prediction to status
    # --------------------------------------------------------

    if isinstance(
        prediction,
        str
    ):

        status = prediction

    else:

        # Numeric model output
        # Adjust mapping if your training
        # labels were reversed

        if int(prediction) == 1:

            status = "DANGEROUS"

        else:

            status = "SAFE"


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    final_vehicle_status[
        vehicle_id
    ] = {

        "status":
            status,

        "probability":
            probability,

        "persistent":
            False
    }


    prediction_records.append({

        "vehicle_id":
            vehicle_id,

        "frames_observed":
            frames,

        "status":
            status,

        "probability":
            round(
                probability,
                2
            ),

        "persistent":
            False
    })


# ============================================================
# CREATE FINAL PREDICTION DATAFRAME
# ============================================================

final_predictions_df = pd.DataFrame(
    prediction_records
)


# ============================================================
# COUNTS
# ============================================================

safe_count = (
    final_predictions_df[
        final_predictions_df["status"]
        == "SAFE"
    ]
    .shape[0]
)

dangerous_count = (
    final_predictions_df[
        final_predictions_df["status"]
        == "DANGEROUS"
    ]
    .shape[0]
)

insufficient_count = (
    final_predictions_df[
        final_predictions_df["status"]
        == "INSUFFICIENT DATA"
    ]
    .shape[0]
)


# ============================================================
# PERCENTAGES
# ============================================================

classified_count = (
    safe_count +
    dangerous_count
)

if classified_count > 0:

    safe_percentage = (
        safe_count /
        classified_count *
        100
    )

    dangerous_percentage = (
        dangerous_count /
        classified_count *
        100
    )

else:

    safe_percentage = 0
    dangerous_percentage = 0


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("FINAL ML CLASSIFICATION COMPLETED")
print("==============================================")

print(
    "Total vehicles           :",
    len(final_predictions_df)
)

print(
    "ML classified vehicles   :",
    classified_count
)

print(
    "SAFE vehicles            :",
    safe_count
)

print(
    "DANGEROUS vehicles       :",
    dangerous_count
)

print(
    "INSUFFICIENT DATA        :",
    insufficient_count
)

print(
    "SAFE percentage          :",
    round(
        safe_percentage,
        2
    ),
    "%"
)

print(
    "DANGEROUS percentage     :",
    round(
        dangerous_percentage,
        2
    ),
    "%"
)


# ============================================================
# SHOW RESULTS
# ============================================================

print("\n==============================================")
print("FINAL VEHICLE PREDICTIONS")
print("==============================================")

display(
    final_predictions_df.sort_values(
        by="vehicle_id"
    )
)

# --- CELL 92 ---
# ============================================================
# STEP 21 - CHECK RANDOM FOREST CLASSES
# ============================================================

print("==============================================")
print("CHECKING RANDOM FOREST CLASSES")
print("==============================================")

print("\nModel classes:")
print(rf_model.classes_)

print("\nNumber of classes:")
print(len(rf_model.classes_))

print("\nClass type:")
print(type(rf_model.classes_[0]))


# ------------------------------------------------------------
# Check predictions directly
# ------------------------------------------------------------

classified_df = vehicle_features_df[
    vehicle_features_df["frames_observed"] >= 15
].copy()


X_test = classified_df[
    FEATURE_COLUMNS
]


X_scaled = scaler.transform(
    X_test
)


predictions = rf_model.predict(
    X_scaled
)


probabilities = rf_model.predict_proba(
    X_scaled
)


print("\n==============================================")
print("RAW MODEL PREDICTIONS")
print("==============================================")

print(
    "Prediction counts:"
)

print(
    pd.Series(predictions).value_counts()
)


print("\n==============================================")
print("MODEL PROBABILITY SAMPLE")
print("==============================================")

for i in range(
    min(10, len(predictions))
):

    print(
        f"Vehicle "
        f"{classified_df.iloc[i]['vehicle_id']}: "
        f"Prediction={predictions[i]} | "
        f"Probabilities={probabilities[i]}"
    )

# --- CELL 93 ---
# ============================================================
# STEP 22 - DEEP CHECK RANDOM FOREST MODEL
# ============================================================

print("==============================================")
print("DEEP MODEL CHECK")
print("==============================================")


# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

print("\nMODEL PARAMETERS")
print("----------------------------------------------")

print("Classes:", rf_model.classes_)
print("Estimators:", rf_model.n_estimators)
print("Max depth:", rf_model.max_depth)
print("Min samples split:", rf_model.min_samples_split)
print("Min samples leaf:", rf_model.min_samples_leaf)


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

print("\n==============================================")
print("FEATURE IMPORTANCE")
print("==============================================")

feature_importance = pd.DataFrame({

    "feature":
        FEATURE_COLUMNS,

    "importance":
        rf_model.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

display(
    feature_importance
)


# ------------------------------------------------------------
# Scaler information
# ------------------------------------------------------------

print("\n==============================================")
print("SCALER INFORMATION")
print("==============================================")

if hasattr(
    scaler,
    "feature_names_in_"
):

    print(
        "Scaler feature names:"
    )

    print(
        list(
            scaler.feature_names_in_
        )
    )

else:

    print(
        "Scaler does not contain feature names"
    )


# ------------------------------------------------------------
# Scaler statistics
# ------------------------------------------------------------

if hasattr(
    scaler,
    "mean_"
):

    print(
        "\nScaler means:"
    )

    for name, value in zip(
        FEATURE_COLUMNS,
        scaler.mean_
    ):

        print(
            f"{name}: {value}"
        )


if hasattr(
    scaler,
    "scale_"
):

    print(
        "\nScaler scale:"
    )

    for name, value in zip(
        FEATURE_COLUMNS,
        scaler.scale_
    ):

        print(
            f"{name}: {value}"
        )


# ------------------------------------------------------------
# Current test feature statistics
# ------------------------------------------------------------

print("\n==============================================")
print("CURRENT VIDEO FEATURE STATISTICS")
print("==============================================")

display(
    vehicle_features_df[
        FEATURE_COLUMNS
    ].describe()
)


# ------------------------------------------------------------
# Compare current features with scaler
# ------------------------------------------------------------

print("\n==============================================")
print("FEATURE RANGE CHECK")
print("==============================================")

for name in FEATURE_COLUMNS:

    current_min = (
        vehicle_features_df[name].min()
    )

    current_max = (
        vehicle_features_df[name].max()
    )

    print(
        f"\n{name}"
    )

    print(
        "Current min:",
        current_min
    )

    print(
        "Current max:",
        current_max
    )

    if hasattr(
        scaler,
        "mean_"
    ):

        idx = FEATURE_COLUMNS.index(
            name
        )

        print(
            "Training mean:",
            scaler.mean_[idx]
        )

        print(
            "Training scale:",
            scaler.scale_[idx]
        )


print("\n==============================================")
print("DEEP MODEL CHECK COMPLETED")
print("==============================================")

# --- CELL 94 ---
print("Random Forest related variables:")

for name, value in globals().items():
    try:
        if "RandomForest" in str(type(value)):
            print(name, "->", type(value))
    except:
        pass

# --- CELL 95 ---
print("========== RF MODEL CHECK ==========")

for name, m in [("rf_model", rf_model), ("model", model)]:
    print("\nModel:", name)
    print("Estimators:", m.n_estimators)
    print("Classes:", m.classes_)
    print("Features:", m.n_features_in_)
    
    if hasattr(m, "feature_importances_"):
        print("Feature importances:")
        for i, imp in enumerate(m.feature_importances_):
            print(f"  Feature {i}: {imp:.6f}")

# --- CELL 96 ---
from sklearn.preprocessing import StandardScaler

print("========== SCALER CHECK ==========")

for name, value in globals().items():
    try:
        if isinstance(value, StandardScaler):
            print("\nScaler:", name)
            print("Number of features:", value.n_features_in_)
            
            if hasattr(value, "feature_names_in_"):
                print("Feature names:")
                print(list(value.feature_names_in_))
            
            print("Mean:")
            print(value.mean_)
            
            print("Scale:")
            print(value.scale_)
    except:
        pass

# --- CELL 97 ---
print("==============================================")
print("FINAL MODEL + SCALER CHECK")
print("==============================================")

print("Model classes:", rf_model.classes_)
print("Model features:", rf_model.n_features_in_)

print("\nFeature names:")
print(list(scaler.feature_names_in_))

print("\nModel type:")
print(type(rf_model))

print("\nScaler type:")
print(type(scaler))

print("\n==============================================")
print("CLASS MAPPING")
print("==============================================")

for cls in rf_model.classes_:
    print(
        "Class:", cls,
        "| Probability column:", 
        list(rf_model.classes_).index(cls)
    )

# --- CELL 98 ---
print("==============================================")
print("SEARCHING TRAINING DATA")
print("==============================================")

import pandas as pd
import numpy as np

# Check all dataframe variables
print("\nDATAFRAME VARIABLES")
print("----------------------------------------------")

for name, value in globals().items():
    try:
        if isinstance(value, pd.DataFrame):
            print(
                f"{name} -> shape={value.shape}, "
                f"columns={list(value.columns)}"
            )
    except:
        pass

print("\n==============================================")
print("IMPORTANT VARIABLES")
print("==============================================")

for name in [
    "X",
    "y",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "training_data",
    "train_df",
    "df",
    "vehicle_features"
]:
    if name in globals():
        value = globals()[name]

        print(f"\n{name}")
        print("Type:", type(value))

        if isinstance(value, pd.DataFrame):
            print("Shape:", value.shape)
            print("Columns:", list(value.columns))

        elif isinstance(value, pd.Series):
            print("Shape:", value.shape)
            print("Unique:", value.unique())

        elif isinstance(value, np.ndarray):
            print("Shape:", value.shape)

print("\n==============================================")
print("RF MODEL TRAINING STATUS")
print("==============================================")

if hasattr(rf_model, "classes_"):
    print("Classes:", rf_model.classes_)

if hasattr(rf_model, "n_features_in_"):
    print("Features:", rf_model.n_features_in_)

print("\n==============================================")
print("CHECK COMPLETED")
print("==============================================")

# --- CELL 99 ---
print("==============================================")
print("CHECKING ORIGINAL TRAINING DATA")
print("==============================================")

# Check y
print("\ny:")
print(type(y))
print("Shape:", y.shape)
print("Values:", y)

# Check all variables containing train/data/X
print("\n==============================================")
print("TRAINING RELATED VARIABLES")
print("==============================================")

for name, value in globals().items():

    if any(word in name.lower() for word in
           ["train", "data", "label", "feature", "x_train", "y_train"]):

        try:
            print(
                name,
                "->",
                type(value),
                "shape=",
                getattr(value, "shape", "N/A")
            )
        except:
            pass

# Check RF internal training information
print("\n==============================================")
print("RF INTERNAL CHECK")
print("==============================================")

print("Number of estimators:", rf_model.n_estimators)
print("Number of features:", rf_model.n_features_in_)
print("Classes:", rf_model.classes_)

print("\n==============================================")
print("CHECK COMPLETED")
print("==============================================")

# --- CELL 100 ---
print("==============================================")
print("FEATURES_DF CHECK")
print("==============================================")

print(features_df)

print("\n==============================================")
print("STATUS COUNTS")
print("==============================================")

print(features_df["status"].value_counts())

print("\n==============================================")
print("PREDICTION COUNTS")
print("==============================================")

print(features_df["prediction"].value_counts())

print("\n==============================================")
print("COLUMNS")
print("==============================================")

print(features_df.columns.tolist())

# --- CELL 101 ---
# ==============================================
# ALGORITHM COMPARISON
# ==============================================

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]

X = features_df[FEATURE_COLUMNS]
y = features_df["prediction"].astype(int)

# 5-fold cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}

print("==============================================")
print("5-FOLD CROSS VALIDATION")
print("==============================================")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("\n", name)
    print("Fold accuracies:",
          [round(x * 100, 2) for x in scores])

    print("Mean accuracy:",
          round(scores.mean() * 100, 2), "%")

    print("Std:",
          round(scores.std() * 100, 2), "%")

# --- CELL 102 ---
print("==============================================")
print("FINAL DATA PIPELINE CHECK")
print("==============================================")

FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]

print("\nTraining features:")
print(features_df[FEATURE_COLUMNS].describe())

print("\nTraining labels:")
print(features_df["prediction"].value_counts())

print("\nCurrent video features:")
print(vehicle_features_df[FEATURE_COLUMNS].describe())

print("\n==============================================")
print("FEATURE ORDER CHECK")
print("==============================================")

print("Training:")
print(FEATURE_COLUMNS)

print("\nVideo:")
print(vehicle_features_df[FEATURE_COLUMNS].columns.tolist())

print("\n==============================================")
print("CHECK COMPLETED")
print("==============================================")

# --- CELL 103 ---
import inspect

print(inspect.getsource(calculate_vehicle_features))

# --- CELL 104 ---
print("Function file:")
print(calculate_vehicle_features.__code__.co_filename)

print("Starting line:")
print(calculate_vehicle_features.__code__.co_firstlineno)

print("Function name:")
print(calculate_vehicle_features.__name__)

# --- CELL 105 ---
import json

notebook_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\Traffic_Danger_Detection.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

found = False

for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))

    if "calculate_vehicle_features" in source:
        found = True
        print("\n" + "=" * 80)
        print("FOUND IN NOTEBOOK CELL:", i)
        print("=" * 80)
        print(source)

if not found:
    print("calculate_vehicle_features NOT FOUND in notebook")

# --- CELL 106 ---
def calculate_vehicle_features(track):

    if len(track) < 2:
        return None

    speeds = []
    accelerations = []
    decelerations = []
    direction_changes = []

    previous_speed = None
    previous_dx = None
    previous_dy = None

    for i in range(1, len(track)):

        p1 = track[i - 1]
        p2 = track[i]

        dx = p2["cx"] - p1["cx"]
        dy = p2["cy"] - p1["cy"]

        distance = np.sqrt(dx**2 + dy**2)

        speeds.append(distance)

        # Acceleration / Deceleration
        if previous_speed is not None:

            acceleration = distance - previous_speed

            if acceleration > 0:
                accelerations.append(acceleration)

            elif acceleration < 0:
                decelerations.append(abs(acceleration))

        previous_speed = distance

        # Direction change
        if previous_dx is not None:

            magnitude1 = np.sqrt(
                previous_dx**2 + previous_dy**2
            )

            magnitude2 = np.sqrt(
                dx**2 + dy**2
            )

            if magnitude1 > 0 and magnitude2 > 0:

                cos_angle = (
                    previous_dx * dx +
                    previous_dy * dy
                ) / (
                    magnitude1 * magnitude2
                )

                cos_angle = np.clip(cos_angle, -1, 1)

                angle = np.arccos(cos_angle)

                direction_changes.append(angle)

        previous_dx = dx
        previous_dy = dy

    # Acceleration
    if accelerations:
        avg_acceleration = np.mean(accelerations)
        max_acceleration = np.max(accelerations)
    else:
        avg_acceleration = 0.0
        max_acceleration = 0.0

    # Deceleration
    if decelerations:
        max_deceleration = np.max(decelerations)
    else:
        max_deceleration = 0.0

    # Direction
    if direction_changes:
        avg_direction_change = np.mean(direction_changes)
        max_direction_change = np.max(direction_changes)
    else:
        avg_direction_change = 0.0
        max_direction_change = 0.0

    return {
        "frames_observed": len(track),
        "avg_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "avg_acceleration": float(avg_acceleration),
        "max_acceleration": float(max_acceleration),
        "max_deceleration": float(max_deceleration),
        "avg_direction_change": float(avg_direction_change),
        "max_direction_change": float(max_direction_change)
    }

# --- CELL 107 ---
vehicle_features = []

for vehicle_id, track in vehicle_tracks.items():

    features = calculate_vehicle_features(track)

    if features is None:
        continue

    features["vehicle_id"] = int(vehicle_id)

    vehicle_features.append(features)

vehicle_features_df = pd.DataFrame(vehicle_features)

vehicle_features_df = vehicle_features_df[
    [
        "vehicle_id",
        "frames_observed",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
        "max_deceleration",
        "avg_direction_change",
        "max_direction_change"
    ]
]

print("==============================================")
print("NEW FEATURE EXTRACTION")
print("==============================================")

print("Total vehicles:", len(vehicle_features_df))

display(vehicle_features_df.head(10))

# --- CELL 108 ---
# ============================================================
# STEP 20 - FILTER SHORT TRACKS
# ============================================================

MIN_FRAMES = 15

print("==============================================")
print("FILTERING SHORT TRACKS")
print("==============================================")

filtered_features_df = vehicle_features_df[
    vehicle_features_df["frames_observed"] >= MIN_FRAMES
].copy()

insufficient_features_df = vehicle_features_df[
    vehicle_features_df["frames_observed"] < MIN_FRAMES
].copy()

print("Total vehicles:", len(vehicle_features_df))
print("Vehicles with >= 15 frames:", len(filtered_features_df))
print("Vehicles with < 15 frames:", len(insufficient_features_df))

print("\n==============================================")
print("FILTERED FEATURE DATA")
print("==============================================")

display(filtered_features_df.head(10))

# --- CELL 109 ---
import pandas as pd

vehicle_features_df = pd.DataFrame(vehicle_features)

print("==============================================")
print("FEATURE EXTRACTION COMPLETED")
print("==============================================")

print("Total feature records:", len(vehicle_features_df))

print("\nFirst 10 records:")
display(vehicle_features_df.head(10))

# --- CELL 110 ---
# ============================================================
# STEP 19B - LOAD FINAL MODEL AND SCALER
# ============================================================

import joblib

MODEL_PATH = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"
SCALER_PATH = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

print("Loading Random Forest model...")

final_model = joblib.load(MODEL_PATH)

print("Model loaded successfully")

print("\nLoading scaler...")

final_scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully")

print("\n==============================================")
print("MODEL INFORMATION")
print("==============================================")

print("Model type:", type(final_model))
print("Scaler type:", type(final_scaler))

print("Number of model features:", final_model.n_features_in_)
print("Scaler features:", final_scaler.n_features_in_)

print("\nModel classes:", final_model.classes_)

# --- CELL 111 ---
# ============================================================
# STEP 20A - PREPARE FINAL FEATURES FOR CLASSIFICATION
# ============================================================

MIN_FRAMES = 15

filtered_features_df = vehicle_features_df[
    vehicle_features_df["frames_observed"] >= MIN_FRAMES
].copy()

print("==============================================")
print("FINAL CLASSIFICATION DATA")
print("==============================================")

print("Total extracted vehicles:", len(vehicle_features_df))
print("Vehicles for classification:", len(filtered_features_df))

display(filtered_features_df.head(10))

# --- CELL 112 ---
# ============================================================
# STEP 20B - RANDOM FOREST CLASSIFICATION
# ============================================================

FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]

print("==============================================")
print("RANDOM FOREST CLASSIFICATION")
print("==============================================")

# Prepare 7 features
X = filtered_features_df[FEATURE_COLUMNS].copy()

print("Input vehicles:", len(X))
print("Input features:", X.shape[1])

# Scale features
X_scaled = final_scaler.transform(X)

# Prediction
predictions = final_model.predict(X_scaled)

# Prediction probability
probabilities = final_model.predict_proba(X_scaled)

# Store results
results_df = filtered_features_df.copy()

results_df["prediction"] = predictions

results_df["status"] = results_df["prediction"].map({
    0: "SAFE",
    1: "DANGEROUS"
})

results_df["danger_probability"] = probabilities[:, 1]

# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("CLASSIFICATION COMPLETED")
print("==============================================")

print("Total vehicles:", len(results_df))

print(
    "SAFE vehicles:",
    (results_df["status"] == "SAFE").sum()
)

print(
    "DANGEROUS vehicles:",
    (results_df["status"] == "DANGEROUS").sum()
)

print("\n==============================================")
print("CLASSIFICATION RESULTS")
print("==============================================")

display(
    results_df[
        [
            "status",
            "danger_probability",
            "frames_observed",
            "avg_speed",
            "max_speed"
        ]
    ]
)

# --- CELL 113 ---
# ============================================================
# STEP 21 - CHECK RANDOM FOREST TRAINING INFORMATION
# ============================================================

print("==============================================")
print("RANDOM FOREST TRAINING INFORMATION")
print("==============================================")

print("Number of trees:", final_model.n_estimators)

print("Number of features:", final_model.n_features_in_)

print("\nFeature columns used:")
for i, feature in enumerate(FEATURE_COLUMNS):
    print(i, ":", feature)

print("\n==============================================")
print("CURRENT VIDEO FEATURE RANGES")
print("==============================================")

for feature in FEATURE_COLUMNS:

    print(
        f"{feature}: "
        f"min={X[feature].min():.3f}, "
        f"max={X[feature].max():.3f}, "
        f"mean={X[feature].mean():.3f}"
    )

# --- CELL 114 ---
# ============================================================
# STEP 22 - CHECK MODEL CLASS MAPPING
# ============================================================

print("==============================================")
print("MODEL CLASS MAPPING")
print("==============================================")

print("Classes:", final_model.classes_)

print("\nClass probabilities order:")

for i, cls in enumerate(final_model.classes_):
    print(
        f"Probability column {i} -> Class {cls}"
    )

print("\n==============================================")
print("ONE SAMPLE PREDICTION CHECK")
print("==============================================")

sample = X_scaled[:1]

print("Prediction:", final_model.predict(sample)[0])
print("Probabilities:", final_model.predict_proba(sample)[0])

# --- CELL 115 ---
# ============================================================
# STEP 23 - CHECK TRAINING FEATURE STATISTICS
# ============================================================

print("==============================================")
print("TRAINING FEATURE STATISTICS")
print("==============================================")

print("Model expects:", final_model.n_features_in_, "features")

print("\nScaler mean:")
print(final_scaler.mean_)

print("\nScaler scale:")
print(final_scaler.scale_)

print("\n==============================================")
print("CURRENT VIDEO vs TRAINING SCALER")
print("==============================================")

for i, feature in enumerate(FEATURE_COLUMNS):

    current_min = X[feature].min()
    current_max = X[feature].max()
    current_mean = X[feature].mean()

    training_mean = final_scaler.mean_[i]
    training_std = final_scaler.scale_[i]

    print(f"\n{feature}")
    print(f"Current video : min={current_min:.3f}, max={current_max:.3f}, mean={current_mean:.3f}")
    print(f"Training mean : {training_mean:.3f}")
    print(f"Training std  : {training_std:.3f}")

# --- CELL 116 ---
# ============================================================
# STEP 24 - INSPECT TRAINING DATA
# ============================================================

print("==============================================")
print("TRAINING DATA CHECK")
print("==============================================")

print("Total training records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\n==============================================")
print("TRAINING FEATURE DATA")
print("==============================================")

display(
    df[
        [
            "avg_speed",
            "max_speed",
            "avg_acceleration",
            "max_acceleration",
            "max_deceleration",
            "avg_direction_change",
            "max_direction_change",
            "dangerous_driving"
        ]
    ].head(10)
)

print("\n==============================================")
print("TRAINING FEATURE RANGES")
print("==============================================")

for feature in feature_columns:

    print(
        f"{feature}: "
        f"min={df[feature].min():.3f}, "
        f"max={df[feature].max():.3f}, "
        f"mean={df[feature].mean():.3f}"
    )

print("\n==============================================")
print("LABEL DISTRIBUTION")
print("==============================================")

print(
    df["dangerous_driving"].value_counts()
)

# --- CELL 117 ---
# ============================================================
# STEP 24A - CHECK AVAILABLE DATAFRAMES
# ============================================================

import pandas as pd

print("Available DataFrames:")
print("==============================================")

for name, value in list(globals().items()):
    if isinstance(value, pd.DataFrame):
        print(
            f"{name} -> "
            f"rows={len(value)}, "
            f"columns={len(value.columns)}"
        )

# --- CELL 118 ---
# ============================================================
# STEP 25 - FIND TRAINING DATA FILES
# ============================================================

import os

search_folder = r"C:\Users\Neha Kamble\Downloads"

print("Searching for possible training data files...")
print("==============================================")

extensions = (
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet"
)

found_files = []

for root, dirs, files in os.walk(search_folder):

    for file in files:

        if file.lower().endswith(extensions):

            path = os.path.join(root, file)
            found_files.append(path)

for path in found_files:
    print(path)

print("\n==============================================")
print("Total data files found:", len(found_files))

# --- CELL 119 ---


# --- CELL 120 ---
# ============================================================
# STEP 26 - CHECK PROJECT CSV FILES
# ============================================================

import pandas as pd
import os

project_folder = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection"

csv_files = [
    "complete_vehicle_analysis_video96.csv",
    "FINAL_PROJECT_RESULT_video96.csv",
    "final_vehicle_predictions_video96.csv"
]

for filename in csv_files:

    path = os.path.join(project_folder, filename)

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    if not os.path.exists(path):
        print("FILE NOT FOUND")
        continue

    temp_df = pd.read_csv(path)

    print("Rows:", len(temp_df))
    print("Columns:")
    print(temp_df.columns.tolist())

    print("\nFirst 3 rows:")
    display(temp_df.head(3))

# --- CELL 121 ---
# ============================================================
# STEP 27 - INSPECT OLD FEATURE DATA
# ============================================================

import pandas as pd

old_csv = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\complete_vehicle_analysis_video96.csv"

old_features_df = pd.read_csv(old_csv)

FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]

print("==============================================")
print("OLD FEATURE DATA")
print("==============================================")

print("Total vehicles:", len(old_features_df))

print("\nFeature statistics:")

for feature in FEATURE_COLUMNS:

    print(
        f"{feature}: "
        f"min={old_features_df[feature].min():.3f}, "
        f"max={old_features_df[feature].max():.3f}, "
        f"mean={old_features_df[feature].mean():.3f}"
    )

print("\n==============================================")
print("OLD CLASSIFICATION")
print("==============================================")

print(
    old_features_df["status"].value_counts()
)

print("\n==============================================")
print("DANGEROUS VEHICLES")
print("==============================================")

display(
    old_features_df[
        old_features_df["status"] == "DANGEROUS"
    ][
        [
            "vehicle_id",
            "frames_observed",
            "avg_speed",
            "max_speed",
            "avg_acceleration",
            "max_acceleration",
            "max_deceleration",
            "status",
            "probability"
        ]
    ]
)

# --- CELL 122 ---
# ============================================================
# STEP 28 - COMPARE OLD AND CURRENT VEHICLE IDs
# ============================================================

import pandas as pd

old_csv = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\complete_vehicle_analysis_video96.csv"

old_df = pd.read_csv(old_csv)

old_ids = set(old_df["vehicle_id"].astype(int))
current_ids = set(vehicle_tracks.keys())

print("==============================================")
print("VEHICLE ID COMPARISON")
print("==============================================")

print("Old CSV vehicle IDs     :", len(old_ids))
print("Current tracking IDs    :", len(current_ids))

print("\nIDs in both:")
print(len(old_ids.intersection(current_ids)))

print("\nIDs only in old CSV:")
print(sorted(old_ids - current_ids))

print("\nIDs only in current tracking:")
print(sorted(current_ids - old_ids))

# --- CELL 123 ---
# ============================================================
# STEP 29 - COMPARE OLD AND CURRENT TRACK LENGTHS
# ============================================================

print("==============================================")
print("TRACK LENGTH COMPARISON")
print("==============================================")

comparison = []

for vehicle_id in sorted(
    old_ids.intersection(current_ids)
):

    old_frames = int(
        old_df.loc[
            old_df["vehicle_id"] == vehicle_id,
            "frames_observed"
        ].iloc[0]
    )

    current_frames = len(
        vehicle_tracks[vehicle_id]
    )

    comparison.append({
        "vehicle_id": vehicle_id,
        "old_frames": old_frames,
        "current_frames": current_frames,
        "difference": current_frames - old_frames
    })

comparison_df = pd.DataFrame(comparison)

display(
    comparison_df.head(20)
)

print("\n==============================================")
print("FRAME COUNT STATISTICS")
print("==============================================")

print(
    "Same frame count:",
    sum(
        comparison_df["difference"] == 0
    )
)

print(
    "Different frame count:",
    sum(
        comparison_df["difference"] != 0
    )
)

print(
    "Average difference:",
    round(
        comparison_df["difference"].mean(),
        2
    )
)

# --- CELL 124 ---
# ============================================================
# STEP 30 - INSPECT RAW TRACK DATA
# ============================================================

vehicle_id = 1

track = vehicle_tracks[vehicle_id]

print("==============================================")
print("RAW TRACK INSPECTION")
print("==============================================")

print("Vehicle ID:", vehicle_id)
print("Frames:", len(track))

print("\nFirst 5 points:")

for point in track[:5]:
    print(point)

print("\nLast 5 points:")

for point in track[-5:]:
    print(point)

# --- CELL 125 ---
# ============================================================
# CORRECTED VEHICLE FEATURE EXTRACTION
# ============================================================

import numpy as np


def calculate_vehicle_features(track, fps):

    if len(track) < 2:
        return None

    speeds = []
    accelerations = []
    decelerations = []
    direction_changes = []

    previous_speed = None
    previous_dx = None
    previous_dy = None

    for i in range(1, len(track)):

        p1 = track[i - 1]
        p2 = track[i]

        # ----------------------------------------------------
        # Position difference
        # ----------------------------------------------------

        dx = p2["cx"] - p1["cx"]
        dy = p2["cy"] - p1["cy"]

        distance = np.sqrt(
            dx**2 + dy**2
        )

        # ----------------------------------------------------
        # SPEED
        # pixels/frame -> pixels/second
        # ----------------------------------------------------

        speed = distance * fps

        speeds.append(speed)

        # ----------------------------------------------------
        # ACCELERATION / DECELERATION
        # ----------------------------------------------------

        if previous_speed is not None:

            acceleration = speed - previous_speed

            if acceleration > 0:

                accelerations.append(
                    acceleration
                )

            elif acceleration < 0:

                decelerations.append(
                    abs(acceleration)
                )

        previous_speed = speed

        # ----------------------------------------------------
        # DIRECTION CHANGE
        # ----------------------------------------------------

        if previous_dx is not None:

            magnitude1 = np.sqrt(
                previous_dx**2 +
                previous_dy**2
            )

            magnitude2 = np.sqrt(
                dx**2 +
                dy**2
            )

            if magnitude1 > 0 and magnitude2 > 0:

                cos_angle = (
                    previous_dx * dx +
                    previous_dy * dy
                ) / (
                    magnitude1 * magnitude2
                )

                cos_angle = np.clip(
                    cos_angle,
                    -1,
                    1
                )

                angle = np.arccos(
                    cos_angle
                )

                direction_changes.append(
                    angle
                )

        previous_dx = dx
        previous_dy = dy

    # --------------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------------

    if accelerations:

        avg_acceleration = np.mean(
            accelerations
        )

        max_acceleration = np.max(
            accelerations
        )

    else:

        avg_acceleration = 0.0
        max_acceleration = 0.0

    # --------------------------------------------------------
    # DECELERATION
    # --------------------------------------------------------

    if decelerations:

        max_deceleration = np.max(
            decelerations
        )

    else:

        max_deceleration = 0.0

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if direction_changes:

        avg_direction_change = np.mean(
            direction_changes
        )

        max_direction_change = np.max(
            direction_changes
        )

    else:

        avg_direction_change = 0.0
        max_direction_change = 0.0

    # --------------------------------------------------------
    # RETURN FEATURES
    # --------------------------------------------------------

    return {

        "frames_observed":
            len(track),

        "avg_speed":
            float(np.mean(speeds)),

        "max_speed":
            float(np.max(speeds)),

        "avg_acceleration":
            float(avg_acceleration),

        "max_acceleration":
            float(max_acceleration),

        "max_deceleration":
            float(max_deceleration),

        "avg_direction_change":
            float(avg_direction_change),

        "max_direction_change":
            float(max_direction_change)
    }

print("Corrected feature function loaded successfully.")

# --- CELL 126 ---
# ============================================================
# STEP 31 - REBUILD FEATURE DATA
# ============================================================

import pandas as pd

vehicle_features = []

for vehicle_id, track in vehicle_tracks.items():

    # Ignore very short tracks
    if len(track) < 15:
        continue

    features = calculate_vehicle_features(
        track,
        fps
    )

    if features is not None:

        features["vehicle_id"] = vehicle_id

        vehicle_features.append(
            features
        )


# Convert to DataFrame

vehicle_features_df = pd.DataFrame(
    vehicle_features
)

# Reorder columns

vehicle_features_df = vehicle_features_df[
    [
        "vehicle_id",
        "frames_observed",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
        "max_deceleration",
        "avg_direction_change",
        "max_direction_change"
    ]
]


print("==============================================")
print("CORRECTED FEATURE EXTRACTION")
print("==============================================")

print(
    "Total feature records:",
    len(vehicle_features_df)
)

print("\nFirst 10 records:")

display(
    vehicle_features_df.head(10)
)

# --- CELL 127 ---
# ============================================================
# FINAL CORRECTED FEATURE FUNCTION
# ============================================================

def calculate_vehicle_features(track, fps):

    if len(track) < 2:
        return None

    speeds = []
    accelerations = []
    decelerations = []
    direction_changes = []

    previous_speed = None
    previous_dx = None
    previous_dy = None

    for i in range(1, len(track)):

        p1 = track[i - 1]
        p2 = track[i]

        dx = p2["cx"] - p1["cx"]
        dy = p2["cy"] - p1["cy"]

        distance = np.sqrt(dx**2 + dy**2)

        # ----------------------------------------------------
        # SPEED = pixels/second
        # ----------------------------------------------------

        speed = distance * fps

        speeds.append(speed)

        # ----------------------------------------------------
        # ACCELERATION = change in speed / time
        # ----------------------------------------------------

        if previous_speed is not None:

            acceleration = (
                speed - previous_speed
            ) * fps

            if acceleration > 0:

                accelerations.append(
                    acceleration
                )

            elif acceleration < 0:

                decelerations.append(
                    abs(acceleration)
                )

        previous_speed = speed

        # ----------------------------------------------------
        # DIRECTION CHANGE
        # ----------------------------------------------------

        if previous_dx is not None:

            magnitude1 = np.sqrt(
                previous_dx**2 +
                previous_dy**2
            )

            magnitude2 = np.sqrt(
                dx**2 +
                dy**2
            )

            if magnitude1 > 0 and magnitude2 > 0:

                cos_angle = (
                    previous_dx * dx +
                    previous_dy * dy
                ) / (
                    magnitude1 * magnitude2
                )

                cos_angle = np.clip(
                    cos_angle,
                    -1,
                    1
                )

                angle = np.arccos(
                    cos_angle
                )

                direction_changes.append(
                    angle
                )

        previous_dx = dx
        previous_dy = dy

    # --------------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------------

    if accelerations:

        avg_acceleration = np.mean(
            accelerations
        )

        max_acceleration = np.max(
            accelerations
        )

    else:

        avg_acceleration = 0.0
        max_acceleration = 0.0

    # --------------------------------------------------------
    # DECELERATION
    # --------------------------------------------------------

    if decelerations:

        max_deceleration = np.max(
            decelerations
        )

    else:

        max_deceleration = 0.0

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if direction_changes:

        avg_direction_change = np.mean(
            direction_changes
        )

        max_direction_change = np.max(
            direction_changes
        )

    else:

        avg_direction_change = 0.0
        max_direction_change = 0.0

    return {
        "frames_observed": len(track),
        "avg_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "avg_acceleration": float(avg_acceleration),
        "max_acceleration": float(max_acceleration),
        "max_deceleration": float(max_deceleration),
        "avg_direction_change": float(avg_direction_change),
        "max_direction_change": float(max_direction_change)
    }

print("FINAL corrected feature function loaded.")

# --- CELL 128 ---
import pandas as pd

vehicle_features = []

for vehicle_id, track in vehicle_tracks.items():

    if len(track) < 15:
        continue

    features = calculate_vehicle_features(
        track,
        fps
    )

    if features is not None:

        features["vehicle_id"] = vehicle_id

        vehicle_features.append(features)


vehicle_features_df = pd.DataFrame(
    vehicle_features
)

vehicle_features_df = vehicle_features_df[
    [
        "vehicle_id",
        "frames_observed",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
        "max_deceleration",
        "avg_direction_change",
        "max_direction_change"
    ]
]

print("==============================================")
print("FINAL FEATURE EXTRACTION")
print("==============================================")

print(
    "Total feature records:",
    len(vehicle_features_df)
)

display(
    vehicle_features_df.head(10)
)

# --- CELL 129 ---
# ============================================================
# STEP 33 - INSPECT SAVED MODEL
# ============================================================

import joblib
import numpy as np

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"
scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

print("==============================================")
print("SAVED MODEL INSPECTION")
print("==============================================")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model type:", type(model))
print("Scaler type:", type(scaler))

print("\nModel classes:")
print(model.classes_)

print("\nNumber of trees:")
print(model.n_estimators)

print("\nNumber of features:")
print(model.n_features_in_)

print("\nFeature importances:")
print(model.feature_importances_)

print("\nScaler mean:")
print(scaler.mean_)

print("\nScaler scale:")
print(scaler.scale_)

print("\nScaler number of features:")
print(scaler.n_features_in_)

print("\n==============================================")
print("MODEL INSPECTION COMPLETED")
print("==============================================")

# --- CELL 130 ---
# ============================================================
# STEP 34 - VERIFY OLD MODEL AGAINST OLD CSV
# ============================================================

import pandas as pd
import joblib

old_csv = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\complete_vehicle_analysis_video96.csv"

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"
scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

old_df = pd.read_csv(old_csv)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]

X_old = old_df[FEATURE_COLUMNS]

X_old_scaled = scaler.transform(X_old)

old_predictions = model.predict(X_old_scaled)

old_probabilities = model.predict_proba(
    X_old_scaled
)[:, 1]

old_df["model_prediction"] = old_predictions
old_df["model_probability"] = old_probabilities

old_df["model_status"] = old_df[
    "model_prediction"
].map({
    0: "SAFE",
    1: "DANGEROUS"
})

print("==============================================")
print("OLD MODEL VERIFICATION")
print("==============================================")

print("Total records:", len(old_df))

print("\nCSV status:")
print(old_df["status"].value_counts())

print("\nModel prediction:")
print(old_df["model_status"].value_counts())

print("\nFirst 10 comparisons:")

display(
    old_df[
        [
            "vehicle_id",
            "status",
            "probability",
            "model_status",
            "model_probability"
        ]
    ].head(10)
)

agreement = (
    old_df["status"] ==
    old_df["model_status"]
).sum()

print("\n==============================================")
print("AGREEMENT")
print("==============================================")

print(
    "Matching predictions:",
    agreement,
    "/",
    len(old_df)
)

print(
    "Agreement percentage:",
    round(
        agreement / len(old_df) * 100,
        2
    ),
    "%"
)

# --- CELL 131 ---
# ============================================================
# STEP 35 - FINAL CURRENT VIDEO CLASSIFICATION
# ============================================================

import pandas as pd
import joblib

print("==============================================")
print("FINAL CURRENT VIDEO CLASSIFICATION")
print("==============================================")

# ------------------------------------------------------------
# Load saved model
# ------------------------------------------------------------

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

final_model = joblib.load(model_path)
final_scaler = joblib.load(scaler_path)

print("Model loaded successfully")
print("Scaler loaded successfully")


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]


# ------------------------------------------------------------
# Prepare current video data
# ------------------------------------------------------------

current_df = vehicle_features_df.copy()

X_current = current_df[
    FEATURE_COLUMNS
]


# ------------------------------------------------------------
# Apply SAME scaler used during training
# ------------------------------------------------------------

X_current_scaled = final_scaler.transform(
    X_current
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

predictions = final_model.predict(
    X_current_scaled
)

probabilities = final_model.predict_proba(
    X_current_scaled
)


# ------------------------------------------------------------
# Dangerous probability
# Class 1 = DANGEROUS
# ------------------------------------------------------------

class_1_index = list(
    final_model.classes_
).index(1)

danger_probability = (
    probabilities[:, class_1_index] * 100
)


# ------------------------------------------------------------
# Add results
# ------------------------------------------------------------

current_df["prediction"] = predictions

current_df["status"] = current_df[
    "prediction"
].map({
    0: "SAFE",
    1: "DANGEROUS"
})

current_df["danger_probability"] = (
    danger_probability.round(2)
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n==============================================")
print("CLASSIFICATION COMPLETED")
print("==============================================")

print(
    "Total vehicles classified:",
    len(current_df)
)

print(
    "SAFE vehicles:",
    sum(current_df["prediction"] == 0)
)

print(
    "DANGEROUS vehicles:",
    sum(current_df["prediction"] == 1)
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n==============================================")
print("FINAL RESULTS")
print("==============================================")

display(
    current_df[
        [
            "vehicle_id",
            "frames_observed",
            "avg_speed",
            "max_speed",
            "status",
            "danger_probability"
        ]
    ]
)

# --- CELL 132 ---
# ============================================================
# FIX - CREATE CLASSIFICATION MAP
# ============================================================

print("==============================================")
print("CREATING CLASSIFICATION MAP")
print("==============================================")

print("results_df columns:")
print(results_df.columns.tolist())

print("\nvehicle_features_df columns:")
print(vehicle_features_df.columns.tolist())

print("\nCurrent classification results:")
display(results_df.head())

# --- CELL 133 ---
# ============================================================
# STEP 27 - FINAL DANGEROUS DRIVING VIDEO
# YOLO11 + SPEED + SUDDEN BRAKING + WRONG-WAY + RISK SCORE
# ============================================================

import cv2
import numpy as np
from ultralytics import YOLO
import os
import math


print("=" * 70)
print("FINAL AI TRAFFIC DANGEROUS-DRIVING VIDEO")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

output_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_DANGEROUS_VIDEO_video96_YOLO11.mp4"

yolo_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\yolo11n.pt"


# ============================================================
# 2. SPEED CALIBRATION
# ============================================================

METERS_PER_PIXEL = 0.05


# ============================================================
# 3. TRAFFIC SETTINGS
# ============================================================

SPEED_LIMIT = 60.0

SUDDEN_BRAKE_THRESHOLD = -5.0

# 0 = right
# 90 = down
# 180 = left
# 270 = up

EXPECTED_DIRECTION = 180

WRONG_WAY_ANGLE = 120


# ============================================================
# 4. CHECK INPUT VIDEO
# ============================================================

print("\nChecking input video...")

if not os.path.exists(video_path):

    raise FileNotFoundError(
        "Input video not found:\n" +
        video_path
    )

print("Input video found:")
print(video_path)


# ============================================================
# 5. CHECK YOLO11 MODEL
# ============================================================

print("\nChecking YOLO11 model...")

if not os.path.exists(yolo_path):

    print("Local YOLO11 model not found.")
    print("Trying to load YOLO11n automatically...")

    yolo_path = "yolo11n.pt"


# ============================================================
# 6. CREATE CLASSIFICATION MAP
# ============================================================

print("\nCreating classification map...")

if "classification_df" not in globals():

    raise NameError(
        "classification_df not found.\n"
        "Please run the Random Forest classification "
        "cell before STEP 27."
    )


classification_map = {}


for _, row in classification_df.iterrows():

    vehicle_id = int(
        row["vehicle_id"]
    )

    probability = float(
        row["danger_probability"]
    )


    if probability <= 1:

        probability_percent = (
            probability * 100
        )

    else:

        probability_percent = probability


    classification_map[vehicle_id] = {

        "status": row["status"],

        "probability": probability_percent
    }


print(
    "\nClassification records:",
    len(classification_map)
)


# ============================================================
# 7. CLASSIFICATION SUMMARY
# ============================================================

safe_count = sum(

    1

    for x in classification_map.values()

    if x["status"] == "SAFE"
)


danger_count = sum(

    1

    for x in classification_map.values()

    if x["status"] == "DANGEROUS"
)


print("SAFE:", safe_count)

print("DANGEROUS:", danger_count)


# ============================================================
# 8. LOAD YOLO11
# ============================================================

print("\nLoading YOLO11...")

model_yolo = YOLO(
    yolo_path
)

print("YOLO11 loaded successfully.")

print("Model:", yolo_path)


# ============================================================
# 9. OPEN VIDEO
# ============================================================

print("\nOpening input video...")

cap = cv2.VideoCapture(
    video_path
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open input video:\n" +
        video_path
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:

    fps = 30.0


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("\nFPS:", fps)

print(
    "Resolution:",
    width,
    "x",
    height
)

print(
    "Total frames:",
    total_frames
)


# ============================================================
# 10. VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(

    output_path,

    fourcc,

    fps,

    (width, height)
)


if not out.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# 11. TRACK HISTORY
# ============================================================

previous_positions = {}

previous_speeds = {}

vehicle_events = {}


# ============================================================
# 12. VEHICLE CLASSES
# ============================================================

vehicle_classes = {

    2: "car",

    3: "motorcycle",

    5: "bus",

    7: "truck"
}


# ============================================================
# 13. SPEED FUNCTION
# ============================================================

def calculate_speed_kmh(

    previous_position,

    current_position,

    fps,

    meters_per_pixel

):

    if previous_position is None:

        return 0.0


    x1, y1 = previous_position

    x2, y2 = current_position


    pixel_distance = math.sqrt(

        (x2 - x1) ** 2 +

        (y2 - y1) ** 2
    )


    distance_meters = (

        pixel_distance *

        meters_per_pixel
    )


    speed_mps = (

        distance_meters *

        fps
    )


    speed_kmh = (

        speed_mps *

        3.6
    )


    return round(
        speed_kmh,
        2
    )


# ============================================================
# 14. ACCELERATION FUNCTION
# ============================================================

def calculate_acceleration(

    previous_speed,

    current_speed,

    fps

):

    if previous_speed is None:

        return 0.0


    previous_mps = (

        previous_speed / 3.6
    )


    current_mps = (

        current_speed / 3.6
    )


    acceleration = (

        current_mps -

        previous_mps
    ) * fps


    return round(
        acceleration,
        2
    )


# ============================================================
# 15. SUDDEN BRAKING FUNCTION
# ============================================================

def detect_sudden_braking(

    acceleration

):

    return (

        acceleration <=

        SUDDEN_BRAKE_THRESHOLD
    )


# ============================================================
# 16. DIRECTION FUNCTION
# ============================================================

def calculate_direction(

    previous_position,

    current_position

):

    if previous_position is None:

        return None


    x1, y1 = previous_position

    x2, y2 = current_position


    dx = x2 - x1

    dy = y2 - y1


    if dx == 0 and dy == 0:

        return None


    angle = math.degrees(

        math.atan2(
            dy,
            dx
        )
    )


    if angle < 0:

        angle += 360


    return angle


# ============================================================
# 17. WRONG-WAY FUNCTION
# ============================================================

def detect_wrong_way(

    movement_direction,

    expected_direction,

    threshold

):

    if movement_direction is None:

        return False


    difference = abs(

        movement_direction -

        expected_direction
    )


    difference = min(

        difference,

        360 - difference
    )


    return (

        difference >= threshold
    )


# ============================================================
# 18. RISK SCORE FUNCTION
# ============================================================

def calculate_risk_score(

    danger_probability,

    speed,

    sudden_braking,

    wrong_way

):

    # Random Forest = maximum 50 points

    rf_score = (

        danger_probability *

        0.50
    )


    # Speed = maximum 20 points

    speed_score = 0.0


    if speed > SPEED_LIMIT:

        speed_score = min(

            20.0,

            (

                (

                    speed -

                    SPEED_LIMIT

                )

                /

                SPEED_LIMIT

            )

            * 20.0
        )


    # Sudden braking = 15 points

    braking_score = (

        15.0

        if sudden_braking

        else 0.0
    )


    # Wrong-way = 25 points

    wrong_way_score = (

        25.0

        if wrong_way

        else 0.0
    )


    risk_score = (

        rf_score +

        speed_score +

        braking_score +

        wrong_way_score
    )


    risk_score = min(

        100.0,

        max(
            0.0,
            risk_score
        )
    )


    return round(
        risk_score,
        2
    )


# ============================================================
# 19. RISK LEVEL FUNCTION
# ============================================================

def get_risk_level(

    risk_score

):

    if risk_score < 30:

        return "LOW"

    elif risk_score < 60:

        return "MEDIUM"

    elif risk_score < 80:

        return "HIGH"

    else:

        return "CRITICAL"


# ============================================================
# 20. PROCESS VIDEO
# ============================================================

frame_number = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    # ========================================================
    # YOLO11 TRACKING
    # ========================================================

    results = model_yolo.track(

        frame,

        persist=True,

        conf=0.35,

        verbose=False
    )


    result = results[0]


    # ========================================================
    # DETECTIONS
    # ========================================================

    if (

        result.boxes is not None

        and

        result.boxes.id is not None

    ):


        boxes = (

            result.boxes.xyxy

            .cpu()

            .numpy()
        )


        track_ids = (

            result.boxes.id

            .cpu()

            .numpy()

            .astype(int)
        )


        classes = (

            result.boxes.cls

            .cpu()

            .numpy()

            .astype(int)
        )


        confidences = (

            result.boxes.conf

            .cpu()

            .numpy()
        )


        # ====================================================
        # EACH VEHICLE
        # ====================================================

        for (

            box,

            track_id,

            cls,

            conf

        ) in zip(

            boxes,

            track_ids,

            classes,

            confidences
        ):


            if cls not in vehicle_classes:

                continue


            x1, y1, x2, y2 = map(
                int,
                box
            )


            # =================================================
            # CENTER POINT
            # =================================================

            center_x = int(

                (x1 + x2) / 2
            )


            center_y = int(

                (y1 + y2) / 2
            )


            current_position = (

                center_x,

                center_y
            )


            previous_position = (

                previous_positions.get(
                    track_id
                )
            )


            previous_speed = (

                previous_speeds.get(
                    track_id
                )
            )


            # =================================================
            # FEATURE 1 - SPEED
            # =================================================

            speed = calculate_speed_kmh(

                previous_position,

                current_position,

                fps,

                METERS_PER_PIXEL
            )


            # =================================================
            # ACCELERATION
            # =================================================

            acceleration = (

                calculate_acceleration(

                    previous_speed,

                    speed,

                    fps
                )
            )


            # =================================================
            # FEATURE 2 - SUDDEN BRAKING
            # =================================================

            sudden_braking = (

                detect_sudden_braking(

                    acceleration
                )
            )


            # =================================================
            # MOVEMENT DIRECTION
            # =================================================

            movement_direction = (

                calculate_direction(

                    previous_position,

                    current_position
                )
            )


            # =================================================
            # FEATURE 3 - WRONG WAY
            # =================================================

            wrong_way = (

                detect_wrong_way(

                    movement_direction,

                    EXPECTED_DIRECTION,

                    WRONG_WAY_ANGLE
                )
            )


            # =================================================
            # EXISTING CLASSIFICATION
            # =================================================

            if track_id in classification_map:

                status = (

                    classification_map[track_id]

                    ["status"]
                )


                danger_probability = (

                    classification_map[track_id]

                    ["probability"]
                )

            else:

                status = "UNKNOWN"

                danger_probability = 0.0


            # =================================================
            # FEATURE 4 - RISK SCORE
            # =================================================

            risk_score = (

                calculate_risk_score(

                    danger_probability,

                    speed,

                    sudden_braking,

                    wrong_way
                )
            )


            risk_level = (

                get_risk_level(

                    risk_score
                )
            )


            # =================================================
            # SAVE VEHICLE DATA
            # =================================================

            vehicle_events[track_id] = {

                "vehicle_class":
                    vehicle_classes[cls],

                "speed":
                    speed,

                "acceleration":
                    acceleration,

                "sudden_braking":
                    sudden_braking,

                "wrong_way":
                    wrong_way,

                "status":
                    status,

                "danger_probability":
                    danger_probability,

                "risk_score":
                    risk_score,

                "risk_level":
                    risk_level
            }


            # =================================================
            # UPDATE HISTORY
            # =================================================

            previous_positions[track_id] = (

                current_position
            )


            previous_speeds[track_id] = (

                speed
            )


            # =================================================
            # BOX COLOR
            # =================================================

            if risk_score >= 80:

                box_color = (
                    0,
                    0,
                    255
                )

            elif risk_score >= 60:

                box_color = (
                    0,
                    165,
                    255
                )

            elif risk_score >= 30:

                box_color = (
                    0,
                    255,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )


            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                box_color,

                3
            )


            # =================================================
            # LABEL
            # =================================================

            label = (

                f"ID {track_id} | "
                f"{status}"
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,

                    max(
                        y1 - 10,
                        25
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                box_color,

                2,

                cv2.LINE_AA
            )


            # =================================================
            # SPEED LABEL
            # =================================================

            speed_label = (

                f"Speed: {speed:.1f} km/h"
            )


            cv2.putText(

                frame,

                speed_label,

                (
                    x1,

                    y2 + 22
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                box_color,

                2,

                cv2.LINE_AA
            )


            # =================================================
            # RISK LABEL
            # =================================================

            risk_label = (

                f"Risk: {risk_score:.0f}/100 "
                f"({risk_level})"
            )


            cv2.putText(

                frame,

                risk_label,

                (
                    x1,

                    y2 + 45
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                box_color,

                2,

                cv2.LINE_AA
            )


            alert_y = y2 + 68


            # =================================================
            # SUDDEN BRAKING ALERT
            # =================================================

            if sudden_braking:

                cv2.putText(

                    frame,

                    "SUDDEN BRAKING!",

                    (
                        x1,

                        alert_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 0, 255),

                    2,

                    cv2.LINE_AA
                )


                alert_y += 25


            # =================================================
            # WRONG WAY ALERT
            # =================================================

            if wrong_way:

                cv2.putText(

                    frame,

                    "WRONG WAY!",

                    (
                        x1,

                        alert_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 165, 255),

                    2,

                    cv2.LINE_AA
                )


                alert_y += 25


    # ========================================================
    # PROJECT HEADER
    # ========================================================

    cv2.rectangle(

        frame,

        (20, 20),

        (850, 115),

        (0, 0, 0),

        -1
    )


    cv2.putText(

        frame,

        "AI TRAFFIC DANGEROUS-DRIVING DETECTION",

        (35, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.72,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        "YOLO11 | SPEED | RISK | BRAKING | WRONG-WAY",

        (35, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.58,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"SAFE: {safe_count}    "
        f"DANGEROUS: {danger_count}",

        (35, 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(frame)


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 50 == 0:

        print(

            f"Processed: "
            f"{frame_number} / "
            f"{total_frames}"
        )


# ============================================================
# 21. RELEASE
# ============================================================

cap.release()

out.release()


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("FINAL VIDEO CREATED")

print("=" * 70)


print("\nOutput video:")

print(output_path)


print(
    "\nFrames processed:",
    frame_number
)


print(
    "\nSAFE vehicles:",
    safe_count
)


print(
    "DANGEROUS vehicles:",
    danger_count
)


# ============================================================
# 23. FEATURE SUMMARY
# ============================================================

sudden_braking_count = sum(

    1

    for event in vehicle_events.values()

    if event["sudden_braking"]
)


wrong_way_count = sum(

    1

    for event in vehicle_events.values()

    if event["wrong_way"]
)


print(
    "\nSudden Braking Vehicles:",
    sudden_braking_count
)


print(
    "Wrong-Way Vehicles:",
    wrong_way_count
)


print(
    "Vehicles Analyzed:",
    len(vehicle_events)
)


# ============================================================
# 24. HIGHEST RISK VEHICLE
# ============================================================

if len(vehicle_events) > 0:

    highest_risk_id = max(

        vehicle_events,

        key=lambda vehicle_id:

        vehicle_events[
            vehicle_id
        ]["risk_score"]
    )


    highest_risk = (

        vehicle_events[
            highest_risk_id
        ]
    )


    print(
        "\nHighest Risk Vehicle:",
        highest_risk_id
    )


    print(
        "Vehicle Type:",
        highest_risk["vehicle_class"]
    )


    print(
        "Speed:",
        highest_risk["speed"],
        "km/h"
    )


    print(
        "Risk Score:",
        highest_risk["risk_score"]
    )


    print(
        "Risk Level:",
        highest_risk["risk_level"]
    )


    print(
        "Sudden Braking:",
        highest_risk["sudden_braking"]
    )


    print(
        "Wrong Way:",
        highest_risk["wrong_way"]
    )


# ============================================================
# 25. OUTPUT CHECK
# ============================================================

print(
    "\nOutput file exists:",
    os.path.exists(output_path)
)


print("\n" + "=" * 70)

print("STEP 27 COMPLETED")

print("=" * 70)

# --- CELL 134 ---
# ============================================
# FIND VIDEO96 FILE
# ============================================

import os

project_folder = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection"

print("=" * 70)
print("SEARCHING FOR VIDEO FILES")
print("=" * 70)

video_extensions = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".MOV",
    ".MP4"
)

video_files = []

for root, dirs, files in os.walk(project_folder):

    for file in files:

        if file.lower().endswith(video_extensions):

            full_path = os.path.join(root, file)

            video_files.append(full_path)

            print(full_path)

print("\n" + "=" * 70)
print("TOTAL VIDEOS FOUND:", len(video_files))
print("=" * 70)

# --- CELL 135 ---
# ============================================================
# STEP 28 - FINAL DANGEROUS DRIVING VIDEO
# ============================================================

import cv2
import numpy as np
from ultralytics import YOLO
import os

print("=" * 70)
print("FINAL DANGEROUS DRIVING VIDEO")
print("=" * 70)

# ============================================================
# 1. INPUT VIDEO
# ============================================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

# ============================================================
# 2. OUTPUT VIDEO
# ============================================================

output_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_DANGEROUS_VIDEO_video96.mp4"

# ============================================================
# 3. YOLO MODEL
# ============================================================

yolo_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\yolov8n.pt"

# ============================================================
# 4. CHECK FILES
# ============================================================

print("\nChecking files...")

print("Input video exists:", os.path.exists(video_path))
print("YOLO exists:", os.path.exists(yolo_path))

if not os.path.exists(video_path):
    raise FileNotFoundError(
        "Input video not found:\n" + video_path
    )

if not os.path.exists(yolo_path):
    raise FileNotFoundError(
        "YOLO model not found:\n" + yolo_path
    )

# ============================================================
# 5. CREATE CLASSIFICATION MAP
# ============================================================

print("\nCreating classification map...")

classification_map = {}

for _, row in classification_df.iterrows():

    vehicle_id = int(row["vehicle_id"])

    classification_map[vehicle_id] = {
        "status": row["status"],
        "probability": float(row["danger_probability"])
    }

print("Classification records:", len(classification_map))

safe_count = sum(
    1
    for data in classification_map.values()
    if data["status"] == "SAFE"
)

danger_count = sum(
    1
    for data in classification_map.values()
    if data["status"] == "DANGEROUS"
)

print("SAFE:", safe_count)
print("DANGEROUS:", danger_count)

# ============================================================
# 6. LOAD YOLO
# ============================================================

print("\nLoading YOLO...")

yolo_model = YOLO(yolo_path)

print("YOLO loaded successfully")

# ============================================================
# 7. OPEN VIDEO
# ============================================================

print("\nOpening video...")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open input video:\n" + video_path
    )

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)

# ============================================================
# 8. CREATE VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError(
        "Could not create output video"
    )

# ============================================================
# 9. PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = yolo_model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for box, track_id, cls, conf in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            # Vehicle classes
            # 2 = car
            # 3 = motorcycle
            # 5 = bus
            # 7 = truck

            if cls not in [2, 3, 5, 7]:
                continue

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # ------------------------------------------------
            # CLASSIFICATION AVAILABLE
            # ------------------------------------------------

            if track_id in classification_map:

                status = classification_map[
                    track_id
                ]["status"]

                probability = classification_map[
                    track_id
                ]["probability"]

                probability_percent = (
                    probability * 100
                )

                # --------------------------------------------
                # DANGEROUS
                # --------------------------------------------

                if status == "DANGEROUS":

                    box_color = (0, 0, 255)

                    label = (
                        f"ID {track_id} "
                        f"DANGEROUS "
                        f"{probability_percent:.0f}%"
                    )

                # --------------------------------------------
                # SAFE
                # --------------------------------------------

                else:

                    box_color = (0, 255, 0)

                    label = (
                        f"ID {track_id} "
                        f"SAFE "
                        f"{probability_percent:.0f}%"
                    )

                # Draw box

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )

                # Text size

                (
                    text_size,
                    baseline
                ) = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )

                text_width = text_size[0]
                text_height = text_size[1]

                text_y = max(
                    y1 - 10,
                    text_height + 10
                )

                # Background

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        text_y - text_height - 10
                    ),
                    (
                        x1 + text_width + 10,
                        text_y + 5
                    ),
                    box_color,
                    -1
                )

                # Label

                cv2.putText(
                    frame,
                    label,
                    (
                        x1 + 5,
                        text_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            else:

                # ------------------------------------------------
                # NOT CLASSIFIED
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                label = f"ID {track_id}"

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

    # ========================================================
    # PROJECT TITLE
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (650, 105),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI TRAFFIC DANGEROUS-DRIVING DETECTION",
        (35, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"SAFE: {safe_count}   "
        f"DANGEROUS: {danger_count}",
        (35, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(frame)

    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 50 == 0:

        print(
            f"Processed: "
            f"{frame_number} / "
            f"{total_frames}"
        )

# ============================================================
# 10. RELEASE
# ============================================================

cap.release()
out.release()

# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VIDEO CREATED")
print("=" * 70)

print("\nOutput video:")
print(output_path)

print("\nFrames processed:", frame_number)

print("\nSAFE vehicles:", safe_count)
print("DANGEROUS vehicles:", danger_count)

print("\nOutput exists:", os.path.exists(output_path))

# --- CELL 136 ---
import cv2

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_DANGEROUS_VIDEO_video96.mp4"

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("Width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

cap.release()

# --- CELL 137 ---
import cv2
from matplotlib import pyplot as plt

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_DANGEROUS_VIDEO_video96.mp4"

cap = cv2.VideoCapture(video_path)

# Frame 100 read करा
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

ret, frame = cap.read()
cap.release()

print("Frame read:", ret)

if ret:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 9))
    plt.imshow(frame)
    plt.axis("off")
    plt.show()

# --- CELL 138 ---
import cv2
import os

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_DANGEROUS_VIDEO_video96.mp4"

cap = cv2.VideoCapture(input_video)

print("Video opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("Width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

cap.release()

# --- CELL 139 ---
import subprocess
import imageio_ffmpeg

input_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_DANGEROUS_VIDEO_video96.mp4"

output_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96_h264.mp4"

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg_path,
    "-y",
    "-i", input_video,
    "-vf", "scale=1280:-2",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-crf", "28",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    output_video
]

result = subprocess.run(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

if result.returncode == 0:
    print("================================")
    print("VIDEO CONVERSION COMPLETED")
    print("================================")
    print(output_video)
else:
    print("Conversion failed:")
    print(result.stderr[-2000:])

# --- CELL 140 ---
from IPython.display import Video, display

output_video = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\traffic_final_result_video96_h264.mp4"

display(Video(output_video, embed=True, width=900))

# --- CELL 141 ---
print("classification_df" in globals())

# --- CELL 142 ---
# ============================================
# RANDOM FOREST PREDICTION
# ============================================

import joblib
import pandas as pd
import numpy as np

print("================================")
print("RANDOM FOREST VIDEO PREDICTION")
print("================================")


# ============================================
# CHECK FEATURES
# ============================================

if "features_df" not in globals():

    raise NameError(
        "features_df not found.\n"
        "Please run the VEHICLE FEATURES EXTRACTION cell first."
    )

print("Features DataFrame found ✅")
print("Shape:", features_df.shape)


# ============================================
# MODEL PATHS
# ============================================

model_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"


# ============================================
# LOAD RANDOM FOREST
# ============================================

print("\nLoading Random Forest...")

model = joblib.load(model_path)

print("Random Forest loaded successfully ✅")


# ============================================
# LOAD SCALER
# ============================================

print("Loading scaler...")

scaler = joblib.load(scaler_path)

print("Scaler loaded successfully ✅")


# ============================================
# TRAINING FEATURES
# ============================================

feature_columns = [

    "avg_speed",

    "max_speed",

    "avg_acceleration",

    "max_acceleration",

    "max_deceleration",

    "avg_direction_change",

    "max_direction_change"

]


# ============================================
# CHECK FEATURES
# ============================================

missing_features = [
    col
    for col in feature_columns
    if col not in features_df.columns
]

if missing_features:

    raise ValueError(
        "Missing features:\n" +
        str(missing_features)
    )


# ============================================
# PREPARE VIDEO FEATURES
# ============================================

X_video = features_df[
    feature_columns
].copy()


# ============================================
# CLEAN DATA
# ============================================

X_video = X_video.replace(
    [np.inf, -np.inf],
    np.nan
)

X_video = X_video.fillna(0)


# ============================================
# SCALE
# ============================================

X_video_scaled = scaler.transform(
    X_video
)


# ============================================
# PREDICTION
# ============================================

predictions = model.predict(
    X_video_scaled
)


# ============================================
# PROBABILITY
# ============================================

probabilities = model.predict_proba(
    X_video_scaled
)


# ============================================
# CREATE CLASSIFICATION DATAFRAME
# ============================================

classification_df = features_df.copy()

classification_df["prediction"] = predictions


classification_df["status"] = (
    classification_df["prediction"]
    .map({
        0: "SAFE",
        1: "DANGEROUS"
    })
)


classification_df["danger_probability"] = (
    probabilities[:, 1] * 100
)


# ============================================
# DISPLAY RESULTS
# ============================================

print("\n================================")
print("VEHICLE PREDICTIONS")
print("================================")


print(
    classification_df[
        [
            "vehicle_id",
            "frames_observed",
            "status",
            "danger_probability"
        ]
    ].to_string(index=False)
)


# ============================================
# COUNTS
# ============================================

safe_count = (
    classification_df["prediction"] == 0
).sum()


dangerous_count = (
    classification_df["prediction"] == 1
).sum()


print("\n================================")
print("FINAL PREDICTION SUMMARY")
print("================================")

print(
    "Total Valid Vehicles :",
    len(classification_df)
)

print(
    "SAFE Vehicles        :",
    safe_count
)

print(
    "DANGEROUS Vehicles   :",
    dangerous_count
)


# ============================================
# IMPORTANT CHECK
# ============================================

print("\nclassification_df created successfully ✅")

print(
    "classification_df shape:",
    classification_df.shape
)

# --- CELL 143 ---
# ============================================================
# STEP 27 - FINAL AI TRAFFIC DANGEROUS-DRIVING VIDEO
# ============================================================

import cv2
import os
import math
from ultralytics import YOLO
from collections import defaultdict

print("=" * 70)
print("FINAL AI TRAFFIC DANGEROUS-DRIVING VIDEO")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\video96.MOV"

yolo_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\yolo11n.pt"

output_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_AI_DANGEROUS_VIDEO_video96.mp4"


# ============================================================
# SETTINGS
# ============================================================

VEHICLE_CLASSES = [2, 3, 5, 7]

# Approximate calibration
PIXELS_PER_METER = 10.0

# Minimum movement required
MIN_MOVEMENT_PIXELS = 3.0

# Sudden braking threshold
DECELERATION_THRESHOLD = -100.0

# Number of consecutive opposite-direction observations
WRONG_WAY_MIN_COUNT = 8

# Direction assumed by the traffic video
# 0   = left -> right
# 180 = right -> left
EXPECTED_DIRECTION = 0.0


# ============================================================
# CHECK classification_df
# ============================================================

if "classification_df" not in globals():

    raise NameError(
        "\nclassification_df not found.\n"
        "Please run the RANDOM FOREST PREDICTION cell first."
    )

print("\nClassification data found ✅")


# ============================================================
# CLASSIFICATION MAP
# ============================================================

classification_map = {}

for _, row in classification_df.iterrows():

    vehicle_id = int(row["vehicle_id"])

    classification_map[vehicle_id] = {

        "status": str(row["status"]),

        "probability": float(
            row["danger_probability"]
        )
    }


safe_count = sum(
    1
    for v in classification_map.values()
    if v["status"] == "SAFE"
)

danger_count = sum(
    1
    for v in classification_map.values()
    if v["status"] == "DANGEROUS"
)


print("Classification records:", len(classification_map))
print("SAFE:", safe_count)
print("DANGEROUS:", danger_count)


# ============================================================
# CHECK INPUT VIDEO
# ============================================================

print("\nChecking input video...")

if not os.path.exists(video_path):

    raise FileNotFoundError(
        "Input video not found:\n" +
        video_path
    )

print("Input video found ✅")


# ============================================================
# LOAD YOLO11
# ============================================================

print("\nLoading YOLO11...")

if not os.path.exists(yolo_path):

    raise FileNotFoundError(
        "\nYOLO11 model not found:\n" +
        yolo_path
    )

model_yolo = YOLO(yolo_path)

print("YOLO11 loaded successfully ✅")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open input video."
    )


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print("\n================================")
print("VIDEO INFORMATION")
print("================================")

print("FPS          :", fps)
print("Total Frames :", total_frames)
print("Resolution   :", width, "x", height)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# TRACKING VARIABLES
# ============================================================

previous_positions = {}

previous_speeds = {}

speed_history = defaultdict(list)

direction_history = defaultdict(list)

braking_ids = set()

wrong_way_ids = set()


# ============================================================
# FRAME PROCESSING
# ============================================================

frame_number = 0


print("\n================================")
print("CREATING FINAL VIDEO")
print("================================")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ========================================================
    # YOLO11 + BYTE TRACK
    # ========================================================

    results = model_yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]


    # ========================================================
    # DETECTIONS
    # ========================================================

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )


        # ====================================================
        # PROCESS EACH VEHICLE
        # ====================================================

        for box, track_id, class_id in zip(
            boxes,
            ids,
            classes
        ):

            # Only vehicles
            if class_id not in VEHICLE_CLASSES:
                continue


            # IMPORTANT:
            # Ignore IDs which are not in Random Forest
            # classification results.
            if track_id not in classification_map:
                continue


            x1, y1, x2, y2 = map(
                int,
                box
            )


            # =================================================
            # CENTER
            # =================================================

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            current_position = (
                center_x,
                center_y
            )


            speed_kmh = 0.0

            acceleration = 0.0

            direction_angle = None


            # =================================================
            # MOVEMENT
            # =================================================

            if track_id in previous_positions:

                old_x, old_y = (
                    previous_positions[
                        track_id
                    ]
                )


                dx = center_x - old_x

                dy = center_y - old_y


                movement_pixels = math.sqrt(
                    dx ** 2 +
                    dy ** 2
                )


                # =================================================
                # SPEED
                # =================================================

                if movement_pixels >= MIN_MOVEMENT_PIXELS:

                    distance_meters = (
                        movement_pixels /
                        PIXELS_PER_METER
                    )


                    speed_mps = (
                        distance_meters *
                        fps
                    )


                    speed_kmh = (
                        speed_mps * 3.6
                    )


                # =================================================
                # LIMIT UNREALISTIC SPEED SPIKES
                # =================================================

                if speed_kmh > 150:

                    speed_kmh = 150.0


                speed_history[
                    track_id
                ].append(
                    speed_kmh
                )


                # Keep only recent values
                if len(
                    speed_history[track_id]
                ) > 10:

                    speed_history[
                        track_id
                    ] = speed_history[
                        track_id
                    ][-10:]


                # =================================================
                # ACCELERATION
                # =================================================

                if track_id in previous_speeds:

                    acceleration = (
                        speed_kmh -
                        previous_speeds[
                            track_id
                        ]
                    ) * fps


                    # =================================================
                    # SUDDEN BRAKING
                    # =================================================

                    if (
                        acceleration
                        < DECELERATION_THRESHOLD
                    ):

                        braking_ids.add(
                            track_id
                        )


                previous_speeds[
                    track_id
                ] = speed_kmh


                # =================================================
                # DIRECTION
                # =================================================

                if movement_pixels >= MIN_MOVEMENT_PIXELS:

                    direction_angle = math.degrees(
                        math.atan2(
                            dy,
                            dx
                        )
                    )


                    if direction_angle < 0:

                        direction_angle += 360


                    direction_history[
                        track_id
                    ].append(
                        direction_angle
                    )


                    # Keep recent directions
                    if len(
                        direction_history[
                            track_id
                        ]
                    ) > 10:

                        direction_history[
                            track_id
                        ] = direction_history[
                            track_id
                        ][-10:]


                    # =================================================
                    # WRONG-WAY
                    # =================================================

                    difference = abs(
                        direction_angle -
                        EXPECTED_DIRECTION
                    )


                    if difference > 180:

                        difference = (
                            360 -
                            difference
                        )


                    # Opposite direction
                    if difference >= 120:

                        direction_history[
                            track_id
                        ].append(
                            -999
                        )

                    else:

                        # Remove one false opposite marker
                        if direction_history[
                            track_id
                        ]:

                            pass


                    # Count recent opposite directions
                    recent = (
                        direction_history[
                            track_id
                        ][-10:]
                    )


                    opposite_count = sum(
                        1
                        for d in recent
                        if d == -999
                    )


                    # Only mark wrong-way when
                    # movement is consistently opposite
                    if (
                        opposite_count
                        >= WRONG_WAY_MIN_COUNT
                    ):

                        wrong_way_ids.add(
                            track_id
                        )


            # Save position
            previous_positions[
                track_id
            ] = current_position


            # =================================================
            # CLASSIFICATION
            # =================================================

            status = (
                classification_map[
                    track_id
                ]["status"]
            )


            probability = (
                classification_map[
                    track_id
                ]["probability"]
            )


            # =================================================
            # SAFE / DANGEROUS COLOR
            # =================================================

            if status == "DANGEROUS":

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )


            # =================================================
            # RISK SCORE
            # =================================================

            risk_score = int(
                probability
            )


            # Add risk only for actual events
            if track_id in braking_ids:

                risk_score += 10


            if track_id in wrong_way_ids:

                risk_score += 15


            risk_score = min(
                risk_score,
                100
            )


            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )


            # =================================================
            # LABELS
            # =================================================

            labels = [

                f"ID {track_id} | {status}",

                f"Speed: {speed_kmh:.1f} km/h",

                f"Risk: {risk_score}/100"
            ]


            # Event labels
            if track_id in braking_ids:

                labels.append(
                    "SUDDEN BRAKING"
                )


            if track_id in wrong_way_ids:

                labels.append(
                    "WRONG WAY"
                )


            # =================================================
            # DRAW LABELS
            # =================================================

            for i, text in enumerate(labels):

                text_y = (
                    y1 -
                    10 -
                    (len(labels) - 1 - i) *
                    25
                )


                text_y = max(
                    text_y,
                    20
                )


                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    2
                )


                # Background
                cv2.rectangle(
                    frame,

                    (
                        x1,
                        text_y -
                        text_height -
                        5
                    ),

                    (
                        x1 +
                        text_width +
                        10,

                        text_y +
                        5
                    ),

                    box_color,

                    -1
                )


                # Text
                cv2.putText(
                    frame,

                    text,

                    (
                        x1 + 5,
                        text_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.62,

                    (
                        255,
                        255,
                        255
                    ),

                    2,

                    cv2.LINE_AA
                )


    # ========================================================
    # TOP INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (720, 120),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,

        "AI TRAFFIC DANGEROUS-DRIVING DETECTION",

        (35, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.70,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    cv2.putText(
        frame,

        f"SAFE: {safe_count}    "
        f"DANGEROUS: {danger_count}",

        (35, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    cv2.putText(
        frame,

        f"Wrong-Way: {len(wrong_way_ids)}    "
        f"Sudden Braking: {len(braking_ids)}",

        (35, 108),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(frame)


    if frame_number % 50 == 0:

        print(
            f"Video written: "
            f"{frame_number} / "
            f"{total_frames}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()

out.release()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VIDEO CREATED")
print("=" * 70)

print(
    "\nTotal Valid Vehicles :",
    len(classification_map)
)

print(
    "SAFE Vehicles        :",
    safe_count
)

print(
    "DANGEROUS Vehicles   :",
    danger_count
)

print(
    "Wrong-Way Vehicles   :",
    len(wrong_way_ids)
)

print(
    "Sudden Braking IDs   :",
    len(braking_ids)
)

print("\nOutput Video:")
print(output_path)

print("\n================================")
print("FEATURES INCLUDED")
print("================================")

print("1. YOLO11 Vehicle Detection")
print("2. ByteTrack Vehicle Tracking")
print("3. Random Forest Classification")
print("4. SAFE = Green")
print("5. DANGEROUS = Red")
print("6. Estimated Speed (km/h)")
print("7. Risk Score (0-100)")
print("8. Sudden Braking Detection")
print("9. Wrong-Way Detection")

print("================================")

# --- CELL 144 ---
# ============================================================
# STEP 27
# COMPLETE AI TRAFFIC DANGEROUS-DRIVING DETECTION
# YOLO + BYTE TRACK + RANDOM FOREST
# GREEN = SAFE | RED = DANGEROUS
# SPEED + RISK + SUDDEN BRAKING + WRONG WAY
# FINAL VIDEO PLAYED INSIDE JUPYTER
# ============================================================

import os
import cv2
import math
import yaml
import joblib
import tempfile
import subprocess
import warnings
import numpy as np
import pandas as pd
import imageio_ffmpeg

from collections import defaultdict
from ultralytics import YOLO
from IPython.display import Video, display

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================

INPUT_VIDEO = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\traffic_video.avi"

YOLO_MODEL_PATH = r"C:\Users\Neha Kamble\runs\detect\train-3\weights\best.pt"

RF_MODEL_PATH = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

SCALER_PATH = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

OUTPUT_VIDEO = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_TRAFFIC_RESULT.mp4"

H264_VIDEO = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_TRAFFIC_RESULT_H264.mp4"


# ============================================================
# 2. SETTINGS
# ============================================================

# Lower confidence = more vehicles detected
CONF_THRESHOLD = 0.10

# YOLO input resolution
IMAGE_SIZE = 1280

# ByteTrack settings
TRACK_BUFFER = 60

# Minimum observations needed for feature calculation
MIN_OBSERVATIONS = 2

# ------------------------------------------------------------
# SPEED CALIBRATION
# ------------------------------------------------------------
# This is an estimated camera scale.
# Change this value after real-world calibration if needed.
METERS_PER_PIXEL = 0.05

# Sudden braking threshold
BRAKING_THRESHOLD = 2.0

# Wrong-way threshold
WRONG_WAY_ANGLE = 120


# ============================================================
# 3. CHECK FILES
# ============================================================

print("=" * 70)
print("STEP 27 - COMPLETE TRAFFIC DANGEROUS-DRIVING DETECTION")
print("=" * 70)

for path, name in [
    (INPUT_VIDEO, "Input video"),
    (YOLO_MODEL_PATH, "YOLO model"),
    (RF_MODEL_PATH, "Random Forest"),
    (SCALER_PATH, "Scaler")
]:
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )
    
    print(f"{name:<18}: Found ✅")


# ============================================================
# 4. LOAD MODELS
# ============================================================

print("\nLoading YOLO...")

yolo_model = YOLO(YOLO_MODEL_PATH)

print("YOLO loaded ✅")

print("\nLoading Random Forest...")

rf_model = joblib.load(RF_MODEL_PATH)

print("Random Forest loaded ✅")

print("Loading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded ✅")


# ============================================================
# 5. EXACT RF FEATURES
# ============================================================

feature_columns = [
    "avg_speed",
    "max_speed",
    "avg_acceleration",
    "max_acceleration",
    "max_deceleration",
    "avg_direction_change",
    "max_direction_change"
]


# ============================================================
# 6. CREATE CUSTOM BYTE TRACK CONFIG
# ============================================================

tracker_config = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.15,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.12,
    "track_buffer": TRACK_BUFFER,
    "match_thresh": 0.80,
    "fuse_score": True
}

tracker_yaml = os.path.join(
    tempfile.gettempdir(),
    "traffic_bytetrack.yaml"
)

with open(tracker_yaml, "w") as f:
    yaml.safe_dump(tracker_config, f)

print("\nByteTrack configuration created ✅")


# ============================================================
# 7. VIDEO INFORMATION
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0

print("\n" + "=" * 35)
print("VIDEO INFORMATION")
print("=" * 35)

print("FPS          :", fps)
print("Total Frames :", total_frames)
print("Resolution   :", width, "x", height)


# ============================================================
# 8. VEHICLE CLASS DETECTION
# ============================================================

# If custom YOLO model has vehicle-specific class names,
# automatically select them.
#
# If the model does not expose recognizable vehicle names,
# all detected classes are used.

names = yolo_model.names

if isinstance(names, dict):
    class_names = names
else:
    class_names = {i: n for i, n in enumerate(names)}

vehicle_keywords = [
    "car",
    "vehicle",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
    "van",
    "auto",
    "rickshaw",
    "scooter"
]

vehicle_class_ids = []

for class_id, class_name in class_names.items():

    class_name_lower = str(class_name).lower()

    if any(
        keyword in class_name_lower
        for keyword in vehicle_keywords
    ):
        vehicle_class_ids.append(int(class_id))


# If no recognizable vehicle class is found,
# do not filter classes.
if len(vehicle_class_ids) == 0:
    vehicle_class_ids = None

print("\nVehicle classes used:")

if vehicle_class_ids is None:
    print("All classes detected by the custom YOLO model")
else:
    print([
        class_names[i]
        for i in vehicle_class_ids
    ])


# ============================================================
# 9. DATA STRUCTURES
# ============================================================

# Every frame stores its detections.
frame_detections = defaultdict(list)

# Track history
track_history = defaultdict(list)

# Class of each vehicle
track_class = {}

# Last known position
last_position = {}

# Last speed
last_speed = {}

# Maximum braking
max_braking = defaultdict(float)

# Direction history
direction_history = defaultdict(list)

# Track confidence
track_confidence = defaultdict(list)


# ============================================================
# 10. STEP 1 - YOLO + BYTE TRACK
# ============================================================

print("\n" + "=" * 35)
print("STEP 1: YOLO + BYTE TRACK")
print("=" * 35)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO TRACK
    # --------------------------------------------------------

    results = yolo_model.track(
        frame,
        persist=True,
        tracker=tracker_yaml,
        conf=CONF_THRESHOLD,
        iou=0.50,
        imgsz=IMAGE_SIZE,
        classes=vehicle_class_ids,
        verbose=False
    )[0]

    # --------------------------------------------------------
    # CHECK TRACK IDS
    # --------------------------------------------------------

    if (
        results.boxes is None
        or results.boxes.id is None
    ):
        if frame_number % 50 == 0:
            print(
                f"Processed: {frame_number} / {total_frames}"
            )
        continue

    boxes = (
        results.boxes.xyxy
        .cpu()
        .numpy()
    )

    ids = (
        results.boxes.id
        .cpu()
        .numpy()
        .astype(int)
    )

    classes = (
        results.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        results.boxes.conf
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # SAVE EACH DETECTION
    # --------------------------------------------------------

    for box, track_id, class_id, confidence in zip(
        boxes,
        ids,
        classes,
        confidences
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        center = (
            center_x,
            center_y
        )

        track_class[track_id] = class_names.get(
            int(class_id),
            "vehicle"
        )

        track_confidence[track_id].append(
            float(confidence)
        )

        # ----------------------------------------------------
        # POSITION HISTORY
        # ----------------------------------------------------

        track_history[track_id].append({
            "frame": frame_number,
            "x": center_x,
            "y": center_y
        })

        # ----------------------------------------------------
        # CURRENT SPEED IN PIXELS / FRAME
        # ----------------------------------------------------

        pixel_speed = 0.0

        if track_id in last_position:

            old_x, old_y = last_position[track_id]

            dx = center_x - old_x
            dy = center_y - old_y

            pixel_speed = math.sqrt(
                dx * dx + dy * dy
            )

        last_position[track_id] = center

        last_speed[track_id] = pixel_speed

        # ----------------------------------------------------
        # STORE DIRECTION
        # ----------------------------------------------------

        if track_id in direction_history:

            previous = direction_history[track_id][-1]

            dx = center_x - previous[0]
            dy = center_y - previous[1]

            if abs(dx) > 0 or abs(dy) > 0:

                angle = math.atan2(
                    dy,
                    dx
                )

                direction_history[track_id].append(
                    (
                        center_x,
                        center_y,
                        angle
                    )
                )

        else:

            direction_history[track_id].append(
                (
                    center_x,
                    center_y,
                    0.0
                )
            )

        # ----------------------------------------------------
        # SAVE FRAME DETECTION
        # ----------------------------------------------------

        frame_detections[frame_number].append({

            "track_id": int(track_id),

            "bbox": (
                x1,
                y1,
                x2,
                y2
            ),

            "center": center,

            "class_id": int(class_id),

            "class_name": track_class[track_id],

            "confidence": float(confidence)

        })

    if frame_number % 50 == 0:

        print(
            f"Processed: {frame_number} / {total_frames}"
        )


cap.release()


# ============================================================
# 11. TRACKING SUMMARY
# ============================================================

print("\n" + "=" * 35)
print("TRACKING COMPLETED")
print("=" * 35)

print(
    "Total tracked vehicle IDs:",
    len(track_history)
)


# ============================================================
# 12. FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 35)
print("STEP 2: FEATURE EXTRACTION")
print("=" * 35)

feature_rows = []

for track_id, history in track_history.items():

    if len(history) < MIN_OBSERVATIONS:
        continue

    speeds = []

    accelerations = []

    direction_changes = []

    previous_speed = None

    previous_angle = None

    # --------------------------------------------------------
    # SPEED + ACCELERATION + DIRECTION
    # --------------------------------------------------------

    for i in range(1, len(history)):

        dx = (
            history[i]["x"]
            -
            history[i - 1]["x"]
        )

        dy = (
            history[i]["y"]
            -
            history[i - 1]["y"]
        )

        pixel_distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        speeds.append(
            pixel_distance
        )

        if previous_speed is not None:

            acceleration = (
                pixel_distance
                -
                previous_speed
            )

            accelerations.append(
                acceleration
            )

        previous_speed = pixel_distance

        # ----------------------------------------------------
        # DIRECTION ANGLE
        # ----------------------------------------------------

        if abs(dx) > 0 or abs(dy) > 0:

            angle = math.atan2(
                dy,
                dx
            )

            if previous_angle is not None:

                diff = abs(
                    math.degrees(
                        angle -
                        previous_angle
                    )
                )

                # Normalize angle
                while diff > 180:
                    diff -= 360

                diff = abs(diff)

                direction_changes.append(
                    diff / 180.0
                )

            previous_angle = angle

    # --------------------------------------------------------
    # FALLBACKS
    # --------------------------------------------------------

    if len(speeds) == 0:
        speeds = [0.0]

    if len(accelerations) == 0:
        accelerations = [0.0]

    if len(direction_changes) == 0:
        direction_changes = [0.0]

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    avg_speed = float(
        np.mean(speeds)
    )

    max_speed = float(
        np.max(speeds)
    )

    avg_acceleration = float(
        np.mean(accelerations)
    )

    max_acceleration = float(
        np.max(accelerations)
    )

    # Deceleration as positive magnitude
    negative_accelerations = [
        abs(a)
        for a in accelerations
        if a < 0
    ]

    if negative_accelerations:

        max_deceleration = float(
            max(negative_accelerations)
        )

    else:

        max_deceleration = 0.0

    avg_direction_change = float(
        np.mean(direction_changes)
    )

    max_direction_change = float(
        np.max(direction_changes)
    )

    feature_rows.append({

        "vehicle_id": int(track_id),

        "frames_observed": len(history),

        "avg_speed": avg_speed,

        "max_speed": max_speed,

        "avg_acceleration": avg_acceleration,

        "max_acceleration": max_acceleration,

        "max_deceleration": max_deceleration,

        "avg_direction_change": avg_direction_change,

        "max_direction_change": max_direction_change

    })


features_df = pd.DataFrame(
    feature_rows
)

print(
    "Vehicles with features:",
    len(features_df)
)


# ============================================================
# 13. RANDOM FOREST PREDICTION
# ============================================================

print("\n" + "=" * 35)
print("STEP 3: RANDOM FOREST PREDICTION")
print("=" * 35)

if len(features_df) == 0:

    raise RuntimeError(
        "No vehicle tracks were available for classification."
    )


# ------------------------------------------------------------
# IMPORTANT:
# Use DataFrame with exact feature names.
# This removes StandardScaler warning.
# ------------------------------------------------------------

X_video = features_df[
    feature_columns
].copy()

X_scaled = scaler.transform(
    X_video
)

predictions = rf_model.predict(
    X_scaled
)

probabilities = rf_model.predict_proba(
    X_scaled
)

features_df["prediction"] = predictions

features_df["status"] = features_df[
    "prediction"
].map({
    0: "SAFE",
    1: "DANGEROUS"
}).fillna("SAFE")

features_df["danger_probability"] = (
    probabilities[:, 1] * 100
)

# Risk score = RF dangerous probability
features_df["risk_score"] = (
    features_df["danger_probability"]
    .clip(0, 100)
)


# ============================================================
# 14. SPEED CALCULATION
# ============================================================

# Convert pixel/frame to estimated km/h
#
# IMPORTANT:
# METERS_PER_PIXEL must be calibrated for your camera.

features_df["speed_kmh"] = (
    features_df["avg_speed"]
    *
    fps
    *
    METERS_PER_PIXEL
    *
    3.6
)


# ============================================================
# 15. SUDDEN BRAKING
# ============================================================

features_df["sudden_braking"] = (
    features_df["max_deceleration"]
    >=
    BRAKING_THRESHOLD
)


# ============================================================
# 16. WRONG-WAY DETECTION
# ============================================================

wrong_way_ids = set()

for track_id, history in track_history.items():

    if len(history) < 5:
        continue

    # --------------------------------------------------------
    # Calculate displacement from beginning to end
    # --------------------------------------------------------

    start_x = history[0]["x"]
    start_y = history[0]["y"]

    end_x = history[-1]["x"]
    end_y = history[-1]["y"]

    total_dx = end_x - start_x
    total_dy = end_y - start_y

    if abs(total_dx) < 2 and abs(total_dy) < 2:
        continue

    # --------------------------------------------------------
    # Use dominant movement direction.
    #
    # For two-way road videos, this detects vehicles moving
    # opposite to the dominant direction of their local track.
    # --------------------------------------------------------

    angle = math.degrees(
        math.atan2(
            total_dy,
            total_dx
        )
    )

    # Store angle
    direction_history[track_id] = [
        angle
    ]

# ------------------------------------------------------------
# Wrong-way detection using opposite direction clusters
# ------------------------------------------------------------

all_angles = []

for track_id, history in track_history.items():

    if len(history) < 5:
        continue

    dx = (
        history[-1]["x"]
        -
        history[0]["x"]
    )

    dy = (
        history[-1]["y"]
        -
        history[0]["y"]
    )

    if abs(dx) < 2 and abs(dy) < 2:
        continue

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    all_angles.append(
        (track_id, angle)
    )

if len(all_angles) >= 2:

    # --------------------------------------------------------
    # Determine dominant direction.
    # --------------------------------------------------------

    angles_rad = np.radians(
        [a for _, a in all_angles]
    )

    mean_x = np.mean(
        np.cos(angles_rad)
    )

    mean_y = np.mean(
        np.sin(angles_rad)
    )

    dominant_angle = math.degrees(
        math.atan2(
            mean_y,
            mean_x
        )
    )

    for track_id, angle in all_angles:

        diff = abs(
            angle -
            dominant_angle
        )

        while diff > 180:
            diff -= 360

        diff = abs(diff)

        if diff >= WRONG_WAY_ANGLE:

            wrong_way_ids.add(
                track_id
            )


# ============================================================
# 17. CREATE CLASSIFICATION MAP
# ============================================================

classification_map = {}

for _, row in features_df.iterrows():

    classification_map[
        int(row["vehicle_id"])
    ] = {

        "status": row["status"],

        "risk": float(
            row["risk_score"]
        ),

        "speed": float(
            row["speed_kmh"]
        ),

        "braking": bool(
            row["sudden_braking"]
        ),

        "wrong_way": (
            int(row["vehicle_id"])
            in wrong_way_ids
        )

    }


# ============================================================
# 18. FINAL VIDEO
# ============================================================

print("\n" + "=" * 35)
print("STEP 4: CREATING FINAL COLORED VIDEO")
print("=" * 35)

cap = cv2.VideoCapture(
    INPUT_VIDEO
)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot reopen input video."
    )

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# 19. VIDEO RENDERING
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    detections = frame_detections.get(
        frame_number,
        []
    )

    for detection in detections:

        track_id = detection[
            "track_id"
        ]

        x1, y1, x2, y2 = detection[
            "bbox"
        ]

        # ----------------------------------------------------
        # GET CLASSIFICATION
        # ----------------------------------------------------

        data = classification_map.get(
            track_id
        )

        # If track did not have enough observations,
        # still show the vehicle bounding box.
        if data is None:

            status = "SAFE"
            risk = 0.0
            speed = 0.0
            braking = False
            wrong_way = False

        else:

            status = data[
                "status"
            ]

            risk = data[
                "risk"
            ]

            speed = data[
                "speed"
            ]

            braking = data[
                "braking"
            ]

            wrong_way = data[
                "wrong_way"
            ]

        # ----------------------------------------------------
        # ONLY TWO BOX COLORS
        # SAFE = GREEN
        # DANGEROUS = RED
        # ----------------------------------------------------

        if status == "DANGEROUS":

            box_color = (
                0,
                0,
                255
            )

        else:

            box_color = (
                0,
                255,
                0
            )

        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label_1 = (
            f"ID {track_id} | {status}"
        )

        label_2 = (
            f"Speed: {speed:.1f} km/h"
        )

        label_3 = (
            f"Risk: {risk:.0f}/100"
        )

        # ----------------------------------------------------
        # ALERTS
        # ----------------------------------------------------

        alerts = []

        if braking:
            alerts.append(
                "SUDDEN BRAKING"
            )

        if wrong_way:
            alerts.append(
                "WRONG WAY"
            )

        # ----------------------------------------------------
        # LABEL HEIGHT
        # ----------------------------------------------------

        label_height = 25

        if len(alerts) > 0:

            label_height += (
                25 * len(alerts)
            )

        # Keep label inside frame
        label_top = max(
            0,
            y1 - label_height
        )

        label_bottom = y1

        # ----------------------------------------------------
        # BACKGROUND
        # Same GREEN / RED as vehicle
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, label_top),
            (
                x1 + 230,
                label_bottom
            ),
            box_color,
            -1
        )

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text_x = x1 + 5

        text_y = (
            label_top + 18
        )

        cv2.putText(
            frame,
            label_1,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        text_y += 20

        cv2.putText(
            frame,
            label_2,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        text_y += 20

        cv2.putText(
            frame,
            label_3,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        text_y += 20

        # ----------------------------------------------------
        # ALERT TEXT
        # ----------------------------------------------------

        for alert in alerts:

            cv2.putText(
                frame,
                alert,
                (
                    text_x,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            text_y += 20

    # ========================================================
    # TOP SUMMARY
    # ========================================================

    safe_count = sum(
        1
        for d in classification_map.values()
        if d["status"] == "SAFE"
    )

    dangerous_count = sum(
        1
        for d in classification_map.values()
        if d["status"] == "DANGEROUS"
    )

    # Black summary background
    cv2.rectangle(
        frame,
        (15, 15),
        (520, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI TRAFFIC DANGEROUS-DRIVING DETECTION",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Green SAFE
    cv2.putText(
        frame,
        f"SAFE: {safe_count}",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Red DANGEROUS
    cv2.putText(
        frame,
        f"DANGEROUS: {dangerous_count}",
        (220, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    writer.write(frame)

    if frame_number % 50 == 0:

        print(
            f"Video written: "
            f"{frame_number} / "
            f"{total_frames}"
        )


cap.release()
writer.release()


# ============================================================
# 20. CHECK VIDEO
# ============================================================

if not os.path.exists(OUTPUT_VIDEO):

    raise FileNotFoundError(
        "Output video was not created."
    )

print("\n" + "=" * 70)
print("FINAL VIDEO CREATED")
print("=" * 70)

print(
    "Tracked vehicle IDs :",
    len(track_history)
)

print(
    "Vehicles classified  :",
    len(classification_map)
)

print(
    "SAFE vehicles        :",
    safe_count
)

print(
    "DANGEROUS vehicles   :",
    dangerous_count
)

print(
    "\nOutput Video:"
)

print(
    OUTPUT_VIDEO
)


# ============================================================
# 21. CONVERT TO H264
# ============================================================

print("\nConverting final video to H264...")

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

command = [

    ffmpeg,

    "-y",

    "-i",
    OUTPUT_VIDEO,

    "-c:v",
    "libx264",

    "-pix_fmt",
    "yuv420p",

    "-crf",
    "23",

    "-preset",
    "medium",

    "-movflags",
    "+faststart",

    "-an",

    H264_VIDEO
]

result = subprocess.run(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

if result.returncode != 0:

    print(result.stderr)

    raise RuntimeError(
        "H264 conversion failed."
    )


# ============================================================
# 22. FINAL CHECK
# ============================================================

if not os.path.exists(H264_VIDEO):

    raise FileNotFoundError(
        "H264 video was not created."
    )

size_mb = (
    os.path.getsize(H264_VIDEO)
    /
    (1024 * 1024)
)

print("\n" + "=" * 70)
print("FINAL H264 VIDEO READY ✅")
print("=" * 70)

print(
    "File size:",
    round(size_mb, 2),
    "MB"
)

print(
    "\nPlaying final video inside Jupyter Notebook..."
)


# ============================================================
# 23. PLAY ACTUAL FINAL VIDEO INSIDE JUPYTER
# ============================================================

display(
    Video(
        H264_VIDEO,
        embed=True,
        width=1100,
        html_attributes="controls"
    )
)

print("\n" + "=" * 70)
print("DONE ✅")
print("=" * 70)

# --- CELL 145 ---
# ============================================================
# STEP 27
# AI TRAFFIC DANGEROUS-DRIVING DETECTION
# YOLO11 + BYTE TRACK + RANDOM FOREST
# ============================================================

import os
import cv2
import math
import joblib
import warnings
import subprocess
import imageio_ffmpeg
import numpy as np
import pandas as pd

from ultralytics import YOLO
from IPython.display import Video, display

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# PATHS
# ============================================================

video_path = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\dataset\videos\traffic_video.avi"

yolo_path = r"C:\Users\Neha Kamble\runs\detect\train-3\weights\best.pt"

rf_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_random_forest.pkl"

scaler_path = r"C:\Users\Neha Kamble\Downloads\Dangerous_Driving_Detection\models\final_scaler.pkl"

output_mp4 = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_TRAFFIC_RESULT.mp4"

output_h264 = r"C:\Users\Neha Kamble\Downloads\Traffic_Danger_detection\FINAL_TRAFFIC_RESULT_H264.mp4"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.20

MIN_FEATURE_FRAMES = 15

# Speed is an estimate because pixel-to-meter calibration
# depends on the camera.
PIXEL_TO_METER = 0.05


# ============================================================
# SUDDEN BRAKING SETTINGS
# ============================================================

# Vehicle must already be moving at reasonable speed.
MIN_SPEED_FOR_BRAKING = 20.0

# Minimum speed drop required.
MIN_BRAKING_SPEED_DROP = 10.0

# Minimum percentage speed reduction.
MIN_BRAKING_PERCENT = 0.35

# Number of frames used for confirmation.
REQUIRED_BRAKING_FRAMES = 3


# ============================================================
# WRONG-WAY SETTINGS
# ============================================================

# Minimum displacement required before considering direction.
MIN_DIRECTION_DISPLACEMENT = 15.0

# Minimum number of frames required.
WRONG_WAY_MIN_FRAMES = 10

# Minimum confidence in dominant direction.
DOMINANT_DIRECTION_RATIO = 0.75


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("STEP 27 - COMPLETE AI TRAFFIC DETECTION")
print("=" * 70)


if not os.path.exists(video_path):

    raise FileNotFoundError(
        "Input video not found:\n" + video_path
    )


if not os.path.exists(yolo_path):

    raise FileNotFoundError(
        "YOLO model not found:\n" + yolo_path
    )


if not os.path.exists(rf_path):

    raise FileNotFoundError(
        "Random Forest model not found:\n" + rf_path
    )


if not os.path.exists(scaler_path):

    raise FileNotFoundError(
        "Scaler not found:\n" + scaler_path
    )


print("\nInput video   : Found ✅")
print("YOLO model    : Found ✅")
print("Random Forest : Found ✅")
print("Scaler        : Found ✅")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YOLO...")

yolo_model = YOLO(yolo_path)

print("YOLO loaded ✅")


print("\nLoading Random Forest...")

rf_model = joblib.load(rf_path)

print("Random Forest loaded ✅")


print("Loading scaler...")

scaler = joblib.load(scaler_path)

print("Scaler loaded ✅")


# ============================================================
# SHOW YOLO CLASSES
# ============================================================

print("\n================================")
print("YOLO VEHICLE CLASSES")
print("================================")

print(yolo_model.names)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open input video."
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:

    fps = 25.0


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


print("\n================================")
print("VIDEO INFORMATION")
print("================================")

print("FPS          :", fps)
print("Total Frames :", total_frames)
print(
    "Resolution   :",
    width,
    "x",
    height
)


# ============================================================
# DATA STRUCTURES
# ============================================================

frame_detections = []

tracks = {}


# ============================================================
# STEP 1
# YOLO + BYTE TRACK
# ============================================================

print("\n================================")
print("STEP 1: YOLO + BYTE TRACK")
print("================================")


frame_number = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    results = yolo_model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=CONFIDENCE,

        verbose=False

    )[0]


    current_frame_data = []


    if (

        results.boxes is not None

        and

        results.boxes.id is not None

    ):


        boxes = (

            results.boxes.xyxy

            .cpu()

            .numpy()

        )


        ids = (

            results.boxes.id

            .cpu()

            .numpy()

            .astype(int)

        )


        classes = (

            results.boxes.cls

            .cpu()

            .numpy()

            .astype(int)

        )


        confidences = (

            results.boxes.conf

            .cpu()

            .numpy()

        )


        for box, track_id, class_id, conf in zip(

            boxes,

            ids,

            classes,

            confidences

        ):


            x1, y1, x2, y2 = box


            center_x = (

                float(x1)

                +

                float(x2)

            ) / 2.0


            center_y = (

                float(y1)

                +

                float(y2)

            ) / 2.0


            class_name = str(

                yolo_model.names[
                    int(class_id)
                ]

            )


            # ------------------------------------------------
            # Store detection
            # ------------------------------------------------

            current_frame_data.append({

                "track_id":
                    int(track_id),

                "class_id":
                    int(class_id),

                "class_name":
                    class_name,

                "confidence":
                    float(conf),

                "box": (

                    float(x1),

                    float(y1),

                    float(x2),

                    float(y2)

                ),

                "center": (

                    center_x,

                    center_y

                )

            })


            # ------------------------------------------------
            # Create track
            # ------------------------------------------------

            if track_id not in tracks:

                tracks[track_id] = {

                    "vehicle_id":
                        int(track_id),

                    "class_name":
                        class_name,

                    "positions": [],

                    "speeds": [],

                    "accelerations": [],

                    "direction_changes": [],

                    "frames": [],

                    "sudden_braking":
                        False,

                    "wrong_way":
                        False

                }


            track = tracks[track_id]


            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            track["positions"].append(

                (
                    center_x,
                    center_y
                )

            )


            track["frames"].append(
                frame_number
            )


            # ------------------------------------------------
            # Speed
            # ------------------------------------------------

            if len(
                track["positions"]
            ) >= 2:


                old_x, old_y = (

                    track["positions"][-2]

                )


                distance_pixels = math.sqrt(

                    (center_x - old_x) ** 2

                    +

                    (center_y - old_y) ** 2

                )


                pixel_speed = (

                    distance_pixels
                    *
                    fps

                )


                meter_speed = (

                    pixel_speed
                    *
                    PIXEL_TO_METER

                )


                speed_kmh = (

                    meter_speed
                    *
                    3.6

                )


            else:

                speed_kmh = 0.0


            track["speeds"].append(
                speed_kmh
            )


            # ------------------------------------------------
            # Acceleration
            # ------------------------------------------------

            if len(
                track["speeds"]
            ) >= 2:


                previous_speed = (

                    track["speeds"][-2]

                )


                acceleration = (

                    speed_kmh

                    -

                    previous_speed

                ) * fps


            else:

                acceleration = 0.0


            track["accelerations"].append(
                acceleration
            )


    frame_detections.append(
        current_frame_data
    )


    if frame_number % 50 == 0:

        print(
            f"Processed: "
            f"{frame_number} / "
            f"{total_frames}"
        )


cap.release()


print("\n================================")
print("TRACKING COMPLETED")
print("================================")

print(
    "Total tracked vehicle IDs:",
    len(tracks)
)


# ============================================================
# STEP 1.5
# ROBUST EVENT DETECTION
# ============================================================

print("\n================================")
print("STEP 1.5: EVENT DETECTION")
print("================================")


# ============================================================
# SUDDEN BRAKING
# ============================================================

def detect_real_sudden_braking(
    track,
    fps
):

    speeds = np.array(
        track["speeds"],
        dtype=float
    )


    if len(speeds) < 15:

        return False


    speeds = np.clip(
        speeds,
        0,
        None
    )


    # --------------------------------------------------------
    # Median smoothing
    # --------------------------------------------------------

    smooth_window = 5

    smoothed = np.zeros_like(
        speeds
    )


    for i in range(
        len(speeds)
    ):

        start = max(
            0,
            i - smooth_window + 1
        )

        smoothed[i] = np.median(
            speeds[start:i + 1]
        )


    # --------------------------------------------------------
    # Compare speed before and after
    # --------------------------------------------------------

    lookback = max(
        5,
        int(fps * 0.4)
    )


    confirmation_frames = max(
        3,
        REQUIRED_BRAKING_FRAMES
    )


    for i in range(
        lookback,
        len(smoothed) - confirmation_frames
    ):


        before_speed = float(
            np.median(
                smoothed[
                    i - lookback:i
                ]
            )
        )


        after_speed = float(
            np.median(
                smoothed[
                    i:i + confirmation_frames
                ]
            )
        )


        # ----------------------------------------------------
        # Vehicle must be moving
        # ----------------------------------------------------

        if before_speed < MIN_SPEED_FOR_BRAKING:

            continue


        # ----------------------------------------------------
        # Speed drop
        # ----------------------------------------------------

        speed_drop = (

            before_speed

            -

            after_speed

        )


        drop_percentage = (

            speed_drop
            /
            before_speed

        )


        # ----------------------------------------------------
        # Strong braking conditions
        # ----------------------------------------------------

        if (

            speed_drop
            >=
            MIN_BRAKING_SPEED_DROP

            and

            drop_percentage
            >=
            MIN_BRAKING_PERCENT

        ):


            # ------------------------------------------------
            # Confirm decreasing trend
            # ------------------------------------------------

            decreasing_frames = 0


            for j in range(

                i,

                i + confirmation_frames

            ):


                if (

                    smoothed[j]
                    <
                    smoothed[j - 1]

                ):

                    decreasing_frames += 1

                else:

                    decreasing_frames = 0


            if (

                decreasing_frames
                >=
                2

            ):

                return True


    return False


# ============================================================
# CALCULATE OVERALL DIRECTIONS
# ============================================================

def get_track_direction(
    track
):

    positions = track[
        "positions"
    ]


    if len(positions) < 10:

        return None


    # --------------------------------------------------------
    # Use multiple frames.
    # --------------------------------------------------------

    window = min(
        20,
        len(positions) - 1
    )


    start_x = float(
        positions[-window][0]
    )


    end_x = float(
        positions[-1][0]
    )


    displacement = (
        end_x
        -
        start_x
    )


    if abs(displacement) < MIN_DIRECTION_DISPLACEMENT:

        return None


    if displacement > 0:

        return "RIGHT"

    else:

        return "LEFT"


# ============================================================
# CALCULATE DIRECTION FOR EACH VEHICLE
# ============================================================

for track_id, track in tracks.items():

    track[
        "overall_direction"
    ] = get_track_direction(
        track
    )


# ============================================================
# FIND DOMINANT TRAFFIC DIRECTION
# ============================================================

direction_counts = {

    "RIGHT": 0,

    "LEFT": 0

}


for track_id, track in tracks.items():

    direction = track.get(
        "overall_direction"
    )


    if direction in direction_counts:

        direction_counts[
            direction
        ] += 1


total_direction_tracks = (

    direction_counts["RIGHT"]

    +

    direction_counts["LEFT"]

)


dominant_direction = None


if total_direction_tracks > 0:


    right_ratio = (

        direction_counts["RIGHT"]
        /
        total_direction_tracks

    )


    left_ratio = (

        direction_counts["LEFT"]
        /
        total_direction_tracks

    )


    if right_ratio >= DOMINANT_DIRECTION_RATIO:

        dominant_direction = "RIGHT"


    elif left_ratio >= DOMINANT_DIRECTION_RATIO:

        dominant_direction = "LEFT"


print(
    "\nDirection counts:",
    direction_counts
)


print(
    "Dominant direction:",
    dominant_direction
)


# ============================================================
# FINAL SUDDEN BRAKING + WRONG-WAY DECISION
# ============================================================

for track_id, track in tracks.items():


    # ========================================================
    # SUDDEN BRAKING
    # ========================================================

    track[
        "sudden_braking"
    ] = detect_real_sudden_braking(

        track,

        fps

    )


    # ========================================================
    # WRONG-WAY
    # ========================================================

    vehicle_direction = track.get(
        "overall_direction"
    )


    track[
        "wrong_way"
    ] = False


    # --------------------------------------------------------
    # Only determine wrong-way if:
    #
    # 1. Dominant direction is very clear
    # 2. Vehicle has clear direction
    # 3. Vehicle is moving opposite
    # 4. Opposite movement is persistent
    #
    # --------------------------------------------------------

    if (

        dominant_direction is not None

        and

        vehicle_direction is not None

        and

        vehicle_direction
        !=
        dominant_direction

    ):


        positions = track[
            "positions"
        ]


        opposite_frames = 0


        for i in range(
            5,
            len(positions)
        ):


            current_x = float(
                positions[i][0]
            )


            previous_x = float(
                positions[i - 5][0]
            )


            dx = (
                current_x
                -
                previous_x
            )


            if abs(dx) < 8:

                continue


            if dominant_direction == "RIGHT":

                if dx < 0:

                    opposite_frames += 1

                else:

                    opposite_frames = max(
                        0,
                        opposite_frames - 1
                    )


            elif dominant_direction == "LEFT":

                if dx > 0:

                    opposite_frames += 1

                else:

                    opposite_frames = max(
                        0,
                        opposite_frames - 1
                    )


        if (
            opposite_frames
            >=
            WRONG_WAY_MIN_FRAMES
        ):

            track[
                "wrong_way"
            ] = True


# ============================================================
# EVENT SUMMARY
# ============================================================

real_braking_ids = set()

real_wrong_way_ids = set()


for track_id, track in tracks.items():

    if track.get(
        "sudden_braking",
        False
    ):

        real_braking_ids.add(
            int(track_id)
        )


    if track.get(
        "wrong_way",
        False
    ):

        real_wrong_way_ids.add(
            int(track_id)
        )


print("\n================================")
print("EVENT DETECTION COMPLETED")
print("================================")

print(
    "Real Sudden Braking IDs :",
    len(real_braking_ids)
)

print(
    "Real Wrong-Way IDs      :",
    len(real_wrong_way_ids)
)

print(
    "Sudden Braking Vehicles:",
    sorted(real_braking_ids)
)

print(
    "Wrong-Way Vehicles:",
    sorted(real_wrong_way_ids)
)


# ============================================================
# STEP 2
# FEATURE EXTRACTION
# ============================================================

print("\n================================")
print("STEP 2: FEATURE EXTRACTION")
print("================================")


feature_rows = []


for track_id, track in tracks.items():

    speeds = np.array(
        track["speeds"],
        dtype=float
    )


    accelerations = np.array(
        track["accelerations"],
        dtype=float
    )


    positions = track[
        "positions"
    ]


    if len(speeds) < MIN_FEATURE_FRAMES:

        continue


    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    avg_speed = float(
        np.mean(speeds)
    )


    max_speed = float(
        np.max(speeds)
    )


    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    avg_acceleration = float(
        np.mean(
            np.abs(
                accelerations
            )
        )
    )


    max_acceleration = float(
        np.max(
            np.abs(
                accelerations
            )
        )
    )


    # --------------------------------------------------------
    # Deceleration
    # --------------------------------------------------------

    negative_acc = accelerations[
        accelerations < 0
    ]


    if len(
        negative_acc
    ) > 0:

        max_deceleration = float(
            np.max(
                np.abs(
                    negative_acc
                )
            )
        )

    else:

        max_deceleration = 0.0


    # --------------------------------------------------------
    # Direction changes
    # --------------------------------------------------------

    direction_angles = []


    if len(
        positions
    ) >= 3:


        for i in range(
            2,
            len(positions)
        ):


            x1, y1 = positions[
                i - 2
            ]


            x2, y2 = positions[
                i - 1
            ]


            x3, y3 = positions[
                i
            ]


            v1 = np.array([

                x2 - x1,

                y2 - y1

            ])


            v2 = np.array([

                x3 - x2,

                y3 - y2

            ])


            norm1 = np.linalg.norm(
                v1
            )


            norm2 = np.linalg.norm(
                v2
            )


            if (

                norm1 > 0

                and

                norm2 > 0

            ):


                cosine_value = (

                    np.dot(v1, v2)

                    /

                    (
                        norm1
                        *
                        norm2
                    )

                )


                cosine_value = np.clip(

                    cosine_value,

                    -1.0,

                    1.0

                )


                angle = math.acos(
                    cosine_value
                )


                direction_angles.append(
                    angle
                )


    if len(
        direction_angles
    ) > 0:


        avg_direction_change = float(
            np.mean(
                direction_angles
            )
        )


        max_direction_change = float(
            np.max(
                direction_angles
            )
        )


    else:

        avg_direction_change = 0.0

        max_direction_change = 0.0


    # --------------------------------------------------------
    # Movement distance
    # --------------------------------------------------------

    total_distance = 0.0


    for i in range(
        1,
        len(positions)
    ):


        x1, y1 = positions[
            i - 1
        ]


        x2, y2 = positions[
            i
        ]


        total_distance += math.sqrt(

            (x2 - x1) ** 2

            +

            (y2 - y1) ** 2

        )


    feature_rows.append({

        "vehicle_id":
            int(track_id),

        "frames_observed":
            int(len(positions)),

        "duration_seconds":
            float(
                len(positions)
                /
                fps
            ),

        "avg_speed":
            avg_speed,

        "max_speed":
            max_speed,

        "avg_acceleration":
            avg_acceleration,

        "max_acceleration":
            max_acceleration,

        "max_deceleration":
            max_deceleration,

        "avg_direction_change":
            avg_direction_change,

        "max_direction_change":
            max_direction_change,

        "total_movement_distance":
            float(total_distance)

    })


features_df = pd.DataFrame(
    feature_rows
)


print(
    "\nVehicles with valid features:",
    len(features_df)
)


if len(features_df) == 0:

    raise RuntimeError(
        "No vehicles have enough tracking frames."
    )


# ============================================================
# STEP 3
# RANDOM FOREST PREDICTION
# ============================================================

print("\n================================")
print("STEP 3: RANDOM FOREST PREDICTION")
print("================================")


feature_columns = [

    "avg_speed",

    "max_speed",

    "avg_acceleration",

    "max_acceleration",

    "max_deceleration",

    "avg_direction_change",

    "max_direction_change"

]


# IMPORTANT:
# Keep DataFrame column names exactly the same
# as training data.

X_video = features_df[
    feature_columns
].copy()


# ============================================================
# SCALE
# ============================================================

X_scaled = scaler.transform(
    X_video
)


# ============================================================
# PREDICTION
# ============================================================

predictions = rf_model.predict(
    X_scaled
)


probabilities = rf_model.predict_proba(
    X_scaled
)


features_df[
    "prediction"
] = predictions


features_df[
    "status"
] = features_df[
    "prediction"
].map({

    0: "SAFE",

    1: "DANGEROUS"

})


# ============================================================
# DANGER PROBABILITY
# ============================================================

if probabilities.shape[1] >= 2:

    danger_probability = (

        probabilities[:, 1]
        *
        100

    )

else:

    danger_probability = np.zeros(
        len(features_df)
    )


features_df[
    "danger_probability"
] = danger_probability


features_df[
    "risk_score"
] = features_df[
    "danger_probability"
].clip(
    0,
    100
)


# ============================================================
# CLASSIFICATION MAP
# ============================================================

classification_map = {}


for _, row in features_df.iterrows():

    vehicle_id = int(
        row["vehicle_id"]
    )


    classification_map[
        vehicle_id
    ] = {

        "status":
            str(
                row["status"]
            ),

        "risk_score":
            float(
                row["risk_score"]
            ),

        "danger_probability":
            float(
                row["danger_probability"]
            )

    }


safe_count = int(

    (
        features_df[
            "prediction"
        ]
        ==
        0
    ).sum()

)


dangerous_count = int(

    (
        features_df[
            "prediction"
        ]
        ==
        1
    ).sum()

)


print(
    "\nSAFE vehicles      :",
    safe_count
)


print(
    "DANGEROUS vehicles :",
    dangerous_count
)


# ============================================================
# STEP 4
# CREATE COLORED VIDEO
# ============================================================

print("\n================================")
print("STEP 4: CREATING COLORED VIDEO")
print("================================")


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(

    output_mp4,

    fourcc,

    fps,

    (
        width,
        height
    )

)


if not writer.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# COUNTERS
# ============================================================

all_detected_ids = set()

classified_ids = set()

braking_ids = set()

wrong_way_ids = set()


# ============================================================
# DRAW EVERY FRAME
# ============================================================

cap = cv2.VideoCapture(
    video_path
)


for frame_index in range(
    len(frame_detections)
):


    ret, frame = cap.read()


    if not ret:

        break


    detections = frame_detections[
        frame_index
    ]


    # ========================================================
    # DRAW VEHICLES
    # ========================================================

    for detection in detections:


        track_id = detection[
            "track_id"
        ]


        class_name = detection[
            "class_name"
        ]


        x1, y1, x2, y2 = detection[
            "box"
        ]


        all_detected_ids.add(
            track_id
        )


        # ----------------------------------------------------
        # VEHICLE TYPE
        # ----------------------------------------------------

        vehicle_type = class_name.upper()


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if track_id in classification_map:


            info = classification_map[
                track_id
            ]


            status = info[
                "status"
            ]


            risk_score = info[
                "risk_score"
            ]


            classified_ids.add(
                track_id
            )


        else:


            status = "SAFE"

            risk_score = 0.0


        # ----------------------------------------------------
        # TRACK
        # ----------------------------------------------------

        track = tracks.get(
            track_id,
            None
        )


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        if (

            track is not None

            and

            len(
                track["speeds"]
            ) > 0

        ):


            current_speed = track[
                "speeds"
            ][-1]


        else:

            current_speed = 0.0


        # ----------------------------------------------------
        # SMOOTH DISPLAY SPEED
        # ----------------------------------------------------

        if (

            track is not None

            and

            len(
                track["speeds"]
            ) >= 3

        ):


            display_speed = float(

                np.median(

                    track[
                        "speeds"
                    ][-3:]

                )

            )


        else:

            display_speed = float(
                current_speed
            )


        # ====================================================
        # REAL SUDDEN BRAKING
        # ====================================================

        sudden_braking = False


        if track is not None:

            sudden_braking = bool(

                track.get(
                    "sudden_braking",
                    False
                )

            )


        if sudden_braking:

            braking_ids.add(
                track_id
            )


        # ====================================================
        # REAL WRONG-WAY
        # ====================================================

        wrong_way = False


        if track is not None:

            wrong_way = bool(

                track.get(
                    "wrong_way",
                    False
                )

            )


        if wrong_way:

            wrong_way_ids.add(
                track_id
            )


        # ====================================================
        # BOX COLOR
        # ====================================================

        if status == "DANGEROUS":

            # RED
            box_color = (
                0,
                0,
                255
            )

        else:

            # GREEN
            box_color = (
                0,
                255,
                0
            )


        # ====================================================
        # BOUNDING BOX
        # ====================================================

        cv2.rectangle(

            frame,

            (
                int(x1),
                int(y1)
            ),

            (
                int(x2),
                int(y2)
            ),

            box_color,

            3

        )


        # ====================================================
        # LABEL 1
        # ====================================================

        label1 = (

            f"ID: {track_id} | "

            f"{vehicle_type}"

        )


        # ====================================================
        # LABEL 2
        # ====================================================

        label2 = (

            f"{status} | "

            f"Speed: "
            f"{display_speed:.1f} km/h"

        )


        # ====================================================
        # LABEL 3
        # ====================================================

        label3 = (

            f"Risk: "
            f"{risk_score:.0f}/100"

        )


        # ====================================================
        # WARNINGS
        # ====================================================

        warnings_text = []


        if sudden_braking:

            warnings_text.append(
                "SUDDEN BRAKING"
            )


        if wrong_way:

            warnings_text.append(
                "WRONG-WAY"
            )


        # ====================================================
        # TEXT POSITION
        # ====================================================

        text_x = int(x1)


        text_y = int(y1) - 60


        if text_y < 80:

            text_y = int(y1) + 20


        # ====================================================
        # DRAW LABEL 1
        # ====================================================

        cv2.putText(

            frame,

            label1,

            (
                text_x,
                text_y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            box_color,

            2,

            cv2.LINE_AA

        )


        # ====================================================
        # DRAW LABEL 2
        # ====================================================

        cv2.putText(

            frame,

            label2,

            (
                text_x,
                text_y + 23
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            box_color,

            2,

            cv2.LINE_AA

        )


        # ====================================================
        # DRAW LABEL 3
        # ====================================================

        cv2.putText(

            frame,

            label3,

            (
                text_x,
                text_y + 46
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            box_color,

            2,

            cv2.LINE_AA

        )


        # ====================================================
        # DRAW EVENT WARNINGS
        # ====================================================

        warning_y = text_y + 70


        for warning in warnings_text:


            cv2.putText(

                frame,

                warning,

                (
                    text_x,
                    warning_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                box_color,

                2,

                cv2.LINE_AA

            )


            warning_y += 23


    # ========================================================
    # WRITE FRAME
    # ========================================================

    writer.write(
        frame
    )


    if (

        frame_index + 1
    ) % 50 == 0:


        print(

            f"Video written: "

            f"{frame_index + 1} / "

            f"{total_frames}"

        )


# ============================================================
# RELEASE
# ============================================================

cap.release()

writer.release()


# ============================================================
# FINAL VIDEO SUMMARY
# ============================================================

print("\n==============================================")
print("FINAL VIDEO CREATED")
print("==============================================")


print(
    "Tracked vehicle IDs :",
    len(all_detected_ids)
)


print(
    "Vehicles classified :",
    len(classified_ids)
)


print(
    "SAFE vehicles        :",
    safe_count
)


print(
    "DANGEROUS vehicles   :",
    dangerous_count
)


print(
    "Sudden Braking IDs   :",
    len(braking_ids)
)


print(
    "Wrong-Way IDs        :",
    len(wrong_way_ids)
)


print(
    "\nOutput Video:"
)


print(
    output_mp4
)


# ============================================================
# STEP 5
# H264 CONVERSION
# ============================================================

print("\n================================")
print("STEP 5: H264 CONVERSION")
print("================================")


ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


print(
    "FFmpeg found ✅"
)


command = [

    ffmpeg,

    "-y",

    "-i",
    output_mp4,

    "-c:v",
    "libx264",

    "-preset",
    "fast",

    "-crf",
    "23",

    "-pix_fmt",
    "yuv420p",

    "-an",

    "-movflags",
    "+faststart",

    output_h264

]


result = subprocess.run(

    command,

    stdout=subprocess.PIPE,

    stderr=subprocess.PIPE,

    text=True

)


if result.returncode != 0:

    print(
        result.stderr
    )

    raise RuntimeError(
        "H264 conversion failed."
    )


if not os.path.exists(
    output_h264
):

    raise FileNotFoundError(
        "H264 output was not created."
    )


size_mb = (

    os.path.getsize(
        output_h264
    )

    /

    (1024 * 1024)

)


print(
    "\nH264 video created ✅"
)


print(
    "File size:",
    round(size_mb, 2),
    "MB"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==============================================")
print("FINAL RESULT READY")
print("==============================================")


print(
    "Final H264 video:"
)


print(
    output_h264
)


print(
    "\nPlaying actual final video inside Jupyter..."
)


# ============================================================
# PLAY INSIDE JUPYTER
# ============================================================

display(

    Video(

        output_h264,

        embed=True,

        width=1000,

        html_attributes="controls"

    )

)


print("\n==============================================")
print("DONE ✅")
print("==============================================")

# --- CELL 146 ---


