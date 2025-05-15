from ultralytics import YOLO
import cv2
import numpy as np
import time
import csv
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

model = YOLO(r'C:\Users\gaura\PycharmProjects\DroneBasedCrowdAnalysis\runs\detect\train\weights\best.pt')

cap = cv2.VideoCapture(1)

PERSON_CLASS_ID = 0

GRID_SIZE = 4
ZOOM_FACTOR = 2

csv_filename = "people_count_log.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Timestamp", "People Count"])


start_time = time.time()

root = tk.Tk()
root.title("Crowd Detection")


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return

    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Couldn't read the image file.")
        return

    # Detect people in the image
    results = model.predict(source=image, conf=0.5, device='cpu', show=False, verbose=False)
    detections = results[0].boxes
    person_count = sum(1 for box in detections if int(box.cls[0]) == PERSON_CLASS_ID and box.conf[0] >= 0.5)

    # Annotate image
    image_annotated = results[0].plot()

    # Convert image for Tkinter display
    image_annotated = cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_annotated)
    img_pil = img_pil.resize((400, 300), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update image on UI
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Show people count
    result_label.config(text=f"People Count: {person_count}")


# UI Elements
upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 12), bg="lightblue")
upload_button.pack(pady=10)

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="green")
result_label.pack()


# Run in a separate thread
def process_video():
    global start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        h, w, _ = frame.shape
        grid_h, grid_w = h // GRID_SIZE, w // GRID_SIZE

        person_count = 0

        # Process each grid cell
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1, y1 = j * grid_w, i * grid_h
                x2, y2 = x1 + grid_w, y1 + grid_h

                roi = frame[y1:y2, x1:x2]
                zoomed_roi = cv2.resize(roi, (int(grid_w * ZOOM_FACTOR), int(grid_h * ZOOM_FACTOR)))

                # Predict on zoomed region
                results = model.predict(source=zoomed_roi, conf=0.5, device='cpu', show=False, verbose=False)
                detections = results[0].boxes

                if detections is not None:
                    for box in detections:
                        cls_id = int(box.cls[0])
                        conf_score = box.conf[0]

                        if cls_id == PERSON_CLASS_ID and conf_score >= 0.5:
                            person_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Run detection on full frame
        full_results = model.predict(source=frame, conf=0.5, device='cpu', show=False, verbose=False)
        frame_annotated = full_results[0].plot()

        # Display people count
        cv2.putText(frame_annotated, f'People Count: {person_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Grid-Based Crowd Analysis", frame_annotated)

        # Log every 30 seconds
        if time.time() - start_time >= 30:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, person_count])
            print(f"Logged at {timestamp}: {person_count} people detected.")
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Start video processing in parallel
import threading

video_thread = threading.Thread(target=process_video, daemon=True)
video_thread.start()

# Run the Tkinter loop
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
