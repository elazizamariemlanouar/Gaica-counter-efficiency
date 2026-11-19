"""
Counter service-time tracker (GAICA 2025 - Counter Efficiency Award)

Detects people with YOLOv8, tracks them with ByteTrack, and measures how long
each client stays in a counter zone drawn by the user on the first frame.
Outputs the average service time per counter and a CSV of every service.

Usage:
    python service_time_tracker.py path/to/video.mp4
"""
import sys
import cv2
from ultralytics import YOLO
import supervision as sv
import torch
import csv

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Video path
# -----------------------------
video_path = sys.argv[1] if len(sys.argv) > 1 else input("Path to the video: ").strip()
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Cannot read video")

# -----------------------------
# Ask number of zones
# -----------------------------
num_zones = int(input("Enter number of zones to select: "))

# -----------------------------
# Interactive zone selection
# -----------------------------
zones = []
drawing = False
ix, iy = -1, -1
current_zone = 0

def draw_zone(event, x, y, flags, param):
    global ix, iy, drawing, first_frame, current_zone
    temp_frame = first_frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.rectangle(temp_frame, (ix, iy), (x, y), (0,255,0), 2)
        cv2.imshow("Select Zones", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        zones.append({
            "id": current_zone + 1,
            "x1": min(ix, x), "y1": min(iy, y),
            "x2": max(ix, x), "y2": max(iy, y),
            "label": f"Zone {current_zone+1}"
        })
        current_zone += 1
        cv2.rectangle(first_frame, (ix, iy), (x, y), (0,255,0), 2)
        cv2.imshow("Select Zones", first_frame)
        print(f"Zone {current_zone} selected.")
        if current_zone >= num_zones:
            cv2.destroyWindow("Select Zones")
            print("All zones selected!")

cv2.namedWindow("Select Zones")
cv2.setMouseCallback("Select Zones", draw_zone)
print(f"Draw {num_zones} zones on the first frame. Press any key to start drawing.")
cv2.imshow("Select Zones", first_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------
# Zone states (+2s buffers)
# -----------------------------
zone_states = {
    z["id"]: {
        "current_id": None,
        "start_time": None,
        "history": [],
        "pending_id": None,
        "pending_enter_time": None,
        "pending_exit_time": None,
    } for z in zones
}

# -----------------------------
# Helpers
# -----------------------------
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int((y1 + y2)/2)

def point_in_rect(cx, cy, zone):
    return zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]

# -----------------------------
# YOLO + ByteTrack
# -----------------------------
model = YOLO("yolov8n.pt").to(device)
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=int(round(fps)),
    minimum_consecutive_frames=1
)

# -----------------------------
# Main loop
# -----------------------------
ENTRY_BUFFER = 2.0   # seconds inside before starting timer
EXIT_BUFFER  = 2.0   # seconds outside before stopping timer
MIN_AREA = 1200      # ignore small far-away people

# Time is measured on the VIDEO clock (frame index / fps), not the computer clock,
# so durations stay correct whether processing runs faster or slower than real time.
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    now = frame_idx / fps

    # Detect ONLY people (class 0)
    results = model(frame, classes=[0], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Ignore small detections (too far)
    detections = detections[detections.area > MIN_AREA]

    # Track people
    tracks = tracker.update_with_detections(detections)
    track_centers = {tid: bbox_center(xyxy)
                     for xyxy, tid in zip(tracks.xyxy, tracks.tracker_id)}

    # Process each zone
    for z in zones:
        zid = z["id"]
        st = zone_states[zid]

        # 1) Get candidates INSIDE the zone
        candidates = [(tid, cx, cy) for tid, (cx, cy)
                      in track_centers.items() if point_in_rect(cx, cy, z)]

        chosen_id = None
        if len(candidates) == 1:
            chosen_id = candidates[0][0]
        elif len(candidates) > 1:
            # closest person to bottom side (cashier)
            chosen_id = max(candidates, key=lambda x: x[2])[0]

        # -------------------------
        # ENTRY BUFFER LOGIC
        # -------------------------
        if st["current_id"] is None:  
            if chosen_id is not None:
                # someone newly enters
                if st["pending_id"] != chosen_id:
                    st["pending_id"] = chosen_id
                    st["pending_enter_time"] = now
                else:
                    # if same pending id stays 2s → confirm entry
                    if now - st["pending_enter_time"] >= ENTRY_BUFFER:
                        st["current_id"] = chosen_id
                        st["start_time"] = now
                        st["pending_id"] = None
                        print(f"[{z['label']}] Person {chosen_id} ENTERED (after buffer)")
        else:
            # -------------------------
            # EXIT BUFFER LOGIC
            # -------------------------
            cid = st["current_id"]

            if cid in track_centers:
                cx, cy = track_centers[cid]
                inside = point_in_rect(cx, cy, z)
            else:
                inside = False

            if inside:
                st["pending_exit_time"] = None
                duration = now - st["start_time"]
                cv2.putText(frame, f"ID {cid} - {int(duration)}s",
                            (z["x1"], z["y2"] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                if st["pending_exit_time"] is None:
                    st["pending_exit_time"] = now
                else:
                    if now - st["pending_exit_time"] >= EXIT_BUFFER:
                        duration = now - st["start_time"]
                        st["history"].append({"id": cid, "duration_sec": round(duration,1)})
                        print(f"[{z['label']}] Person {cid} LEFT → {duration:.1f}s")
                        st["current_id"] = None
                        st["start_time"] = None
                        st["pending_exit_time"] = None

        # Draw the zone
        color = (0,255,0) if st["current_id"] is None else (0,165,255)
        cv2.rectangle(frame, (z["x1"], z["y1"]), (z["x2"], z["y2"]), color, 2)
        cv2.putText(frame, z["label"], (z["x1"], z["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw average service time
        if len(st["history"]) > 0:
            avg_time = sum(h["duration_sec"] for h in st["history"]) / len(st["history"])
            cv2.putText(frame, f"Avg: {int(avg_time)}s", (z["x2"]+10, z["y2"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Draw tracks
    for xyxy, track_id in zip(tracks.xyxy, tracks.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Service Time Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Close services still in progress when the video ends
for zid, st in zone_states.items():
    if st["current_id"] is not None:
        duration = now - st["start_time"]
        st["history"].append({"id": st["current_id"], "duration_sec": round(duration, 1)})

# -----------------------------
# Save CSV
# -----------------------------
with open("service_times.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["zone_id","client_id","duration_sec"])
    writer.writeheader()
    for zid, st in zone_states.items():
        for record in st["history"]:
            writer.writerow({
                "zone_id": zid,
                "client_id": record["id"],
                "duration_sec": record["duration_sec"]
            })

# Summary
for zid, st in zone_states.items():
    print(f"Zone {zid} history:", st["history"])
    if len(st["history"]) > 0:
        avg = sum(h["duration_sec"] for h in st["history"]) / len(st["history"])
        print(f"Average service time for Zone {zid}: {avg:.1f}s")