"""
camera_pipeline.py
-------------------

This script provides a minimal demonstration of how to use a webcam as a
stand‑in for a drone camera in the context of the Right of Way Encroachment
project. It implements two modes:

1. **Setup mode (Project 1)**
   - Captures a series of static photos from the default camera (e.g. the floor).
   - Stitches them into a single mosaic using OpenCV’s Stitcher API.
   - Lets the user draw pipeline centrelines on the stitched image by clicking
     points. Multiple lines can be drawn; press `n` to start a new line.
   - Saves the stitched panorama as `stitched.png` and the pipeline lines as
     `pipes.json` (pixel coordinates relative to the stitched image).

2. **Monitor mode (Project 2)**
   - Loads the stitched panorama and pipe definitions.
   - Captures live frames from the camera and computes a homography to align
     the current view with the stored stitched image using ORB feature
     matching and RANSAC.
   - Warps the stored pipeline lines onto each live frame and overlays them.
   - Performs a very simple object detection: thresholds for bright blobs,
     extracts contours and treats any bright contour that touches a pipeline
     (within a few pixels) as an encroacher. Encroaching objects are drawn
     with red bounding boxes; non‑encroaching objects are drawn with green.

**Disclaimer:** This code is a prototype. It relies on a very rudimentary
detection algorithm (brightness thresholding) and may not work reliably in
complex scenes or varying lighting. For real deployments, replace the
thresholding with a trained object detector (e.g. YOLOv26) and tune the
alignment and distance thresholds accordingly.
"""

import cv2
import numpy as np
import json
import math
import os
import time

# Try to import a sophisticated detector (YOLOv26 via ultralytics) if available.
# This import will fail if the ultralytics package or the weights file are not present.
try:
    from ultralytics import YOLO  # type: ignore
    # Attempt to load a small YOLOv26 model.  You can replace 'yolov26n.pt' with
    # another weight file path if you have downloaded one.
    _MODEL_PATH = "yolo26n.pt"
    if os.path.exists(_MODEL_PATH):
        yolo_model = YOLO(_MODEL_PATH)
        print(f"Loaded YOLO model from {_MODEL_PATH}")
    else:
        yolo_model = None
        if 'YOLO' in globals():
            print(
                f"ultralytics imported but weights not found at {_MODEL_PATH}; "
                "falling back to simple detector."
            )
except Exception as _e:
    yolo_model = None



def capture_images(output_dir="captures"):
    """Capture still images from the default camera until the user presses
    's' (stitch) or 'q' (quit). Saves each captured frame into ``output_dir``.
    Manual capture is triggered with the **C** key.  Press **T** to toggle
    automatic capture.  When auto capture is enabled, a frame will be saved
    automatically every half second.  Returns the list of image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return []
    captured_paths: list[str] = []
    frame_count = 0
    auto_capture = False
    # Keep track of last automatic capture time
    last_auto_time = time.time()
    print("Press 'c' to capture a frame, 's' to stitch, 'q' to quit.")
    print("Press 't' to toggle automatic capture every 0.5 seconds.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        # Overlay capture mode status on the live preview
        overlay = frame.copy()
        status_text = "Auto: ON" if auto_capture else "Auto: OFF"
        cv2.putText(
            overlay,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if auto_capture else (0, 0, 255),
            2,
        )
        cv2.imshow("Camera – Capture Mode", overlay)
        key = cv2.waitKey(1) & 0xFF
        # Manual capture on 'c'
        if key == ord('c'):
            path = os.path.join(output_dir, f"capture_{frame_count:03d}.png")
            cv2.imwrite(path, frame)
            captured_paths.append(path)
            frame_count += 1
            print(f"Captured {path} (manual)")
            # Reset auto timer to avoid immediate extra capture if switching
            last_auto_time = time.time()
        # Toggle auto capture on 't'
        elif key == ord('t'):
            auto_capture = not auto_capture
            mode = "enabled" if auto_capture else "disabled"
            print(f"Automatic capture {mode}.")
            last_auto_time = time.time()
        # Start stitching
        elif key == ord('s'):
            print("Stitching images...")
            break
        # Quit and discard captures
        elif key == ord('q'):
            print("Exiting capture mode.")
            captured_paths = []
            break
        # If auto capture is enabled, save a frame every 0.5 seconds
        if auto_capture:
            current_time = time.time()
            if current_time - last_auto_time >= 0.5:
                path = os.path.join(output_dir, f"capture_{frame_count:03d}.png")
                cv2.imwrite(path, frame)
                captured_paths.append(path)
                frame_count += 1
                last_auto_time = current_time
                print(f"Captured {path} (auto)")
    cap.release()
    cv2.destroyAllWindows()
    return captured_paths


def stitch_images(image_paths, output_path="stitched.png"):
    """
    Stitches multiple images into a single panorama.  Uses the
    ``cv2.Stitcher_SCANS`` mode, which is more tolerant of sequences of images
    taken while scanning a flat scene (e.g. a floor).  Returns the stitched
    image array if successful, otherwise ``None``.  Saves the result to
    ``output_path``.
    """
    if len(image_paths) < 2:
        print("Need at least two images to stitch.")
        return None
    images = [cv2.imread(p) for p in image_paths]
    images = [img for img in images if img is not None]
    if len(images) < 2:
        print("Error: Could not load enough images for stitching.")
        return None
    # Use SCANS mode for stitching flat surfaces.  If your images come from
    # a handheld sweep of a planar area, this mode often yields better results.
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    except AttributeError:
        # Fallback if cv2.Stitcher_SCANS is unavailable; use default.
        stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        print("Error during stitching: status", status)
        return None
    cv2.imwrite(output_path, stitched)
    print(f"Stitched image saved to {output_path}")
    return stitched


def draw_pipelines(image):
    """Allows the user to interactively draw multiple pipeline lines on the
    provided image. Returns a list of polylines, where each polyline is a
    list of (x, y) tuples. The user can press 'n' to start a new line,
    'u' to undo the last point, and 'q' to finish drawing.
    """
    drawing = {
        'lines': [],  # list of polylines; each polyline is a list of points
        'current': []  # points for the polyline currently being drawn
    }

    display = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing['current'].append((x, y))
    
    cv2.namedWindow("Draw Pipes")
    cv2.setMouseCallback("Draw Pipes", on_mouse)
    print(
        "Draw pipelines: click to add points. Press 'n' to start a new line, 'u' to undo, 'q' to finish."
    )
    while True:
        tmp = display.copy()
        # Draw completed lines
        for poly in drawing['lines']:
            for i in range(len(poly) - 1):
                cv2.line(tmp, poly[i], poly[i + 1], (255, 255, 0), 2)
        # Draw current polyline
        for i in range(len(drawing['current']) - 1):
            cv2.line(tmp, drawing['current'][i], drawing['current'][i + 1], (0, 255, 255), 2)
        cv2.imshow("Draw Pipes", tmp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            if drawing['current']:
                drawing['lines'].append(drawing['current'])
                drawing['current'] = []
                print("Started new line.")
        elif key == ord('u'):
            if drawing['current']:
                drawing['current'].pop()
                print("Removed last point.")
            elif drawing['lines']:
                # Undo last completed line
                drawing['current'] = drawing['lines'].pop()
                print("Undid last line.")
        elif key == ord('q'):
            # Save current line if not empty
            if drawing['current']:
                drawing['lines'].append(drawing['current'])
            break
    cv2.destroyWindow("Draw Pipes")
    return drawing['lines']


def save_pipelines(pipelines, path="pipes.json"):
    """Saves pipeline polylines to a JSON file."""
    with open(path, "w") as f:
        json.dump({'polylines': pipelines}, f)
    print(f"Saved pipelines to {path}")


def load_pipelines(path="pipes.json"):
    """Loads pipeline polylines from a JSON file. Returns list of polylines."""
    if not os.path.exists(path):
        print(f"Pipelines file {path} not found.")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get('polylines', [])


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Computes the shortest distance between a point (px,py) and a line segment
    defined by points (ax,ay)-(bx,by). Returns the distance in pixels."""
    # Vector from A to B
    vx, vy = bx - ax, by - ay
    # Vector from A to P
    wx, wy = px - ax, py - ay
    # Project w onto v
    c1 = vx * wx + vy * wy
    c2 = vx * vx + vy * vy
    if c2 == 0:
        # A and B are the same point
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, c1 / c2))
    closest_x = ax + t * vx
    closest_y = ay + t * vy
    return math.hypot(px - closest_x, py - closest_y)


# Classes that we consider encroachers.  When using a YOLO detector, only
# detections with labels in this set will be considered.  For example,
# 'car', 'truck' and 'bus' correspond to common COCO classes.
ENCROACHER_CLASSES = {
    'car', 'truck', 'bus', 'train', 'excavator', 'digger', 'tractor', 'spoon', 'knife', 'fork', 'hand', 'thumb', 'finger'
}


def detect_objects(frame):
    """
    Detect objects in the given frame and return a list of detections.
    Each detection is a tuple ``(x1, y1, x2, y2, label, confidence)`` in
    pixel coordinates.  This function will attempt to use a YOLOv26 model
    (via the ultralytics package) if available.  If no model is available,
    it falls back to a simple brightness-based blob detector.

    You can customise the global variable ``yolo_model`` by placing the
    appropriate YOLO weights file (e.g. ``yolov26n.pt``) in the current
    directory.  The encroacher classes are defined in the set
    ``ENCROACHER_CLASSES``.
    """
    detections = []
    # Use YOLOv26 if available
    if yolo_model is not None:
        # Run inference; this returns a list of Result objects
        results = yolo_model(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            # xyxy: [N, 4] bounding boxes, cls: [N] class IDs, conf: [N]
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else None
            cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
            if xyxy is None or cls_ids is None or confs is None:
                continue
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                label = result.names.get(cls_id, str(cls_id))
                if label.lower() in ENCROACHER_CLASSES:
                    detections.append(
                        (int(x1), int(y1), int(x2), int(y2), label, float(conf))
                    )
        return detections
    # Fallback: simple bright blob detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    # Threshold bright areas; adjust threshold as needed
    _, thresh = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Use a generic label for fallback detections
        detections.append((x, y, x + w, y + h, 'blob', 1.0))
    return detections


def monitor_mode(stitched_path="stitched.png", pipes_path="pipes.json"):
    """Loads the stitched image and pipelines, then runs a loop capturing
    live frames, aligning them to the stitched panorama and detecting
    encroaching objects. Press 'q' to exit the monitoring loop."""
    if not os.path.exists(stitched_path):
        print(f"Stitched image {stitched_path} not found. Run setup mode first.")
        return
    stitched = cv2.imread(stitched_path)
    if stitched is None:
        print(f"Failed to load {stitched_path}")
        return
    pipelines = load_pipelines(pipes_path)
    if not pipelines:
        print("No pipelines defined. Run setup mode to draw pipelines.")
        return
    # Precompute ORB features for stitched image
    stitched_gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(stitched_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    print("Monitoring... press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame.")
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(frame_gray, None)
        H = None
        if des2 is not None and len(des2) > 10:
            # Match descriptors and compute homography
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            # Lowe's ratio test
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # Prepare a copy for drawing
        output = frame.copy()
        encroacher_count = 0
        warped_pipelines = []
        if H is not None:
            # Warp pipeline lines into the frame and cache results
            for poly in pipelines:
                pts = np.float32(poly).reshape(-1, 1, 2)
                warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                warped_pipelines.append(warped)
                for i in range(len(warped) - 1):
                    p1 = tuple(map(int, warped[i]))
                    p2 = tuple(map(int, warped[i + 1]))
                    cv2.line(output, p1, p2, (0, 255, 255), 2)
        # Detect objects in the frame
        detections = detect_objects(frame)
        for (x1, y1, x2, y2, label, conf) in detections:
            # Determine if this detection is encroaching
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            is_encroaching = False
            if warped_pipelines:
                for warped in warped_pipelines:
                    for i in range(len(warped) - 1):
                        ax, ay = warped[i]
                        bx, by = warped[i + 1]
                        dist = point_to_segment_distance(cx, cy, ax, ay, bx, by)
                        if dist < 15:
                            is_encroaching = True
                            break
                    if is_encroaching:
                        break
            # Choose colour based on encroachment status
            color = (0, 255, 0)
            if is_encroaching:
                color = (0, 0, 255)
                encroacher_count += 1
            # Draw bounding box and label
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label}" if is_encroaching else f"{label}"
            cv2.putText(
                output,
                label_text,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        cv2.putText(
            output,
            f"Encroachers: {encroacher_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if encroacher_count else (0, 255, 0),
            2,
        )
        cv2.imshow("Monitoring – Live", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Right of Way Encroachment Camera Pipeline")
    print("1) Setup mode: capture images, stitch and draw pipelines")
    print("2) Monitor mode: live alignment and encroachment detection")
    choice = input("Choose mode (1/2): ").strip()
    if choice == '1':
        # Capture images
        captures = capture_images()
        if captures:
            stitched = stitch_images(captures)
            if stitched is not None:
                # Draw pipelines
                pipelines = draw_pipelines(stitched)
                save_pipelines(pipelines)
    elif choice == '2':
        monitor_mode()
    else:
        print("Invalid choice.")


if __name__ == '__main__':
    main()