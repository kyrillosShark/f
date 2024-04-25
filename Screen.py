import numpy as np
import cv2
import supervision as sv
from inference import get_model, InferencePipeline
import time
import os
from dotenv import load_dotenv
import supervision as sv
from inference.core.interfaces.stream.sinks import VideoFileSink
import time
from supervision.detection.core import Detections
import pyautogui
import pygetwindow as gw
import mss
from shapely.geometry import box
from rtree import index
import numpy as np
from inference.core.interfaces.camera.entities import VideoFrame
load_dotenv()  # This loads the environment variables from `.env` into the environment

tracker = sv.ByteTrack(match_thresh=0.1, track_buffer=130)
bounding_box_annotator = sv.BoundingBoxAnnotator()
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
print("Roboflow API Key:", roboflow_api_key)
# Load the video source and set up the video writer
with mss.mss() as sct:
    # Get total dimensions of the primary monitor
    total_width = sct.monitors[1]['width']
    total_height = sct.monitors[1]['height']
    
    # Calculate the dimensions for the top right quarter
    monitor = {
        "top": 100,  # Start at the top of the screen
        "left": int(total_width / 2),  # Start from half the total width to get the right side
        "width": int(total_width / 2),  # Half the total width
        "height": int(total_height / 2),  # Half the total height
        "mon": 1  # The monitor number to capture from, usually 1 on single-monitor setups
    }

    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_sink = cv2.VideoWriter('window_output.avi', fourcc, fps, (monitor['width'], monitor['height']))

def add_detections(detections1: Detections, detections2: Detections) -> Detections:
    idx = index.Index()
    # Insert each bounding box from detections2 into the R-tree index
    for i, det in enumerate(detections2.xyxy):
        idx.insert(i, box(*det).bounds)

    filtered_detections1 = []
    filtered_class_name1 = []
    filtered_confidence1 = []
    for i, det1 in enumerate(detections1.xyxy):
        overlap = False
        # Check for overlaps using the R-tree index
        for pos in idx.intersection(box(*det1).bounds):
            if calculate_iou(det1, detections2.xyxy[pos]) > 0.1:
                overlap = True
                break
        if not overlap and detections1.confidence[i] > 0.6:
            filtered_detections1.append(det1)
            filtered_class_name1.append(detections1.data['class_name'][i])
            filtered_confidence1.append(detections1.confidence[i])

    filtered_detections2 = []
    filtered_class_name2 = []
    filtered_confidence2 = []
    for i, det2 in enumerate(detections2.xyxy):
        if detections2.confidence[i] > 0.4:
            filtered_detections2.append(det2)
            filtered_class_name2.append(detections2.data['class_name'][i])
            filtered_confidence2.append(detections2.confidence[i])

    if not filtered_detections1 and not filtered_detections2:
        # Initialize empty arrays with the correct shape
        new_xyxy = np.empty((0, 4))
        new_confidence = np.empty((0,))
        new_class_name = np.empty((0,))
        new_class_id = np.empty((0,))
    else:
        # Prepare the arrays for the new Detections object
        new_xyxy = np.array(filtered_detections1 + filtered_detections2)
        new_confidence = np.array(filtered_confidence1 + filtered_confidence2)
        new_class_name = np.array(filtered_class_name1 + filtered_class_name2)

        # Map class names to class IDs
        new_class_id = []
        for class_name in new_class_name:
            if class_name == "Distress":
                new_class_id.append(1)
            elif class_name == "Warning":
                new_class_id.append(3)
            elif class_name == "Rescue":
                new_class_id.append(10)
            elif class_name == "EllisGuard":
                new_class_id.append(10)
            else:
                new_class_id.append(0)  # Assign a default class ID for other classes
        new_class_id = np.array(new_class_id)

    # Create and return the new Detections object
    return Detections(
        xyxy=new_xyxy,
        mask=None,
        confidence=new_confidence,
        class_id=new_class_id,
        tracker_id=None,
        data={'class_name': new_class_name}
    )


# Initialize the models

# Define a function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# Process frames
first_model = get_model(model_id="face-detection-ugzy1/21")
second_model = get_model(model_id="face-detection-ugzy1/21")


# Initialize ByteTrack tracker and bounding box annotator
tracker = sv.ByteTrack(match_thresh=0.1, track_buffer=130)
bounding_box_annotator = sv.BoundingBoxAnnotator()

# Setup video capture and writing
with mss.mss() as sct:
    monitor = sct.monitors[1]
    monitor_dimensions = {'top': 100, 'left': monitor['width'] // 2, 'width': monitor['width'] // 2, 'height': monitor['height'] // 2, 'mon': 1}
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_sink = cv2.VideoWriter('window_output.avi', fourcc, fps, (monitor_dimensions['width'], monitor_dimensions['height']))

    while True:
        img = sct.grab(monitor_dimensions)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results1 = first_model.infer(frame)
        results2 = second_model.infer(frame)
        detections1 = Detections.from_inference(results1[0])
        detections2 = Detections.from_inference(results2[0])

        merged_detections = add_detections(detections1, detections2)
        tracked_detections=tracker.update_with_detections(merged_detections)
        annotated_frame = bounding_box_annotator.annotate(frame, detections=tracked_detections.with_nms(threshold=0.5, class_agnostic=True))
        video_sink.write(annotated_frame)
        cv2.imshow('Window Capture', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video_sink.release()
