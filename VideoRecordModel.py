import numpy as np
import cv2
import supervision as sv
from inference import InferencePipeline, get_model
from inference.core.interfaces.camera.entities import VideoFrame
from supervision.detection.core import Detections
from inference.core.interfaces.stream.sinks import VideoFileSink
import time
import os
from dotenv import load_dotenv


load_dotenv()  # This loads the environment variables from `.env` into the environment

# Access your environment variable
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
print("Roboflow API Key:", roboflow_api_key)
# Load the video source and get its resolution
video_source = "a2.avi"
cap = cv2.VideoCapture(video_source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rate = int(cap.get(cv2.CAP_PROP_FPS))-5
output_size = (width, height)
video_sink = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), rate, output_size)
display_window_duration = 8  # seconds to display the small window after distress is no longer present
last_time_distress_detected = None
roi_to_display = None
last_roi = None
last_detection_center = None
roi_change_threshold = 2000
# Initialize the ByteTrack tracker
tracker = sv.ByteTrack(match_thresh=0.1, track_buffer=130)
# Initialize annotators for bounding boxes and labels
bounding_box_annotator = sv.BoundingBoxAnnotator()
distress_status = {}
last_detection_center = None
# Load the second model
second_model = get_model(model_id="persondetection2-lrsz4/6")

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

def add_detections(detections1: Detections, detections2: Detections) -> Detections:
    # Filter out detections from the first model that overlap with detections from the second model
    filtered_detections1 = []
    filtered_confidence1 = []
    filtered_class_name1 = []
    for i, det1 in enumerate(detections1.xyxy):
        overlap = False
        for det2 in detections2.xyxy:
            if calculate_iou(det1, det2) > 0.1:  # Overlap threshold
                overlap = True
                break
        if not overlap and detections1.confidence[i] > 0.6:  # Filter by confidence level of the first model
            filtered_detections1.append(det1)
            filtered_confidence1.append([detections1.confidence[i]])  # Make sure it's a 2D array
            filtered_class_name1.append([detections1.data['class_name'][i]])  # Make sure it's a 2D array

    # Filter detections from the second model based on confidence level
    filtered_detections2 = []
    filtered_confidence2 = []
    filtered_class_name2 = []
    for i, det2 in enumerate(detections2.xyxy):
        if detections2.confidence[i] > 0.4:  # Filter by confidence level of the second model
            filtered_detections2.append(det2)
            filtered_confidence2.append([detections2.confidence[i]])  # Make sure it's a 2D array
            filtered_class_name2.append([detections2.data['class_name'][i]])  # Make sure it's a 2D array

    # Ensure all arrays are 2D arrays before concatenation
    filtered_detections1 = np.array(filtered_detections1).reshape(-1, 4)
    filtered_detections2 = np.array(filtered_detections2).reshape(-1, 4)
    filtered_confidence1 = np.array(filtered_confidence1).reshape(-1, 1)
    filtered_confidence2 = np.array(filtered_confidence2).reshape(-1, 1)
    filtered_class_name1 = np.array(filtered_class_name1).reshape(-1, 1)
    filtered_class_name2 = np.array(filtered_class_name2).reshape(-1, 1)

    # Concatenate non-overlapping detections from the first model with filtered detections from the second model
    new_xyxy = np.concatenate((filtered_detections1, filtered_detections2), axis=0)
    new_confidence = np.concatenate((filtered_confidence1, filtered_confidence2), axis=0)  # Concatenate 2D arrays
    new_class_name = np.concatenate((filtered_class_name1, filtered_class_name2), axis=0)  # Ensure 2D array

    # Update class IDs based on class names
    new_class_id = []
    for class_name in new_class_name.flatten():
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

    # Create a new Detections object with the concatenated attributes
    new_detections = Detections(
        xyxy=new_xyxy,
        mask=None,  # Assuming mask is not used, set to None
        confidence=new_confidence.flatten(),  # Flatten to 1D array
        class_id=new_class_id,
        tracker_id=None,  # Assuming tracker_id is not used, set to None
        data={'class_name': new_class_name.flatten()}  # Flatten to 1D array
    )
    return new_detections
def my_custom_sink(predictions1: dict, video_frame: VideoFrame):
    # Process predictions from the first model
    roi_change_threshold = 10
    last_roi = None
    last_detection_center = None
    current_time = time.time()
    last_time_distress_detected = None
    detections1 = sv.Detections.from_inference(predictions1)
    # Run inference with the second model on the same video frame
    results2 = second_model.infer(video_frame.image)
    detections2 = sv.Detections.from_inference(results2[0])
    detections = add_detections(detections1, detections2)
    # Annotate the main frame with bounding boxes and labels
    annotated_frame = bounding_box_annotator.annotate(video_frame.image.copy(), detections=detections.with_nms(threshold=0.5, class_agnostic=True))
    tracked_detections=tracker.update_with_detections(detections)
    distress_found = False
    for i, class_name in enumerate(detections.data['class_name']):
        if class_name == "Distress":
            distress_found = True
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if last_detection_center is None or np.linalg.norm(np.array(current_center) - np.array(last_detection_center)) > roi_change_threshold:
                roi = video_frame.image[y1-100:y2+100, x1-100:x2+200]
                roi_to_display = cv2.resize(roi, (512, 328 - 40))  # 328 total height, 40 for the banner
                last_detection_center = current_center
            last_time_distress_detected = current_time

    # Only update the small window if distress is still being detected or within the display window duration
    if last_time_distress_detected and (current_time - last_time_distress_detected <= display_window_duration):
        if roi_to_display is not None:
            display_roi_with_banner(annotated_frame, roi_to_display)

    # Initialize a variable to store the region of interest from the original, unannotated frame
    
    
    
    # Display the annotated frame
    cv2.imshow('Annotated Frame', annotated_frame)
    cv2.waitKey(1)
    video_sink.write(annotated_frame)

# Initialize the inference pipeline and other setups remain unchanged
def display_roi_with_banner(frame, roi):
    small_win_x = width - 512 - 10
    small_win_y = height - 328 - 10
    banner_height = 40
    banner_area = np.zeros((banner_height, 512, 3), dtype=np.uint8)
    cv2.rectangle(banner_area, (0, 0), (512, banner_height), (0, 0, 200), cv2.FILLED)
    combined_display = np.vstack((banner_area, roi))
    cv2.rectangle(combined_display, (0, 0), (512, 328), (0, 0, 150), 10)
    frame[small_win_y:small_win_y+328, small_win_x:small_win_x+512] = combined_display


# Initialize the inference pipeline for the first model
pipeline = InferencePipeline.init(
    model_id="persondetection-jugx5/2",
    video_reference=video_source,
    on_prediction=my_custom_sink,
)

# Start the pipeline
pipeline.start()

pipeline.join()

video_sink.release()






'''import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import optuna

# Parameters
frame_size = (224, 224)
num_frames = 30
batch_size = 16
epochs = 10

def resize_video(video_path, target_size):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, target_size)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    cap.release()
    return np.array(frames)

def extract_frames(video_path, target_size, num_frames):
    frames = resize_video(video_path, target_size)
    total_frames = len(frames)

    sampled_frames = []
    step = max(1, total_frames // num_frames)
    for i in range(0, total_frames, step):
        sampled_frames.append(frames[i])
        if len(sampled_frames) == num_frames:
            break

    return np.array(sampled_frames)

def augment_frames(frames, augmentation_generator):
    augmented_frames = []
    for frame in frames:
        # Add a dimension to the frame to make it compatible with the generator
        frame_with_batch_and_channel_dimension = np.expand_dims(frame, axis=0)
        frame_with_batch_and_channel_dimension = np.expand_dims(frame_with_batch_and_channel_dimension, axis=-1)  # Add channel dimension
        augmented_frame = next(augmentation_generator.flow(frame_with_batch_and_channel_dimension, batch_size=1))[0]
        augmented_frame = np.squeeze(augmented_frame, axis=-1)  # Remove channel dimension for consistency
        augmented_frames.append(augmented_frame)
    return np.array(augmented_frames)

def load_videos(directory, target_size, num_frames, label, augmentation_generator=None):
    videos = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            video_path = os.path.join(directory, filename)
            frames = extract_frames(video_path, target_size, num_frames)
            if augmentation_generator is not None:
                frames = augment_frames(frames, augmentation_generator)
            videos.append(frames)
            labels.append(label)

    videos_padded = pad_sequences(videos, maxlen=num_frames, padding='post', dtype='float32')
    return videos_padded, np.array(labels)

# Data augmentation
augmentation_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load videos with augmentation
drowning_videos, drowning_labels = load_videos('Model1/drowning', frame_size, num_frames, 1, augmentation_generator)
not_drowning_videos, not_drowning_labels = load_videos('Model1/not_drowning', frame_size, num_frames, 0, augmentation_generator)

# Combine and shuffle data
X = np.concatenate((drowning_videos, not_drowning_videos))
y = np.concatenate((drowning_labels, not_drowning_labels))
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Normalize pixel values
X = X / 255.0

def create_model(learning_rate, lstm_units):
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(num_frames, *frame_size, 1)),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.5),
        LSTM(lstm_units),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    lstm_units = trial.suggest_categorical('lstm_units', [64, 128, 256])

    # Build and compile the model
    model = create_model(learning_rate, lstm_units)

    # Train the model
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

    # Get the validation accuracy
    val_accuracy = history.history['val_accuracy'][-1]

    # Print the trial number and its result
    print(f"Trial {trial.number}: Validation Accuracy = {val_accuracy}")

    # Return the validation accuracy
    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # You can adjust the number of trials

# Retrieve the best hyperparameters
best_hyperparams = study.best_params
print("Best hyperparameters:", best_hyperparams)

# Train the final model with the best hyperparameters
best_model = create_model(best_hyperparams['learning_rate'], best_hyperparams['lstm_units'])
best_model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Save the best model
best_model.save('lstm_drowning_detection_best.keras')'''
