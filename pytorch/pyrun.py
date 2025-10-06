import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import numpy as np
from threading import Thread
import time
import playsound
import os
from pathlib import Path

# --- CONFIG ---
# Primary path: current workspace SavedModels; fallback to previous Midterm path
model_path = r"C:\Subjects 2024\4thYr\CSC-123\Files\SavedModels2\best_ft_model.pt"
fallback_model_path = r"C:\Subjects 2024\4thYr\CSC-123\Midterm\SavedModels2\best_ft_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_width = 1280  # Increased resolution for better eye detection
frame_height = 720

# Eye detection threshold
EYE_AR_THRESH = 0.25  # Lower threshold for more accurate detection

# Fusion weights for fast UI feedback
W_CNN = 0.4
W_EAR = 0.6

# Global variables
eye_frame_counter = 0
last_detection_states = []  # Store last few detection states for consistency
max_state_history = 5  # Keep last 5 frames for temporal consistency

# Fatigue detection timing
closed_start_time = None
alert_active = False
alarm_cooldown_until = 0.0  # epoch seconds

# Temporal stability (debounce/hysteresis) to avoid flicker
stable_state = "open"            # persisted state: "open" or "closed"
candidate_state = None            # transient candidate differing from stable_state
candidate_since = 0.0            # monotonic timestamp when candidate observed
SWITCH_TO_CLOSED_SEC = 0.20      # require 0.20s of closed evidence to flip
SWITCH_TO_OPEN_SEC = 0.25        # require 0.25s of open evidence to flip

# --- LOAD MODEL ---
def load_model():
    try:
        path_to_load = model_path if os.path.exists(model_path) else fallback_model_path
        checkpoint = torch.load(path_to_load, map_location=device, weights_only=True)
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        import torch.nn as nn
        
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights)
        
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )
        
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {path_to_load} on {device}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    print("Failed to load model. Exiting...")
    exit()

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- MEDIAPIPE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,  # Increased for better small face detection
    min_tracking_confidence=0.7   # Increased for better tracking
)

# Eye coordinates
right_eye_coordinates = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye_coordinates = [[263, 362], [387, 373], [386, 374], [385, 380]]

# --- CAMERA ---
def initialize_camera():
    """Initialize camera with better Windows compatibility"""
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            # Set camera properties first
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try multiple times to read a frame (Windows sometimes needs this)
            for attempt in range(5):
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.shape[0] > 0:
                    print(f"Camera {camera_index} initialized successfully: {frame_width}x{frame_height}")
                    print(f"Frame shape: {test_frame.shape}")
                    return cap
                else:
                    print(f"Attempt {attempt + 1}: Camera {camera_index} couldn't read frame")
                    import time
                    time.sleep(0.1)  # Small delay between attempts
            
            print(f"Camera {camera_index} opened but couldn't read frame after 5 attempts")
            cap.release()
        else:
            print(f"Camera {camera_index} could not be opened")
            cap.release()
    
    print("No working camera found. Please check your camera connection.")
    print("Make sure no other applications are using the camera.")
    return None

cap = initialize_camera()
if cap is None:
    print("Exiting due to camera issues...")
    exit()

# --- UTILITY FUNCTIONS ---
def distance(point1, point2):
    return (((point1[:2] - point2[:2])**2).sum())**0.5

def enhance_image(frame):
    """Lightweight enhancement for speed"""
    try:
        # Fast CLAHE only on L channel with milder params
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Enhancement error: {e}")
        return frame

def crop_eye_regions(frame, landmarks_positions, zoom_factor=1.5):
    """Crop and zoom eye regions for better detection of small faces"""
    try:
        if landmarks_positions is None or len(landmarks_positions) == 0:
            return frame
        
        # Get eye landmark points
        left_eye_points = []
        right_eye_points = []
        
        for coord in left_eye_coordinates:
            for point_idx in coord:
                if point_idx < len(landmarks_positions):
                    left_eye_points.append(landmarks_positions[point_idx])
        
        for coord in right_eye_coordinates:
            for point_idx in coord:
                if point_idx < len(landmarks_positions):
                    right_eye_points.append(landmarks_positions[point_idx])
        
        if not left_eye_points or not right_eye_points:
            return frame
        
        # Calculate bounding boxes for both eyes
        left_eye_points = np.array(left_eye_points)
        right_eye_points = np.array(right_eye_points)
        
        # Get combined eye region
        all_eye_points = np.vstack([left_eye_points, right_eye_points])
        x_min, y_min = np.min(all_eye_points, axis=0)[:2]
        x_max, y_max = np.max(all_eye_points, axis=0)[:2]
        
        # Add padding and apply zoom
        eye_width = x_max - x_min
        eye_height = y_max - y_min
        padding_x = int(eye_width * (zoom_factor - 1) / 2)
        padding_y = int(eye_height * (zoom_factor - 1) / 2)
        
        # Ensure coordinates are within frame bounds
        x_min = max(0, int(x_min - padding_x))
        y_min = max(0, int(y_min - padding_y))
        x_max = min(frame.shape[1], int(x_max + padding_x))
        y_max = min(frame.shape[0], int(y_max + padding_y))
        
        # Crop the eye region
        eye_crop = frame[y_min:y_max, x_min:x_max]
        
        if eye_crop.size == 0:
            return frame
        
        # Resize back to original size for processing
        eye_crop_resized = cv2.resize(eye_crop, (frame.shape[1], frame.shape[0]))
        
        return eye_crop_resized
        
    except Exception as e:
        print(f"Eye cropping error: {e}")
        return frame

def apply_temporal_consistency(current_state, confidence):
    """Apply temporal consistency to prevent flickering between states"""
    global last_detection_states, max_state_history
    
    # Add current state to history
    last_detection_states.append((current_state, confidence))
    
    # Keep only last max_state_history frames
    if len(last_detection_states) > max_state_history:
        last_detection_states.pop(0)
    
    # Need at least 3 frames for consistency check
    if len(last_detection_states) < 3:
        return current_state
    
    # Count states in recent history
    closed_count = sum(1 for state, _ in last_detection_states if state == "closed")
    open_count = len(last_detection_states) - closed_count
    
    # Require majority agreement for state change
    if closed_count >= 3:  # At least 3 out of 5 frames must agree
        return "closed"
    elif open_count >= 3:  # At least 3 out of 5 frames must agree
        return "open"
    else:
        # If no clear majority, keep previous state
        return last_detection_states[-2][0] if len(last_detection_states) >= 2 else current_state

def sound_alarm():
    """Play alarm sound on Windows reliably (winsound) with playsound fallback."""
    try:
        alarm_path = Path(__file__).resolve().parent.parent / "app" / "alarm.wav"
        alarm_str = str(alarm_path)
        if not alarm_path.exists():
            print("Alarm file not found:", alarm_str)
            return
        # Prefer winsound for WAV on Windows
        try:
            import winsound
            winsound.PlaySound(alarm_str, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        except Exception as we:
            print("winsound failed, falling back to playsound:", we)
        # Fallback: playsound (may use MCI and sometimes fails)
        playsound.playsound(alarm_str)
    except Exception as e:
        print(f"Error playing alarm: {e}")

def detect_eyes_cnn(frame):
    """Detect if eyes are open or closed using CNN with lightweight preprocessing"""
    try:
        # Lightweight enhancement and fast resize for latency
        enhanced_frame = enhance_image(frame)
        cnn_input_img = cv2.resize(enhanced_frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        pil_image = Image.fromarray(cv2.cvtColor(cnn_input_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            # Class 0 = closed eyes, Class 1 = open eyes
            closed_prob = probabilities[0][0].item()
            open_prob = probabilities[0][1].item()
            
            # Much more conservative threshold - only consider closed if extremely confident
            is_eyes_closed = closed_prob > 0.85  # Very high threshold for closed eyes
            confidence = max(closed_prob, open_prob)
        
        return is_eyes_closed, confidence, closed_prob
    except Exception as e:
        print(f"CNN error: {e}")
        return False, 0.0

def calc_eye_aspect_ratio(landmarks, eye):
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def calc_eye_size(landmarks, eye):
    """Calculate eye size for adaptive threshold"""
    # Calculate eye width and height
    eye_width = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    eye_height = (distance(landmarks[eye[1][0]], landmarks[eye[1][1]]) + 
                 distance(landmarks[eye[2][0]], landmarks[eye[2][1]]) + 
                 distance(landmarks[eye[3][0]], landmarks[eye[3][1]])) / 3.0
    
    # Return normalized eye size (relative to frame size)
    return (eye_width + eye_height) / 2.0


def detect_eyes_landmarks(frame):
    """Detect if eyes are open or closed using facial landmarks with enhancements"""
    try:
        # Apply image enhancements for better small face detection
        enhanced_frame = enhance_image(frame)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw only eye landmarks with minimal display
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
                
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
            
            # Get landmark positions for calculations
            landmarks_positions = []
            for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
                landmarks_positions.append([data_point.x, data_point.y, data_point.z])
            landmarks_positions = np.array(landmarks_positions)
            landmarks_positions[:, 0] *= frame.shape[1]
            landmarks_positions[:, 1] *= frame.shape[0]
            
            # Calculate EAR for both eyes
            left_ear = calc_eye_aspect_ratio(landmarks_positions, left_eye_coordinates)
            right_ear = calc_eye_aspect_ratio(landmarks_positions, right_eye_coordinates)
            ear = (left_ear + right_ear) / 2.0
            
            # Calculate eye sizes for adaptive threshold
            left_eye_size = calc_eye_size(landmarks_positions, left_eye_coordinates)
            right_eye_size = calc_eye_size(landmarks_positions, right_eye_coordinates)
            avg_eye_size = (left_eye_size + right_eye_size) / 2.0
            
            # Dynamic cropping for small faces
            if avg_eye_size < 30:  # Small eyes - apply dynamic cropping
                cropped_frame = crop_eye_regions(frame, landmarks_positions, zoom_factor=2.0)
                # Re-process with cropped frame for better detection
                enhanced_cropped = enhance_image(cropped_frame)
                rgb_cropped = cv2.cvtColor(enhanced_cropped, cv2.COLOR_BGR2RGB)
                cropped_results = face_mesh.process(rgb_cropped)
                
                if cropped_results.multi_face_landmarks:
                    # Use cropped results for better accuracy
                    cropped_landmarks = []
                    for _, data_point in enumerate(cropped_results.multi_face_landmarks[0].landmark):
                        cropped_landmarks.append([data_point.x, data_point.y, data_point.z])
                    cropped_landmarks = np.array(cropped_landmarks)
                    cropped_landmarks[:, 0] *= frame.shape[1]
                    cropped_landmarks[:, 1] *= frame.shape[0]
                    
                    # Recalculate EAR with cropped landmarks
                    left_ear_cropped = calc_eye_aspect_ratio(cropped_landmarks, left_eye_coordinates)
                    right_ear_cropped = calc_eye_aspect_ratio(cropped_landmarks, right_eye_coordinates)
                    ear = (left_ear_cropped + right_ear_cropped) / 2.0
            
            # Much more conservative adaptive thresholds to prevent false positives
            if avg_eye_size < 20:  # Very small eyes
                threshold = 0.08  # Very low threshold - only extremely closed eyes
            elif avg_eye_size < 40:  # Small eyes
                threshold = 0.10  # Very low threshold
            elif avg_eye_size < 60:  # Medium eyes
                threshold = 0.12  # Low threshold
            else:  # Normal/large eyes
                threshold = 0.15  # Conservative threshold
            
            # Only consider eyes closed if EAR is below adaptive threshold
            is_eyes_closed = ear < threshold
            return is_eyes_closed, ear, threshold
        else:
            return False, 0.0, 0.0
            
    except Exception as e:
        print(f"Landmark detection error: {e}")
        return False, 0.0

# --- MAIN LOOP ---
print("Starting Eye Detection... Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera")
            break
        
        if frame is None:
            print("Error: Received None frame from camera")
            break
        
        # Detect eyes using both methods
        try:
            # CNN detection
            cnn_closed, cnn_confidence, cnn_closed_prob = detect_eyes_cnn(frame)
            
            # Landmark detection
            landmark_closed, ear_value, ear_threshold = detect_eyes_landmarks(frame)

            # --- Fast UI fusion (per-frame, blink-responsive) ---
            # Convert EAR to a closed score: >0 means below threshold; clamp to [0,1]
            ear_closed_score = 0.0
            if ear_threshold > 0:
                ear_closed_score = max(0.0, min(1.0, (ear_threshold - ear_value) / ear_threshold))
            ui_score = W_CNN * cnn_closed_prob + W_EAR * ear_closed_score
            ui_closed = (cnn_closed or landmark_closed) or (ui_score > 0.5)
            
            # --- ULTRA-CONSERVATIVE DETECTION LOGIC ---
            # Check if face is too far or detection is unreliable
            if cnn_confidence < 0.7 and ear_value < 0.15:
                # Face too far or unreliable detection - default to open
                final_result = False
                method = "Default (Unreliable detection)"
            elif cnn_confidence < 0.5:
                # CNN not confident - rely only on landmarks with very strict criteria
                if ear_value < 0.08:  # Extremely low EAR
                    final_result = True
                    method = "Landmarks (Extremely low EAR)"
                else:
                    final_result = False
                    method = "Landmarks (Conservative)"
            elif landmark_closed and cnn_closed:
                # Both methods agree on closed - but require high confidence
                if cnn_confidence > 0.8 and ear_value < 0.10:
                    final_result = True
                    method = "Both (High confidence)"
                else:
                    final_result = False
                    method = "Both (Low confidence - defaulting to open)"
            elif not landmark_closed and not cnn_closed:
                # Both methods agree on open
                final_result = False
                method = "Both (Open)"
            else:
                # Methods disagree - use ultra-conservative approach
                if cnn_confidence > 0.9 and cnn_closed:  # Extremely high CNN confidence
                    final_result = True
                    method = "CNN (Ultra-high confidence)"
                elif ear_value < 0.06:  # Extremely low EAR
                    final_result = True
                    method = "Landmarks (Extremely low EAR)"
                else:
                    # Default to open unless extremely confident
                    final_result = False
                    method = "Conservative (Default to open)"
            
            # --- Temporal stability with hysteresis to avoid 1-frame flips ---
            now = time.monotonic()
            # Use conservative result or strong fused score for alert pipeline
            alert_observed_closed = final_result or (ui_score > 0.7)
            observed_state = "closed" if alert_observed_closed else "open"
            if observed_state == stable_state:
                candidate_state = None
                candidate_since = 0.0
            else:
                if candidate_state != observed_state:
                    candidate_state = observed_state
                    candidate_since = now
                else:
                    dwell = now - candidate_since
                    if candidate_state == "closed" and dwell >= SWITCH_TO_CLOSED_SEC:
                        stable_state = "closed"
                        candidate_state = None
                        candidate_since = 0.0
                    elif candidate_state == "open" and dwell >= SWITCH_TO_OPEN_SEC:
                        stable_state = "open"
                        candidate_state = None
                        candidate_since = 0.0

            # --- Fatigue detection over time (1.5s continuous closed) ---
            threshold_seconds = 1.5
            if stable_state == "closed":
                if closed_start_time is None:
                    closed_start_time = now
                elapsed = now - closed_start_time
                if elapsed >= threshold_seconds and not alert_active:
                    alert_active = True
                    if now >= alarm_cooldown_until:
                        alarm_cooldown_until = now + 3.0
                        Thread(target=sound_alarm, daemon=True).start()
            else:
                closed_start_time = None
                alert_active = False

            # Overlay status: show fast per-frame UI state for blinks
            # and show fatigue alert when active
            if alert_active:
                cv2.putText(frame, "Alert: Fatigue detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if ui_closed:
                cv2.putText(frame, "Eyes Closed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Eyes Open", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error in detection: {e}")
            cv2.putText(frame, "Eyes Open", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Eye Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit key pressed. Stopping...")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Eye detection stopped.")
