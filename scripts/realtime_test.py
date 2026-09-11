import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import math

# --- 1. Configurations ---
# Must match Colab exactly!
TARGET_WORDS = [
    'hello', 'help', 'father', 'mother', 'book', 'drink', 'computer', 
    'chair', 'clothes', 'candy', 'cousin', 'deaf', 'fine', 'finish', 
    'go', 'hearing', 'language', 'later', 'like', 'man', 'no', 
    'orange', 'pizza', 'play', 'shirt', 'sign', 'student', 'teacher', 
    'walk', 'yes'
]
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 225

# --- 2. Rebuild the Transformer Architecture ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SignTransformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=512, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)

# Initialize Model and Load Weights
print("Loading AI Model...")
model = SignTransformer(FEATURE_SIZE, 256, 8, 3, len(TARGET_WORDS))
model_path = os.path.join('..', 'models', 'checkpoints', 'wlasl_30_words.pth')
# We map_location='cpu' because your laptop doesn't have the Colab GPU
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# --- 3. UNIVERSAL NORMALIZATION ---
def normalize_skeleton(frame_data):
    if np.sum(frame_data) == 0: return frame_data
    nose_x, nose_y, nose_z = frame_data[0], frame_data[1], frame_data[2]
    l_sh_x, l_sh_y = frame_data[11*3], frame_data[11*3+1]
    r_sh_x, r_sh_y = frame_data[12*3], frame_data[12*3+1]
    shoulder_width = math.sqrt((l_sh_x - r_sh_x)**2 + (l_sh_y - r_sh_y)**2)
    scale = shoulder_width if shoulder_width > 0.05 else 1.0 
    
    normalized_frame = np.zeros_like(frame_data)
    for i in range(0, len(frame_data), 3):
        normalized_frame[i]   = (frame_data[i] - nose_x) / scale
        normalized_frame[i+1] = (frame_data[i+1] - nose_y) / scale
        normalized_frame[i+2] = (frame_data[i+2] - nose_z) / scale
    return normalized_frame

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    raw_frame = np.concatenate([pose, lh, rh])
    return normalize_skeleton(raw_frame) # Apply math before saving!

# --- 4. Live Webcam Inference ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
sequence = []
current_word = "Waiting for sign..."
confidence = 0.0

# NOTE: Remember you said your Fedora laptop camera was on index 1 earlier! 
# Change to 0 or 2 if the camera doesn't open.
cap = cv2.VideoCapture(1)

print("Opening Webcam...")
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Custom Sci-Fi Blue styling for landmarks
        custom_style = mp_drawing.DrawingSpec(color=(255, 195, 0), thickness=2, circle_radius=2) # Neon Cyan
        custom_connections = mp_drawing.DrawingSpec(color=(255, 140, 0), thickness=2, circle_radius=2) # Deep Blue

        # Draw hands with the new style
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  custom_style, custom_connections)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  custom_style, custom_connections)
        
        # Extract and Normalize
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:] # Keep rolling buffer of 30 frames
        
        # If we have a full second of video (30 frames), predict!
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                res = model(input_data)
            
            probabilities = torch.softmax(res, dim=1).squeeze()
            max_prob_index = torch.argmax(probabilities).item()
            confidence = probabilities[max_prob_index].item()
            
            # Only update text if AI is highly confident (85%+)
            if confidence > 0.85:
                current_word = TARGET_WORDS[max_prob_index]

        # --- UI Overlay ---
        cv2.rectangle(image, (0, 400), (640, 480), (245, 117, 16), -1)
        cv2.putText(image, f"Prediction: {current_word.upper()}", (10, 440), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, f"Confidence: {confidence*100:.1f}%", (10, 470), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Real-Time Edge AI Translator', image)
        
        # Press 'q' to quit, 'c' to clear the current word
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_word = "Waiting for sign..."
            sequence = [] # Flush buffer

cap.release()
cv2.destroyAllWindows()