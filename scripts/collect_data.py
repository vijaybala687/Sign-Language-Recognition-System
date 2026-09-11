import cv2
import mediapipe as mp
import numpy as np
import os

# --- Configurations ---
DATA_PATH = os.path.join('..', 'data', 'processed')
ACTIONS = np.array(['father','hello','yes','no','help','please'])  # Added more actions for a richer dataset
NO_SEQUENCES = 15     # Reduced to 15 videos per sign for a quick test
SEQUENCE_LENGTH = 30  # 30 frames per video (~1 second of motion)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create folders
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# NOTE: Change the index below (0, 1, 2, or -1) based on what worked for your Fedora laptop!
cap = cv2.VideoCapture(1) 

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in ACTIONS:
        
        # --- NEW: Pause and wait for the user to be ready ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Put instructions on the screen
            cv2.putText(image, f"Word: {action.upper()}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Press SPACEBAR when ready to record", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Sign Language Tracker', image)
            
            # Wait for spacebar (key 32) to break the loop and start recording
            if cv2.waitKey(10) & 0xFF == 32: 
                break
        
        # --- Start recording the sequences ---
        for sequence in range(NO_SEQUENCES):
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret or frame is None: break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Give a 1.5-second pause between each video so you can reset your hands
                if frame_num == 0:
                    cv2.putText(image, 'RESET YOUR HANDS...', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Recording {action.upper()} - Video {sequence + 1}/{NO_SEQUENCES}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Sign Language Tracker', image)
                    cv2.waitKey(1500) 
                else:
                    cv2.putText(image, f'Recording {action.upper()} - Video {sequence + 1}/{NO_SEQUENCES}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Sign Language Tracker', image)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()