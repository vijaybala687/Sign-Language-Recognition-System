import math
import os

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, Response, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "wlasl_30_words.pth")

TARGET_WORDS = [
    "hello",
    "help",
    "father",
    "mother",
    "book",
    "drink",
    "computer",
    "chair",
    "clothes",
    "candy",
    "cousin",
    "deaf",
    "fine",
    "finish",
    "go",
    "hearing",
    "language",
    "later",
    "like",
    "man",
    "no",
    "orange",
    "pizza",
    "play",
    "shirt",
    "sign",
    "student",
    "teacher",
    "walk",
    "yes",
]
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 225

app = Flask(__name__, template_folder="templates")

current_prediction = "Waiting..."
confidence_score = 0.0
last_confident_word = "WAITING"
last_confident_score = 0.0
DISPLAY_CONFIDENCE_THRESHOLD = 0.9
sequence_buffer = []


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
        return x + self.pe[:, : x.size(1), :]


class SignTransformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True,
        )
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


def normalize_skeleton(frame_data):
    if np.sum(frame_data) == 0:
        return frame_data

    nose_x, nose_y, nose_z = frame_data[0], frame_data[1], frame_data[2]
    l_sh_x, l_sh_y = frame_data[11 * 3], frame_data[11 * 3 + 1]
    r_sh_x, r_sh_y = frame_data[12 * 3], frame_data[12 * 3 + 1]
    shoulder_width = math.sqrt((l_sh_x - r_sh_x) ** 2 + (l_sh_y - r_sh_y) ** 2)
    scale = shoulder_width if shoulder_width > 0.05 else 1.0

    normalized_frame = np.zeros_like(frame_data)
    for i in range(0, len(frame_data), 3):
        normalized_frame[i] = (frame_data[i] - nose_x) / scale
        normalized_frame[i + 1] = (frame_data[i + 1] - nose_y) / scale
        normalized_frame[i + 2] = (frame_data[i + 2] - nose_z) / scale
    return normalized_frame


def extract_keypoints(results):
    pose = (
        np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 3)
    )
    left_hand = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    right_hand = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    raw_frame = np.concatenate([pose, left_hand, right_hand])
    return normalize_skeleton(raw_frame)


def load_model():
    model = SignTransformer(FEATURE_SIZE, 256, 8, 3, len(TARGET_WORDS))
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if isinstance(state_dict, dict):
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned[key.replace("module.", "", 1)] = value
            else:
                cleaned[key] = value
        state_dict = cleaned

    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def open_camera():
    for camera_index in (0, 1, 2):
        capture = cv2.VideoCapture(camera_index)
        if capture.isOpened():
            return capture
    raise RuntimeError("No webcam found on the system.")


def update_prediction_from_sequence(sequence):
    global current_prediction, confidence_score, last_confident_word, last_confident_score

    if len(sequence) < SEQUENCE_LENGTH:
        current_prediction = last_confident_word
        confidence_score = last_confident_score
        return

    input_data = torch.tensor(np.array(sequence, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_data)

    probabilities = torch.softmax(logits, dim=1).squeeze()
    best_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[best_index].item())

    if confidence >= DISPLAY_CONFIDENCE_THRESHOLD:
        predicted_word = TARGET_WORDS[best_index].upper()
        current_prediction = predicted_word
        confidence_score = confidence
        last_confident_word = predicted_word
        last_confident_score = confidence
    else:
        current_prediction = last_confident_word
        confidence_score = last_confident_score


def generate_frames():
    global sequence_buffer

    capture = open_camera()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            custom_style = mp_drawing.DrawingSpec(color=(255, 195, 0), thickness=2, circle_radius=2)
            custom_connections = mp_drawing.DrawingSpec(color=(255, 140, 0), thickness=2, circle_radius=2)
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                custom_style,
                custom_connections,
            )
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                custom_style,
                custom_connections,
            )

            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints)
            sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                update_prediction_from_sequence(sequence_buffer)

            cv2.rectangle(image, (0, 400), (640, 480), (245, 117, 16), -1)
            cv2.putText(
                image,
                f"Prediction: {current_prediction}",
                (10, 440),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"Confidence: {confidence_score * 100:.1f}%",
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 255),
                2,
                cv2.LINE_AA,
            )

            ret, buffer = cv2.imencode(".jpg", image)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    capture.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_prediction")
def get_prediction():
    if confidence_score < DISPLAY_CONFIDENCE_THRESHOLD:
        return jsonify({"word": last_confident_word, "confidence": last_confident_score})
    return jsonify({"word": current_prediction, "confidence": confidence_score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
