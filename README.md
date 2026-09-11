# Sign Language Recognition System

A real-time sign language recognition project that uses MediaPipe pose and hand landmarks plus a transformer-based classifier to detect sign words from a webcam stream and display the predicted word in a browser UI.

## Overview

This project captures live video from a webcam, extracts pose and hand keypoints, normalizes the skeleton, and feeds a sliding window of frames into a PyTorch model. The app then predicts the current sign and renders the live video stream plus word prediction through a Flask web interface.

The interface is built in Flask and served from `app.py` with HTML/CSS in `templates/index.html`.

## Features

- Real-time webcam capture
- MediaPipe Holistic landmark extraction
- Pose + hand normalization for sequence-based classification
- Sliding-window sign recognition using a transformer model
- Live web dashboard with video feed and confidence display
- Support for multiple sign words such as hello, help, father, no, yes, and others

## Supported words

The model currently targets the following sign classes:

- hello
- help
- father
- mother
- book
- drink
- computer
- chair
- clothes
- candy
- cousin
- deaf
- fine
- finish
- go
- hearing
- language
- later
- like
- man
- no
- orange
- pizza
- play
- shirt
- sign
- student
- teacher
- walk
- yes

## Project structure

```text
.
├── app.py                    # Flask application and real-time inference entry point
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── data/
│   └── processed/           # Normalized keypoint sequences used for training
├── models/
│   └── checkpoints/        # Saved trained model weights
├── scripts/
│   ├── collect_data.py      # Record webcam landmarks for a target sign
│   ├── realtime_test.py     # Optional manual inference/debug script
│   └── train_model.py      # Training script for the classifier
├── src/                     # Supporting project source files
├── templates/
│   └── index.html           # Web UI for showing the live stream and prediction
├── sign_language_project_analysis.ipynb
└── venv/                   # Local Python environment (not typically committed)
```

## Prerequisites

- Python 3.10+
- A working webcam
- A compatible GPU is optional; the project can run on CPU
- Linux/macOS/Windows with a Python environment available

## Setup

From the project root:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

The app streams the webcam feed and fetches the predicted sign word from `/get_prediction` every 500 ms.

## Data collection

To create or extend the dataset:

```bash
python scripts/collect_data.py
```

This script opens the webcam and records sequences for the actions defined in the script. It saves per-frame keypoints under `data/processed/<word>/<sequence>/frame.npy`.

## Model training

```bash
python scripts/train_model.py
```

This loads the collected sequences, trains a classifier, and saves the model to:

```text
models/checkpoints/sign_model.pth
```

## Notes

- The app currently looks for the model in `models/checkpoints/wlasl_30_words.pth` as defined in `app.py`.
- Some scripts in the repository may still reference older training constants or alternate camera indexes; adjust them if your local setup requires a different webcam index.
- The project relies on MediaPipe and OpenCV, so camera permissions must be enabled for local use.

## Troubleshooting

- If no webcam is detected, try changing the camera index in `scripts/collect_data.py` or the detector logic in `app.py`.
- If the model does not load, verify that the checkpoint exists in `models/checkpoints/`.
- For a fresh environment, reinstall dependencies with `pip install -r requirements.txt`.

## License

This project is intended for educational and research use in sign-language recognition experiments.
