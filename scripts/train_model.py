import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Configurations ---
DATA_PATH = os.path.join('..', 'data', 'processed')
ACTIONS = np.array(['hello', 'father', 'help', 'no', 'please', 'yes'])  # Added more actions for a richer dataset   
NO_SEQUENCES = 15
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 225 # 33*3 (pose) + 21*3 (left hand) + 21*3 (right hand)

# --- 2. Load the Data ---
print("Loading data...")
label_map = {label:num for num, label in enumerate(ACTIONS)}
sequences, labels = [], []

for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert lists to PyTorch Tensors
X = torch.tensor(np.array(sequences), dtype=torch.float32)
y = torch.tensor(np.array(labels), dtype=torch.long)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data loaded! Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# --- 3. Define the Lightweight Neural Network ---
# This acts as the Temporal Modeling Phase (Phase 4)
class SignLanguageModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLanguageModel, self).__init__()
        # LSTM processes the sequence of frames
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Fully Connected layer for classification (Phase 5)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Pass data through LSTM
        lstm_out, _ = self.lstm(x)
        # We only want the output from the very last frame of the video
        last_frame_out = lstm_out[:, -1, :] 
        # Pass it to the final classification layer
        out = self.fc(last_frame_out)
        return out

# Initialize the model, loss function, and optimizer
model = SignLanguageModel(input_size=FEATURE_SIZE, hidden_size=64, num_classes=len(ACTIONS))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. Train the Model ---
EPOCHS = 100
print("\nStarting Training...")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds, y_test)
            # Calculate accuracy
            _, predicted_classes = torch.max(test_preds, 1)
            correct = (predicted_classes == y_test).sum().item()
            accuracy = correct / len(y_test) * 100
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Test Accuracy: {accuracy:.2f}%")

# --- 5. Save the Weights ---
os.makedirs(os.path.join('..', 'models', 'checkpoints'), exist_ok=True)
model_path = os.path.join('..', 'models', 'checkpoints', 'sign_model.pth')
torch.save(model.state_dict(), model_path)
print(f"\nTraining Complete! Model saved to {model_path}")