import numpy as np
import torch
import os
from model import HybridModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = HybridModel().to(device)
model.load_state_dict(torch.load("HAR_final_model.pth", map_location=device))
model.eval()

# Load one sequence from your dataset (IMPORTANT)
seq_path = "PATH_TO_YOUR_SEQUENCE.npy"   # 🔴 CHANGE THIS

seq = np.load(seq_path).astype(np.float32)

# Convert to tensor
seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)

class_names = {
    0: "Walking",
    1: "Running",
    2: "Standing",
    3: "Falling"
}

with torch.no_grad():
    output = model(seq_tensor)
    _, pred = torch.max(output, 1)

print("Prediction:", class_names[pred.item()])