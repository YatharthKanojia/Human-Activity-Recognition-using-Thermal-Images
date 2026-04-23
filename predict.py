import torch
import numpy as np
import os
from model import HybridModel
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# safer path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "HAR_final_model.pth")

model = HybridModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

class_names = {
    0: "Walking",
    1: "Running",
    2: "Standing",
    3: "Falling"
}

def predict_sequence(frames):
    frames = [preprocess(f) for f in frames]
    seq = np.array(frames)

    seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(seq_tensor)
        _, pred = torch.max(output, 1)

    return class_names[pred.item()]