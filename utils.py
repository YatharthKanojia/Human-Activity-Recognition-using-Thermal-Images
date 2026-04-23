import cv2
import numpy as np
import torch

def preprocess(img):
    img = cv2.resize(img,(128,128))
    img = img / 255.0
    return img.astype(np.float32)

def optical_flow_batch(sequences):
    batch_flows = []

    for seq in sequences:
        flows = []
        seq = seq.detach().cpu().numpy()

        for i in range(len(seq)-1):
            f1 = (seq[i]*255).astype(np.uint8)
            f2 = (seq[i+1]*255).astype(np.uint8)

            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            flow = np.transpose(flow,(2,0,1))
            flows.append(flow)

        batch_flows.append(np.array(flows))

    return torch.tensor(np.array(batch_flows)).float()