# inference/inference.py

import torch
from models.ldm import LDM

def run_inference():
    model = LDM(config={})
    model.eval()

    sample = torch.randn(1, 1, 64, 64, 64)
    output = model(sample)

    print("Inference complete:", output.shape)
