#need to add utils
import torch
import os

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path):
    return torch.load(path, map_location="cpu")
