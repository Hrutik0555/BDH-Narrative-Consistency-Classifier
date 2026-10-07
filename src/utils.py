"""Text preprocessing helpers."""
import random
import re

import numpy as np
import torch


def sentence_split(text):
    """Split text into sentences, dropping fragments of 5 characters or fewer."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_label_tensor(labels, device):
    """Convert labels to a LongTensor on `device` without the
    `torch.tensor(tensor)` copy-construct warning."""
    if isinstance(labels, torch.Tensor):
        return labels.detach().clone().to(device)
    return torch.tensor(labels, dtype=torch.long, device=device)
