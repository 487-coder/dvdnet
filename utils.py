import torch
import numpy as np


def numpy_to_tensor(np_image_seq, device=None):
    if not isinstance(np_image_seq, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if np_image_seq.ndim != 4:
        raise ValueError("Input must have 4 dimensions: (T, H, W, C)")
    T, H, W, C = np_image_seq.shape
    image_seq = np_image_seq.transpose(0, 3, 1, 2)  # (T, C, H, W)
    tensor_seq = torch.from_numpy(image_seq).float() / 255.0
    tensor_seq = tensor_seq.contiguous().view(1, T * C, H, W)
    if device is not None:
        tensor_seq = tensor_seq.to(device)
    return tensor_seq