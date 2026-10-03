import math

import torch


def calculate_psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def calculate_mse(pred, target):
    return torch.mean((pred - target) ** 2).item()
