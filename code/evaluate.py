import torch

from metrics import calculate_psnr_from_mse


def evaluate_model(model, loader, device):
    model.eval()

    total_squared_error = 0.0
    total_pixels = 0

    with torch.no_grad():
        for lr, hr in loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            output = model(lr).clamp(0, 1)
            total_squared_error += torch.sum((output - hr) ** 2).item()
            total_pixels += hr.numel()

    mse = total_squared_error / total_pixels
    psnr = calculate_psnr_from_mse(mse)

    return mse, psnr
