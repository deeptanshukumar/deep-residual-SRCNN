import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from config import IMAGE_SIZE
from metrics import calculate_psnr_from_mse


def evaluate_interpolation(test_indices, all_lr_small, all_hr, method):
    total_squared_error = 0.0
    total_pixels = 0

    for pos, idx in enumerate(test_indices):
        pred_small = all_lr_small[idx]
        hr = all_hr[idx]

        pred_small_pil = to_pil_image(pred_small)

        if method == "nearest":
            resample = Image.Resampling.NEAREST
        elif method == "bilinear":
            resample = Image.Resampling.BILINEAR
        elif method == "bicubic":
            resample = Image.Resampling.BICUBIC
        else:
            raise ValueError("method must be nearest, bilinear, or bicubic")

        pred_pil = pred_small_pil.resize((IMAGE_SIZE, IMAGE_SIZE), resample)
        pred = to_tensor(pred_pil)

        total_squared_error += torch.sum((pred - hr) ** 2).item()
        total_pixels += hr.numel()

    mse = total_squared_error / total_pixels
    psnr = calculate_psnr_from_mse(mse)

    return mse, psnr


def run_all_baselines(test_indices, all_lr_small, all_hr):
    baseline_results = {}

    for method in ["nearest", "bilinear", "bicubic"]:
        mse, psnr = evaluate_interpolation(test_indices, all_lr_small, all_hr, method)
        baseline_results[method] = {"MSE": mse, "PSNR": psnr}
        print(f"{method.title():10s} | MSE: {mse:.6f} | PSNR: {psnr:.2f} dB")

    return baseline_results
