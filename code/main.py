import os
import sys

import pandas as pd
import torch

from config import (
    set_seed, get_device, OUTPUT_DIR, CHECKPOINT_DIR,
    NUM_EPOCHS, LEARNING_RATE, DEEP_LR, DEEP_GRAD_CLIP,
)
from data import (
    load_100_image_dataset, create_dataloaders, get_image, preprocess_image,
)
from models import SRCNN, DeepSRCNN, DeepResidualSRCNN, count_parameters
from train import train_model
from evaluate import evaluate_model
from baselines import run_all_baselines
from visualize import (
    get_srcnn_visual, plot_srcnn_comparison,
    plot_training_loss, plot_validation_psnr, plot_model_comparison,
)
from torchvision.transforms.functional import to_pil_image, to_tensor


def run_100_image_experiments():
    print("=" * 60)
    print("100-IMAGE EXPERIMENTS")
    print("=" * 60)

    df, all_hr, all_lr_small, all_lr, train_idx, val_idx, test_idx = load_100_image_dataset()
    train_loader, val_loader, test_loader = create_dataloaders(
        all_hr, all_lr, train_idx, val_idx, test_idx
    )

    device = get_device()

    # 1. Baseline SRCNN
    print("\n--- Training Baseline SRCNN ---")
    srcnn = SRCNN().to(device)
    print(f"SRCNN Parameters: {count_parameters(srcnn):,}")

    srcnn_history = train_model(srcnn, train_loader, val_loader, device, epochs=NUM_EPOCHS, lr=LEARNING_RATE)
    srcnn_test_mse, srcnn_test_psnr = evaluate_model(srcnn, test_loader, device)
    print(f"SRCNN Test MSE : {srcnn_test_mse:.6f}")
    print(f"SRCNN Test PSNR: {srcnn_test_psnr:.2f} dB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(srcnn.state_dict(), os.path.join(OUTPUT_DIR, "srcnn_baseline.pth"))

    # 2. Non-neural baselines
    print("\n--- Non-Neural Baselines ---")
    baseline_results = run_all_baselines(test_idx, all_lr_small, all_hr)

    # 3. SRCNN Visual Comparison
    print("\n--- SRCNN Visual Comparison ---")
    images, titles = get_srcnn_visual(test_idx[0], df, srcnn, device)
    plot_srcnn_comparison(images, titles, save_path=os.path.join(OUTPUT_DIR, "srcnn_visual_comparison.png"))

    # 4. Deep SRCNN (stabilized)
    print("\n--- Training Stabilized Deep SRCNN ---")
    deep_srcnn = DeepSRCNN().to(device)
    print(f"Deep SRCNN Parameters: {count_parameters(deep_srcnn):,}")

    deep_history = train_model(
        deep_srcnn, train_loader, val_loader, device,
        lr=DEEP_LR, grad_clip_norm=DEEP_GRAD_CLIP
    )
    deep_test_mse, deep_test_psnr = evaluate_model(deep_srcnn, test_loader, device)
    print(f"Deep SRCNN Test MSE : {deep_test_mse:.6f}")
    print(f"Deep SRCNN Test PSNR: {deep_test_psnr:.2f} dB")

    torch.save(deep_srcnn.state_dict(), os.path.join(OUTPUT_DIR, "deep_srcnn_stabilized.pth"))

    # 5. Deep Residual SRCNN
    print("\n--- Training Deep Residual SRCNN ---")
    residual_srcnn = DeepResidualSRCNN().to(device)
    print(f"Deep Residual SRCNN Parameters: {count_parameters(residual_srcnn):,}")

    residual_history = train_model(
        residual_srcnn, train_loader, val_loader, device,
        lr=DEEP_LR, grad_clip_norm=DEEP_GRAD_CLIP
    )
    residual_test_mse, residual_test_psnr = evaluate_model(residual_srcnn, test_loader, device)
    print(f"Deep Residual SRCNN Test MSE : {residual_test_mse:.6f}")
    print(f"Deep Residual SRCNN Test PSNR: {residual_test_psnr:.2f} dB")

    torch.save(residual_srcnn.state_dict(), os.path.join(OUTPUT_DIR, "deep_residual_srcnn.pth"))

    # 6. Combined Results Table
    print("\n--- Combined Results ---")
    results = pd.DataFrame([
        {"Method": "Nearest Neighbor", "Test MSE": baseline_results["nearest"]["MSE"], "Test PSNR": baseline_results["nearest"]["PSNR"]},
        {"Method": "Bilinear", "Test MSE": baseline_results["bilinear"]["MSE"], "Test PSNR": baseline_results["bilinear"]["PSNR"]},
        {"Method": "Bicubic", "Test MSE": baseline_results["bicubic"]["MSE"], "Test PSNR": baseline_results["bicubic"]["PSNR"]},
        {"Method": "SRCNN", "Test MSE": srcnn_test_mse, "Test PSNR": srcnn_test_psnr},
        {"Method": "Deep SRCNN", "Test MSE": deep_test_mse, "Test PSNR": deep_test_psnr},
        {"Method": "Deep Residual SRCNN", "Test MSE": residual_test_mse, "Test PSNR": residual_test_psnr},
    ])
    print(results.to_string(index=False))

    # 7. Training Curves
    plot_training_loss(srcnn_history, title="SRCNN Training and Validation Loss",
                       save_path=os.path.join(OUTPUT_DIR, "srcnn_loss_curve.png"))
    plot_validation_psnr(
        [srcnn_history, deep_history],
        ["SRCNN", "Stabilized Deep SRCNN"],
        title="Validation PSNR: SRCNN vs Stabilized Deep SRCNN",
        save_path=os.path.join(OUTPUT_DIR, "validation_psnr_srcnn_vs_deep.png")
    )
    plot_validation_psnr(
        [srcnn_history, deep_history, residual_history],
        ["SRCNN", "Deep SRCNN", "Deep Residual SRCNN"],
        title="Validation PSNR Across SRCNN Variants",
        save_path=os.path.join(OUTPUT_DIR, "validation_psnr_all_srcnn_variants.png")
    )

    # 8. Final Model Comparison
    print("\n--- Final Model Comparison ---")
    idx = test_idx[0]
    original = get_image(df.iloc[idx])
    hr, lr_small, lr = preprocess_image(original)
    bicubic = lr_small.resize((224, 224), Image.Resampling.BICUBIC)

    lr_tensor = to_tensor(lr).unsqueeze(0).to(device)

    srcnn.eval()
    with torch.no_grad():
        srcnn_output = srcnn(lr_tensor).clamp(0, 1)
    srcnn_image = to_pil_image(srcnn_output.squeeze(0).cpu())

    deep_srcnn.eval()
    with torch.no_grad():
        deep_output = deep_srcnn(lr_tensor).clamp(0, 1)
    deep_image = to_pil_image(deep_output.squeeze(0).cpu())

    residual_srcnn.eval()
    with torch.no_grad():
        residual_output = residual_srcnn(lr_tensor).clamp(0, 1)
    residual_image = to_pil_image(residual_output.squeeze(0).cpu())

    images = [lr, bicubic, srcnn_image, deep_image, residual_image, hr]
    titles = ["LR Input", "Bicubic", "SRCNN", "Deep SRCNN", "Deep Residual SRCNN", "Ground Truth"]
    plot_model_comparison(images, titles, save_path=os.path.join(OUTPUT_DIR, "final_model_comparison.png"))

    return results, baseline_results


def run_full_dataset_experiment():
    print("=" * 60)
    print("FULL DATASET EXPERIMENT (800 training images)")
    print("=" * 60)

    from data import load_full_div2k_dataset

    all_hr_train, all_lr_train, all_hr_val, all_lr_val = load_full_div2k_dataset()

    device = get_device()

    train_dataset = torch.utils.data.TensorDataset(all_lr_train, all_hr_train)
    val_dataset = torch.utils.data.TensorDataset(all_lr_val, all_lr_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    residual_srcnn = DeepResidualSRCNN().to(device)

    residual_history = train_model(
        residual_srcnn, train_loader, val_loader, device,
        lr=DEEP_LR, grad_clip_norm=DEEP_GRAD_CLIP
    )

    torch.save(residual_srcnn.state_dict(), os.path.join(OUTPUT_DIR, "deep_residual_srcnn_800.pth"))

    return residual_history


def main():
    set_seed()
    device = get_device()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    results, baselines = run_100_image_experiments()

    print("\n" + "=" * 60)
    print("100-image experiments complete. Results saved to outputs/")

    response = input("\nRun full 800-image dataset experiment? (y/n): ")
    if response.lower() == "y":
        run_full_dataset_experiment()

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
