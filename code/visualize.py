import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from config import IMAGE_SIZE
from data import get_image, preprocess_image


def get_srcnn_visual(idx, df, model, device):
    original = get_image(df.iloc[idx])
    hr, lr_small, lr = preprocess_image(original)

    nearest = lr_small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    bilinear = lr_small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    bicubic = lr_small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)

    lr_tensor = to_tensor(lr).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(lr_tensor).clamp(0, 1)

    srcnn_image = to_pil_image(output.squeeze(0).cpu())

    return [lr, nearest, bilinear, bicubic, srcnn_image, hr], [
        "LR Input", "Nearest", "Bilinear", "Bicubic", "SRCNN", "Ground Truth"
    ]


def plot_srcnn_comparison(images, titles, save_path=None):
    plt.figure(figsize=(18, 4))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_training_loss(history, title="Training and Validation Loss", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_validation_psnr(histories, labels, title="Validation PSNR", save_path=None):
    plt.figure(figsize=(8, 5))
    for history, label in zip(histories, labels):
        plt.plot(history["val_psnr"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_model_comparison(images, titles, save_path=None):
    plt.figure(figsize=(20, 4))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()
