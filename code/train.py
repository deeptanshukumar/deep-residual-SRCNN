import torch
import torch.nn as nn

from metrics import calculate_psnr_from_mse


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=1e-3,
    grad_clip_norm=None,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_psnrs = []

    best_val_psnr = -float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            output = model(lr_batch)
            loss = criterion(output, hr_batch)

            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )

            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        total_squared_error = 0.0
        total_val_pixels = 0

        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device, non_blocking=True)
                hr_batch = hr_batch.to(device, non_blocking=True)

                output = model(lr_batch).clamp(0, 1)
                total_squared_error += torch.sum(
                    (output - hr_batch) ** 2
                ).item()
                total_val_pixels += hr_batch.numel()

        val_mse = total_squared_error / total_val_pixels
        val_psnr = calculate_psnr_from_mse(val_mse)

        train_losses.append(train_loss)
        val_losses.append(val_mse)
        val_psnrs.append(val_psnr)

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val PSNR: {val_psnr:.2f} dB"
        )

    model.load_state_dict(best_state)
    model.to(device)

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_psnr": val_psnrs,
        "best_val_psnr": best_val_psnr,
    }

    return history
