import os
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_tensor

from config import (
    BATCH_SIZE,
    CROP_SIZE,
    DATA_PATH,
    DATA_URL,
    IMAGE_SIZE,
    LR_SIZE,
    SEED,
    NUM_TRAIN_IMAGES,
    NUM_VAL_IMAGES,
    NUM_TEST_IMAGES,
    NUM_TRAIN_IMAGES_FULL,
    NUM_VAL_IMAGES_FULL,
    NUM_TEST_IMAGES_FULL,
    DIV2K_TRAIN_URL,
    DIV2K_VALID_URL,
    DIV2K_TRAIN_DIR,
    DIV2K_VALID_DIR,
)


def download_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        os.system(f'wget -c --tries=20 --timeout=30 "{DATA_URL}" -O "{DATA_PATH}"')


def download_div2k():
    if not os.path.exists(DIV2K_TRAIN_DIR):
        os.makedirs(DIV2K_TRAIN_DIR, exist_ok=True)
        os.system(f'wget -c "{DIV2K_TRAIN_URL}" -O "/tmp/DIV2K_train_HR.zip"')
        os.system(f'unzip -q "/tmp/DIV2K_train_HR.zip" -d "{DIV2K_TRAIN_DIR}"')

    if not os.path.exists(DIV2K_VALID_DIR):
        os.makedirs(DIV2K_VALID_DIR, exist_ok=True)
        os.system(f'wget -c "{DIV2K_VALID_URL}" -O "/tmp/DIV2K_valid_HR.zip"')
        os.system(f'unzip -q "/tmp/DIV2K_valid_HR.zip" -d "{DIV2K_VALID_DIR}"')


def get_image(row):
    image_bytes = row["image"]["bytes"]
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def preprocess_image(img):
    img = img.convert("RGB")

    w, h = img.size
    left = max((w - CROP_SIZE) // 2, 0)
    top = max((h - CROP_SIZE) // 2, 0)

    hr = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    hr = hr.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)

    lr_small = hr.resize((LR_SIZE, LR_SIZE), Image.Resampling.BILINEAR)
    lr = lr_small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    return hr, lr_small, lr


def load_and_preprocess_all(df):
    all_lr = []
    all_hr = []
    all_lr_small = []

    for i in range(len(df)):
        img_i = get_image(df.iloc[i])
        hr_i, lr_small_i, lr_i = preprocess_image(img_i)

        all_hr.append(to_tensor(hr_i))
        all_lr_small.append(to_tensor(lr_small_i))
        all_lr.append(to_tensor(lr_i))

    all_hr = torch.stack(all_hr)
    all_lr_small = torch.stack(all_lr_small)
    all_lr = torch.stack(all_lr)

    return all_hr, all_lr_small, all_lr


def get_split_indices(n_train=NUM_TRAIN_IMAGES, n_val=NUM_VAL_IMAGES, n_test=NUM_TEST_IMAGES):
    import numpy as np

    indices = np.arange(n_train + n_val + n_test)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

    return train_idx, val_idx, test_idx


def create_dataloaders(all_hr, all_lr, train_idx, val_idx, test_idx, batch_size=BATCH_SIZE):
    train_dataset = TensorDataset(all_lr[train_idx], all_hr[train_idx])
    val_dataset = TensorDataset(all_lr[val_idx], all_hr[val_idx])
    test_dataset = TensorDataset(all_lr[test_idx], all_hr[test_idx])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def load_100_image_dataset():
    download_data()
    df = pd.read_parquet(DATA_PATH)
    all_hr, all_lr_small, all_lr = load_and_preprocess_all(df)
    train_idx, val_idx, test_idx = get_split_indices()
    return df, all_hr, all_lr_small, all_lr, train_idx, val_idx, test_idx


def load_full_div2k_dataset():
    download_div2k()

    train_images = sorted([f for f in os.listdir(DIV2K_TRAIN_DIR) if f.endswith('.png')])
    valid_images = sorted([f for f in os.listdir(DIV2K_VALID_DIR) if f.endswith('.png')])

    train_images = train_images[:NUM_TRAIN_IMAGES_FULL]
    valid_images = valid_images[:NUM_VAL_IMAGES_FULL]

    all_hr_train = []
    all_lr_train = []
    all_hr_val = []
    all_lr_val = []

    for img_name in train_images:
        img = Image.open(os.path.join(DIV2K_TRAIN_DIR, img_name)).convert("RGB")
        hr, lr_small, lr = preprocess_image(img)
        all_hr_train.append(to_tensor(hr))
        all_lr_train.append(to_tensor(lr))

    for img_name in valid_images:
        img = Image.open(os.path.join(DIV2K_VALID_DIR, img_name)).convert("RGB")
        hr, lr_small, lr = preprocess_image(img)
        all_hr_val.append(to_tensor(hr))
        all_lr_val.append(to_tensor(lr))

    all_hr_train = torch.stack(all_hr_train)
    all_lr_train = torch.stack(all_lr_train)
    all_hr_val = torch.stack(all_hr_val)
    all_lr_val = torch.stack(all_lr_val)

    return all_hr_train, all_lr_train, all_hr_val, all_lr_val
