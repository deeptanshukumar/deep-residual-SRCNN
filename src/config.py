import os
import random

import numpy as np
import torch

SEED = 42

IMAGE_SIZE = 224
LR_SIZE = 112
CROP_SIZE = 800

BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

DEEP_LR = 3e-4
DEEP_GRAD_CLIP = 1.0

DATA_URL = (
    "https://huggingface.co/datasets/huaweilin/VTBench/"
    "resolve/main/DIV2K/test-00000-of-00001.parquet"
)
DATA_PATH = "data/div2k.parquet"

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"

DIV2K_TRAIN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DIV2K_TRAIN_DIR = "data/DIV2K_train_HR"
DIV2K_VALID_DIR = "data/DIV2K_valid_HR"

NUM_TRAIN_IMAGES = 60
NUM_VAL_IMAGES = 20
NUM_TEST_IMAGES = 20

NUM_TRAIN_IMAGES_FULL = 800
NUM_VAL_IMAGES_FULL = 100
NUM_TEST_IMAGES_FULL = 100


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
