"""
Image transformations for X-ray images
Includes CLAHE preprocessing and data augmentation
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.config import *


class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalization for X-ray images
    Extracted directly from notebook
    """
    def __init__(self, clip_limit=CLAHE_CLIP_LIMIT):
        self.clip_limit = clip_limit

    def __call__(self, img):
        img_np = np.array(img)

        if len(img_np.shape) == 2:
            # Already grayscale
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_np)
        else:
            # Convert RGB to grayscale
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_gray)

        return Image.fromarray(enhanced)


# ==================== BASIC TRANSFORMS ====================
# From notebook - basic version
train_transform_basic = transforms.Compose([
    CLAHETransform(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_test_transforms_basic = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ==================== ADVANCED TRANSFORMS ====================
# From notebook - strong augmentation for advanced models
train_transforms_strong = transforms.Compose([
    CLAHETransform(clip_limit=STRONG_CLAHE_CLIP),
    transforms.Resize((240, 240)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=ROTATION_DEGREES),
    transforms.ColorJitter(brightness=BRIGHTNESS_JITTER, contrast=CONTRAST_JITTER),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

train_transforms_medium = transforms.Compose([
    CLAHETransform(clip_limit=2.0),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

val_test_transforms_advanced = transforms.Compose([
    CLAHETransform(clip_limit=2.0),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])


# ==================== TEST TIME AUGMENTATION ====================
def get_tta_transforms():
    """
    Get list of transforms for Test Time Augmentation
    From notebook
    """
    return [
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ]),
    ]
