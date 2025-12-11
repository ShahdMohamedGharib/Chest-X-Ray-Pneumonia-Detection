"""
Dataset loading and preparation
Extracted directly from notebook with train_test_split logic
"""

import os
from PIL import Image
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from src.config import *


class XRayDataset(Dataset):
    """
    Custom Dataset for X-ray images
    Extracted directly from notebook
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"[Warning] Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data_from_folder(folder_path):
    """
    Load image paths and labels from a folder
    From notebook
    """
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.exists(class_folder):
            for filename in os.listdir(class_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_folder, filename))
                    labels.append(class_idx)
        else:
            print(f"Folder not found: {class_folder}")

    return image_paths, labels


def prepare_data_splits(data_dir):
    """
    Prepare train/val/test splits
    IMPORTANT: Uses train_test_split on train folder (from notebook)
    This is the correct approach that achieves high accuracy!
    """
    print("Loading dataset...")
    
    # Paths
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Load all training data
    all_train_paths = []
    all_train_labels = []
    
    for i, cls in enumerate(CLASSES):
        folder = os.path.join(train_dir, cls)
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_train_paths.append(os.path.join(folder, fname))
                    all_train_labels.append(i)  # 0=NORMAL, 1=PNEUMONIA
    
    # Split train into train/val (80/20) with stratification
    # This is the KEY from notebook that improves accuracy!
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_paths,
        all_train_labels,
        test_size=VAL_SPLIT_RATIO,
        stratify=all_train_labels,
        random_state=SEED
    )
    
    # Load test data
    test_paths, test_labels = load_data_from_folder(test_dir)
    
    print(f"\nData statistics:")
    print(f"   - Training: {len(train_paths)} images")
    print(f"   - Validation: {len(val_paths)} images")
    print(f"   - Testing: {len(test_paths)} images")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def calculate_class_weights(labels):
    """
    Calculate class weights for handling imbalance
    From notebook
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)

    print(f"\nClass distribution:")
    for class_idx, count in class_counts.items():
        print(f"   - Class {class_idx} ({CLASSES[class_idx]}): {count} samples ({count/total_samples*100:.1f}%)")

    weights = []
    for class_idx in range(num_classes):
        weight = total_samples / (num_classes * class_counts[class_idx])
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def create_data_loaders(train_paths, train_labels, val_paths, val_labels, 
                       test_paths, test_labels, train_transform, val_transform):
    """
    Create DataLoaders with WeightedRandomSampler for training
    From notebook
    """
    # Calculate class weights
    class_weights = calculate_class_weights(train_labels).to(DEVICE)
    print(f"\nClass Weights: Normal={class_weights[0]:.4f}, Pneumonia={class_weights[1]:.4f}")
    
    # Create datasets
    train_dataset = XRayDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = XRayDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = XRayDataset(test_paths, test_labels, transform=val_transform)
    
    # Create weighted sampler for training
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nDataLoaders created:")
    print(f"   - Training: {len(train_loader.dataset)} samples")
    print(f"   - Validation: {len(val_loader.dataset)} samples")
    print(f"   - Testing: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader, class_weights
