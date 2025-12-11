"""
Configuration file for Chest X-Ray Classification Project
All constants and hyperparameters in one place
"""

import os
import torch

# ==================== PATHS ====================
import os

# Project root (المجلد الأساسي للمشروع)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory (مجلد الداتا داخل المشروع)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chest_xray")

# Models directory (مجلد حفظ الموديلات)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_saved")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Train, Test, Val directories
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_DIR = os.path.join(DATA_DIR, "val")

# ==================== DEVICE ====================
# Initialize device - check CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==================== DATA ====================
CLASSES = ['NORMAL', 'PNEUMONIA']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced for 4GB GPU (was 32)
VAL_SPLIT_RATIO = 0.2
SEED = 42

# ==================== TRAINING ====================
# Basic CNN (Enhanced)
BASIC_EPOCHS = 30  # Increased epochs
BASIC_LR = 0.0005  # Lower learning rate for better convergence
BASIC_PATIENCE = 8  # More patience

# Advanced Models
ADVANCED_EPOCHS = 25
ADVANCED_LR = 0.0001
ADVANCED_PATIENCE = 7
WEIGHT_DECAY = 1e-4

# Meta-Learner
META_EPOCHS = 50
META_LR = 0.001
META_HIDDEN_SIZE = 128

# ==================== AUGMENTATION ====================
CLAHE_CLIP_LIMIT = 2.0
STRONG_CLAHE_CLIP = 2.5
ROTATION_DEGREES = 15
BRIGHTNESS_JITTER = 0.2
CONTRAST_JITTER = 0.2

# ==================== MODEL NAMES ====================
MODEL_NAMES = {
    'improved_cnn': 'ImprovedCNN',
    'resnet18': 'ResNet18',
    'resnet50': 'ResNet50',
    'densenet121': 'DenseNet121',
    'efficientnet_b0': 'EfficientNet-B0'
}

# ==================== PATHS FOR SAVED MODELS ====================
def get_model_path(model_name):
    """Get the path for a saved model"""
    return os.path.join(MODELS_DIR, f'best_{model_name}.pth')

def get_ensemble_path():
    """Get the path for saved ensemble"""
    return os.path.join(MODELS_DIR, 'best_ensemble_model.pth')

def get_meta_learner_path():
    """Get the path for saved meta-learner"""
    return os.path.join(MODELS_DIR, 'meta_learner.pth')
