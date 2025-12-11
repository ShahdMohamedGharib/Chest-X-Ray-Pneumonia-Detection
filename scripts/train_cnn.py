"""
Train ImprovedCNN only
Script to train just the ImprovedCNN model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import *
from src.dataset import prepare_data_splits, create_data_loaders
from src.transforms import train_transforms_strong, val_test_transforms_advanced
from src.models import create_model
from src.training import train_advanced_model
from src.evaluation import evaluate_model


def main():
    print("="*70)
    print(" CHEST X-RAY CLASSIFICATION - TRAINING ImprovedCNN ONLY")
    print("="*70)
    
    # Check GPU availability
    print(f"\n🔍 Device Check:")
    print(f"   - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   - Using Device: {DEVICE}")
    else:
        print(f"   - ⚠️  WARNING: CUDA not available! Training will use CPU (very slow)")
        print(f"   - Using Device: {DEVICE}")
    
    # Prepare data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_data_splits(DATA_DIR)
    
    # ==================== TRAIN IMPROVED CNN ====================
    print("\n" + "="*70)
    print(" TRAINING: ImprovedCNN (Enhanced Model)")
    print("="*70)
    
    # Use strong augmentation for ImprovedCNN
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        train_transforms_strong, val_test_transforms_advanced
    )
    
    model = create_model('improved_cnn').to(DEVICE)
    
    # Verify model is on correct device
    print(f"\n✅ Model created and moved to: {next(model.parameters()).device}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    # Use lower learning rate for better training stability
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Enable mixed precision only if GPU is available
    use_amp = torch.cuda.is_available()
    if use_amp:
        print(f"✅ Mixed Precision Training (AMP) enabled for GPU acceleration")
    else:
        print(f"⚠️  Mixed Precision Training disabled (CPU mode)")
    
    # Use advanced training function with mixed precision
    trained_model, history = train_advanced_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=30, patience=8, device=DEVICE, model_name='improved_cnn', use_amp=use_amp
    )
    
    print("\n" + "="*70)
    print(" EVALUATING ImprovedCNN ON TEST SET")
    print("="*70)
    results = evaluate_model(trained_model, test_loader, DEVICE)
    
    print("\n" + "="*70)
    print(" ✓ ImprovedCNN TRAINING COMPLETED!")
    print("="*70)
    print(f"\nModel saved in: {get_model_path('improved_cnn')}")
    print(f"Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Test F1-Score: {results['f1']:.4f}")


if __name__ == "__main__":
    main()

