"""
Train all models (ImprovedCNN + Transfer Learning models)
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
from src.transforms import train_transform_basic, val_test_transforms_basic, train_transforms_strong, val_test_transforms_advanced
from src.models import create_model
from src.training import train_model, train_advanced_model
from src.evaluation import evaluate_model


def main():
    print("="*70)
    print(" CHEST X-RAY CLASSIFICATION - TRAINING ALL MODELS")
    print("="*70)
    
    # Prepare data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_data_splits(DATA_DIR)
    
    # ==================== TRAIN IMPROVED CNN ====================
    print("\n" + "="*70)
    print(" TRAINING: ImprovedCNN (Enhanced Model)")
    print("="*70)
    
    # Use strong augmentation for ImprovedCNN too (like advanced models)
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        train_transforms_strong, val_test_transforms_advanced
    )
    
    model = create_model('improved_cnn').to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Use lower learning rate for better training stability
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Use advanced training function with mixed precision
    trained_model, history = train_advanced_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=30, patience=8, device=DEVICE, model_name='improved_cnn', use_amp=True
    )
    
    print("\nEvaluating ImprovedCNN on test set...")
    results = evaluate_model(trained_model, test_loader, DEVICE)
    
    # ==================== TRAIN ADVANCED MODELS ====================
    # Reload data with advanced transforms
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        train_transforms_strong, val_test_transforms_advanced
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    advanced_models = ['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0']
    
    for model_name in advanced_models:
        print("\n" + "="*70)
        print(f" TRAINING: {MODEL_NAMES.get(model_name, model_name)}")
        print("="*70)
        
        try:
            model = create_model(model_name, pretrained=True).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=ADVANCED_LR, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            
            trained_model, history = train_advanced_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=ADVANCED_EPOCHS, patience=ADVANCED_PATIENCE, device=DEVICE,
                model_name=model_name, use_amp=True
            )
            
            print(f"\nEvaluating {model_name} on test set...")
            results = evaluate_model(trained_model, test_loader, DEVICE)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    print("\n" + "="*70)
    print(" ✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModels saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
