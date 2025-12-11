"""
Train Simple Ensemble Only (Final Version)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.config import *
from src.dataset import prepare_data_splits, create_data_loaders
from src.transforms import val_test_transforms_advanced
from src.models import create_model
from src.ensemble import simple_ensemble_predict


def load_trained_models():
    print("\n" + "="*70)
    print("📦 LOADING TRAINED MODELS")
    print("="*70)
    
    base_models = []
    model_names_list = []
    model_names = ['improved_cnn', 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0']

    for model_name in model_names:
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            try:
                model = create_model(model_name, pretrained=False).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                base_models.append(model)
                model_names_list.append(model_name)
                print(f"  ✅ Loaded {MODEL_NAMES.get(model_name, model_name)}")
            except Exception as e:
                print(f"  ❌ Failed to load {model_name}: {e}")
        else:
            print(f"  ⚠️  Model not found: {model_name}")

    return base_models, model_names_list


def main():
    print("="*70)
    print(" 🎯 SIMPLE ENSEMBLE PREDICTION")
    print("="*70)
    
    # Prepare data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_data_splits(DATA_DIR)
    
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        val_test_transforms_advanced, val_test_transforms_advanced
    )
    
    # Load all trained models
    base_models, model_names_list = load_trained_models()
    
    if len(base_models) < 2:
        print("\n❌ Error: Need at least 2 trained models for ensemble!")
        print("Please run: python train_all.py first")
        return
    
    print(f"\n✅ Using {len(base_models)} models for ensemble")
    
    # ==================== SIMPLE ENSEMBLE ====================
    print("\n" + "="*70)
    print(" STRATEGY: SIMPLE ENSEMBLE")
    print("="*70)
    
    # Predict using Simple Ensemble
    final_preds, labels = simple_ensemble_predict(base_models, test_loader, DEVICE)

    # Calculate metrics
    accuracy = (final_preds == labels).mean()
    # Optional: calculate F1-score using sklearn
    try:
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, final_preds, average='weighted')
    except ImportError:
        f1 = None

    print("\n" + "="*70)
    print(" SIMPLE ENSEMBLE TEST SET PERFORMANCE")
    print("="*70)
    print(f"Accuracy: {accuracy*100:.2f}%")
    if f1 is not None:
        print(f"F1-Score: {f1:.4f}")
    print("="*70)

    print("\n✅ Simple Ensemble evaluated successfully!")
    # Path where ensemble could be saved (optional)
    print(f"   - Ensemble Path: {get_ensemble_path()}")


if __name__ == "__main__":
    main()