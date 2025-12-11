"""
Evaluate all models and compare performance
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.config import *
from src.dataset import prepare_data_splits, create_data_loaders
from src.transforms import val_test_transforms_advanced
from src.models import create_model, EnsembleModel
from src.evaluation import evaluate_model


def main():
    print("="*70)
    print(" EVALUATING ALL MODELS")
    print("="*70)
    
    # Prepare data - we only need test data for evaluation
    _, _, _, _, test_paths, test_labels = prepare_data_splits(DATA_DIR)
    
    # Create test dataset and loader directly (no need for train/val or class weights)
    from src.transforms import val_test_transforms_advanced
    from torch.utils.data import DataLoader
    from src.dataset import XRayDataset
    
    test_dataset = XRayDataset(test_paths, test_labels, transform=val_test_transforms_advanced)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nTest dataset created: {len(test_dataset)} images")
    
    results = []
    
    # Evaluate individual models
    model_names = ['improved_cnn', 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0']
    base_models = []
    model_name_to_model = {}  # Map model names to actual models
    
    for model_name in model_names:
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            try:
                print(f"\n{'='*70}")
                print(f" Evaluating: {MODEL_NAMES.get(model_name, model_name)}")
                print(f"{'='*70}")
                
                model = create_model(model_name, pretrained=False).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                
                result = evaluate_model(model, test_loader, DEVICE)
                results.append({
                    'model': MODEL_NAMES.get(model_name, model_name),
                    'accuracy': result['accuracy'],
                    'f1': result['f1']
                })
                
                base_models.append(model)
                model_name_to_model[model_name] = model
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
    
    # Evaluate Weighted Ensemble if we have at least 2 models
    if len(base_models) >= 2:
        print(f"\n{'='*70}")
        print(" Evaluating: Weighted Ensemble")
        print(f"{'='*70}")
        
        try:
            # Default validation accuracies for weights
            default_accuracies = {
                'improved_cnn': 87.5,
                'resnet18': 90.0,
                'resnet50': 91.0,
                'densenet121': 90.0,
                'efficientnet_b0': 91.0
            }
            
            # Get weights based on loaded models
            ensemble_models_list = []
            weights = []
            
            for model_name in model_names:
                if model_name in model_name_to_model:
                    ensemble_models_list.append(model_name_to_model[model_name])
                    weights.append(default_accuracies.get(model_name, 50.0))
            
            # Normalize weights
            if weights and len(ensemble_models_list) >= 2:
                weights = [w / sum(weights) for w in weights]
                
                # Get model display names
                ensemble_display_names = [MODEL_NAMES.get(n, n) for n in model_names if n in model_name_to_model]
                
                print(f"Creating ensemble with {len(ensemble_models_list)} models")
                print(f"Model weights:")
                for name, weight in zip(ensemble_display_names, weights):
                    print(f"  - {name}: {weight:.4f}")
                
                # Create ensemble
                ensemble_model = EnsembleModel(ensemble_models_list, weights=weights).to(DEVICE)
                ensemble_model.eval()
                
                # Evaluate ensemble
                ensemble_result = evaluate_model(ensemble_model, test_loader, DEVICE)
                
                results.append({
                    'model': 'Weighted Ensemble',
                    'accuracy': ensemble_result['accuracy'],
                    'f1': ensemble_result['f1']
                })
            else:
                print("Could not create ensemble - need at least 2 models")
                
        except Exception as e:
            print(f"Error evaluating ensemble: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print(" SUMMARY - ALL MODELS")
    print("="*70)
    print(f"{'Model':<30} {'Accuracy':>15} {'F1-Score':>15}")
    print("-"*70)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['model']:<30} {result['accuracy']*100:>14.2f}% {result['f1']:>15.4f}")
    
    print("="*70)


if __name__ == "__main__":
    main()
