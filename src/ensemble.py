import torch
import numpy as np

from src.config import *

# ==========================================================
#  GET PREDICTIONS FROM ALL BASE MODELS
# ==========================================================
def get_base_model_predictions(models, loader, device):
    preds_all = []
    labels_all = None

    print("\nGenerating predictions from base models...")

    for idx, model in enumerate(models):
        print(f"Model {idx+1}/{len(models)} → {model.__class__.__name__}")
        model.eval()
        model_preds = []
        first_labels = []

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                out = model(imgs)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                model_preds.append(probs)

                if labels_all is None:
                    first_labels.extend(lbls.numpy())

        preds_all.append(np.vstack(model_preds))

        if labels_all is None:
            labels_all = np.array(first_labels)

    # Simple averaging ensemble
    ensemble_preds = np.mean(np.stack(preds_all, axis=1), axis=1)
    return ensemble_preds, labels_all

# ==========================================================
#  PREDICTION FUNCTION USING SIMPLE ENSEMBLE
# ==========================================================
def simple_ensemble_predict(models, loader, device):
    preds, labels = get_base_model_predictions(models, loader, device)
    final_pred = np.argmax(preds, axis=1)
    accuracy = (final_pred == labels).mean() * 100
    print(f"Simple Ensemble Accuracy: {accuracy:.2f}%")
    return final_pred, labels

