"""
FastAPI Backend for Chest X-Ray Classification
Serves predictions from all models + ensemble + Grad-CAM
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
import cv2
import base64
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.models import create_model, EnsembleModel
from src.transforms import val_test_transforms_advanced

# Re-check CUDA after imports (sometimes config loads before CUDA is ready)
import torch
if torch.cuda.is_available() and DEVICE.type != 'cuda':
    DEVICE = torch.device("cuda")
    print(f"✅ CUDA re-detected! Switching to GPU: {torch.cuda.get_device_name(0)}")
elif not torch.cuda.is_available() and DEVICE.type == 'cuda':
    DEVICE = torch.device("cpu")
    print(f"⚠️ CUDA not available, switching to CPU")

app = FastAPI(title="Chest X-Ray Classification API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
loaded_models = {}
ensemble_model = None
model_weights = {}  # Store validation accuracies for weighted ensemble


def load_all_models():
    """Load all trained models on startup and create ensemble"""
    global loaded_models, ensemble_model, model_weights
    
    # Check GPU availability
    print("\n" + "="*70)
    print("🔍 GPU/DEVICE CHECK")
    print("="*70)
    print(f"   - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   - Current GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   - ✅ Using GPU: {DEVICE}")
    else:
        print(f"   - ⚠️  WARNING: CUDA not available! API will use CPU (very slow)")
        print(f"   - Using Device: {DEVICE}")
    print("="*70 + "\n")
    
    print("Loading models...")
    model_names = ['improved_cnn', 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0']
    
    ensemble_models_list = []
    ensemble_names = []
    weights = []
    
    # Default validation accuracies (can be updated from saved ensemble)
    default_accuracies = {
        'improved_cnn': 87.5,
        'resnet18': 90.0,
        'resnet50': 91.0,
        'densenet121': 90.0,
        'efficientnet_b0': 91.0
    }
    
    for model_name in model_names:
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            try:
                model = create_model(model_name, pretrained=False).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                
                # Verify model is on correct device
                model_device = next(model.parameters()).device
                loaded_models[model_name] = model
                ensemble_models_list.append(model)
                ensemble_names.append(model_name)
                weights.append(default_accuracies.get(model_name, 50.0))
                print(f"  ✓ Loaded {model_name} on {model_device}")
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
    
    # Create ensemble with weighted averaging
    if len(ensemble_models_list) >= 2:
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
        model_weights = dict(zip(ensemble_names, weights))
        
        ensemble_model = EnsembleModel(ensemble_models_list, weights=weights).to(DEVICE)
        ensemble_model.eval()
        ensemble_device = next(ensemble_model.parameters()).device
        print(f"  ✓ Created ensemble with {len(ensemble_models_list)} models on {ensemble_device}")
        print(f"  Weights: {model_weights}")
        
        # Show GPU memory usage after loading
        if torch.cuda.is_available():
            print(f"\n📊 GPU Memory Usage:")
            print(f"   - Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"   - Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            print(f"   - Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.2f} GB")
    else:
        print(f"  ⚠ Warning: Need at least 2 models for ensemble")
    
    print(f"\nTotal models loaded: {len(loaded_models)}")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_all_models()


def preprocess_image(image: Image.Image):
    """
    Preprocess image with CLAHE (same as training)
    """
    # Convert to grayscale
    img_gray = image.convert('L')
    
    # Apply CLAHE
    img_np = np.array(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_np)
    
    # Convert back to PIL
    img_preprocessed = Image.fromarray(img_clahe)
    
    # Apply transforms
    img_tensor = val_test_transforms_advanced(img_preprocessed)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    return img_tensor, img_clahe


def get_target_layer_name(model_name, model):
    """Get the appropriate target layer name for Grad-CAM based on model architecture"""
    # Try to find the last convolutional layer for each model type
    target_layer = None
    target_name = None
    
    if 'improved_cnn' in model_name:
        # For ImprovedCNN, find the last conv layer in features
        for name, module in model.named_modules():
            if 'features' in name and isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_name = name
    elif 'resnet' in model_name:
        # For ResNet, use layer4 (last residual block)
        for name, module in model.named_modules():
            if 'layer4' in name and isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_name = name
        # If layer4 not found, try to find any conv in base_model
        if target_layer is None:
            for name, module in model.named_modules():
                if 'base_model' in name and isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    target_name = name
    elif 'densenet' in model_name:
        # For DenseNet, find last dense block
        for name, module in model.named_modules():
            if 'denseblock4' in name and isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_name = name
        # Fallback to any conv in features
        if target_layer is None:
            for name, module in model.named_modules():
                if 'features' in name and isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    target_name = name
    elif 'efficientnet' in model_name:
        # For EfficientNet, find last block
        for name, module in model.named_modules():
            if 'blocks' in name and isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_name = name
    
    # If still not found, find the last conv layer overall
    if target_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_name = name
    
    return target_name


def generate_gradcam(model, img_tensor, model_name='model'):
    """
    Generate Grad-CAM heatmap for any model
    """
    model.eval()
    
    # Get target layer name
    target_layer_name = get_target_layer_name(model_name, model)
    if target_layer_name is None:
        return None
    
    # Forward pass
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0])
    
    # Find target layer
    target_layer = None
    if target_layer_name:
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break
    
    if target_layer is None:
        # Try alternative: find last conv layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_layer_name = name
                break
    
    if target_layer is None:
        return None
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        
        # Backward
        model.zero_grad()
        class_score = output[0, pred_class]
        class_score.backward()
        
        # Generate CAM
        if len(activations) > 0 and len(gradients) > 0:
            activation = activations[0].detach().cpu()
            gradient = gradients[0].detach().cpu()
            
            # Handle different activation shapes
            if len(activation.shape) == 4:
                # Global average pooling of gradients
                weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
                
                # Weighted combination
                cam = torch.sum(weights * activation, dim=1).squeeze()
            else:
                return None
            
            cam = F.relu(cam)
            cam = cam.numpy()
            
            # Normalize
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)
            
            # Resize to original image size
            if len(cam.shape) == 2:
                cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
            else:
                return None
            
            return cam
    except Exception as e:
        print(f"Grad-CAM error for {model_name}: {e}")
        return None
    finally:
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
    
    return None


def array_to_base64(img_array):
    """Convert numpy array to base64 string"""
    # Normalize to 0-255
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"


@app.get("/")
async def root():
    return {"message": "Chest X-Ray Classification API", "models_loaded": len(loaded_models)}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "ensemble_loaded": ensemble_model is not None,
        "device": str(DEVICE)
    }


@app.get("/model-info")
async def model_info():
    """Get information about loaded models"""
    return {
        "models": list(loaded_models.keys()),
        "ensemble_loaded": ensemble_model is not None,
        "ensemble_weights": model_weights,
        "classes": CLASSES,
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia from chest X-ray
    Returns predictions from all models + ensemble + Grad-CAM
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Preprocess
        img_tensor, img_clahe = preprocess_image(image)
        
        # Get predictions from all models
        predictions = {}
        all_probs = []
        gradcam_images = {}
        
        for model_name, model in loaded_models.items():
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = int(probs.argmax())
                confidence = float(probs[pred_class])
                
                predictions[model_name] = {
                    "class": CLASSES[pred_class],
                    "confidence": confidence,
                    "probabilities": {
                        CLASSES[0]: float(probs[0]),
                        CLASSES[1]: float(probs[1])
                    }
                }
                
                all_probs.append(probs)
            
            # Generate Grad-CAM for each model
            try:
                cam = generate_gradcam(model, img_tensor, model_name)
                if cam is not None:
                    # Apply colormap
                    cam_colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
                    
                    # Overlay on original
                    original_resized = cv2.resize(np.array(image.convert('L')), (IMG_SIZE, IMG_SIZE))
                    original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2RGB)
                    
                    overlay = cv2.addWeighted(original_resized, 0.6, cam_colored, 0.4, 0)
                    
                    gradcam_images[model_name] = array_to_base64(overlay)
                else:
                    gradcam_images[model_name] = None
            except Exception as e:
                print(f"Grad-CAM error for {model_name}: {e}")
                gradcam_images[model_name] = None
        
        # Ensemble prediction using weighted average (like original code)
        ensemble_prediction = None
        if ensemble_model is not None and len(all_probs) > 0:
            with torch.no_grad():
                ensemble_output = ensemble_model(img_tensor)
                ensemble_probs = torch.softmax(ensemble_output, dim=1).cpu().numpy()[0]
                ensemble_class = int(ensemble_probs.argmax())
                ensemble_confidence = float(ensemble_probs[ensemble_class])
            
            ensemble_prediction = {
                "class": CLASSES[ensemble_class],
                "confidence": ensemble_confidence,
                "probabilities": {
                    CLASSES[0]: float(ensemble_probs[0]),
                    CLASSES[1]: float(ensemble_probs[1])
                }
            }
        
        # Preprocessed image (CLAHE)
        preprocessed_image = array_to_base64(img_clahe)
        
        return JSONResponse(content={
            "status": "success",
            "individual_predictions": predictions,
            "ensemble_prediction": ensemble_prediction,
            "visualizations": {
                "preprocessed": preprocessed_image,
                "gradcam": gradcam_images  # Now returns dict of all model Grad-CAMs
            }
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset-info")
async def dataset_info():
    """Get information about the dataset"""
    return {
        "name": "Chest X-Ray Images (Pneumonia)",
        "source": "Kaggle",
        "classes": CLASSES,
        "description": "Dataset of chest X-ray images for pneumonia detection",
        "preprocessing": [
            "CLAHE (Contrast Limited Adaptive Histogram Equalization)",
            "Resize to 224x224",
            "Normalization",
            "Data Augmentation (rotation, flip, jitter)"
        ],
        "split": {
            "method": "train_test_split on train folder",
            "train": "80%",
            "validation": "20%",
            "test": "separate test set"
        }
    }


@app.get("/models-info")
async def models_info():
    """Get detailed information about all models"""
    return {
        "models": [
            {
                "name": "ImprovedCNN",
                "key": "improved_cnn",
                "type": "Custom CNN (Enhanced)",
                "description": "Enhanced custom convolutional neural network with 4 conv blocks and deeper architecture",
                "parameters": "~8-10M",
                "architecture": "4 Conv blocks (64→128→256→512 filters) + Enhanced classifier (1024→512→256)",
                "validation_accuracy": "~85-90%",
                "weight": model_weights.get('improved_cnn', 0.0)
            },
            {
                "name": "ResNet18",
                "key": "resnet18",
                "type": "Transfer Learning",
                "description": "ResNet-18 pretrained on ImageNet, fine-tuned for X-rays",
                "parameters": "~11M",
                "architecture": "18-layer residual network",
                "validation_accuracy": "~90%",
                "weight": model_weights.get('resnet18', 0.0)
            },
            {
                "name": "ResNet50",
                "key": "resnet50",
                "type": "Transfer Learning",
                "description": "ResNet-50 pretrained on ImageNet, fine-tuned for X-rays",
                "parameters": "~23M",
                "architecture": "50-layer residual network",
                "validation_accuracy": "~91%",
                "weight": model_weights.get('resnet50', 0.0)
            },
            {
                "name": "DenseNet121",
                "key": "densenet121",
                "type": "Transfer Learning",
                "description": "DenseNet-121 with dense connections for better gradient flow",
                "parameters": "~7M",
                "architecture": "121-layer densely connected network",
                "validation_accuracy": "~90%",
                "weight": model_weights.get('densenet121', 0.0)
            },
            {
                "name": "EfficientNet-B0",
                "key": "efficientnet_b0",
                "type": "Transfer Learning",
                "description": "EfficientNet-B0 with compound scaling",
                "parameters": "~4M",
                "architecture": "Efficient compound scaling",
                "validation_accuracy": "~91%",
                "weight": model_weights.get('efficientnet_b0', 0.0)
            },
            {
                "name": "Weighted Ensemble",
                "key": "ensemble",
                "type": "Weighted Average Ensemble",
                "description": "Weighted average of all 5 models based on validation accuracy",
                "parameters": "0 (no additional parameters)",
                "architecture": "Weighted averaging of logits from all models",
                "validation_accuracy": "~88-92%",
                "weights": model_weights
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
