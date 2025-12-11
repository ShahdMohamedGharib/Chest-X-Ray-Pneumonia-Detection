
# Deep Learning Project: Chest X-Ray Pneumonia Detection

This project is a complete reconstruction of a Deep Learning pipeline for classifying Chest X-Ray images as NORMAL or PNEUMONIA. It includes data preparation, model training (multiple architectures), ensemble techniques, evaluation with advanced metrics, transparency via Grad-CAM, and a full-stack deployment (FastAPI + Frontend).

## Project Structure

```
├── api/                # FastAPI backend
│   └── main.py
├── frontend/           # HTML/CSS/JS Interface
├── models_saved/       # Trained model artifacts
├── scripts/            # Helper scripts
├── src/
│   ├── data/           # Dataset, transforms, splitting logic
│   ├── models/         # Model definitions (CNN, ResNet, DenseNet, EfficientNet, Ensemble)
│   ├── train.py        # Training loop with AMP
│   ├── evaluate.py     # Evaluation metrics
│   ├── gradcam.py      # Grad-CAM implementation
│   └── utils.py        # Utilities
├── Dockerfile
└── requirements.txt
```

## Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Preparation**:
    - Download the dataset (e.g., from Kaggle).
    - ensure the structure is `data/chest_xray/train` and `data/chest_xray/val` (or use `src/data/split_and_balance.py` to organize it).
    - If you have a single folder, run:
      ```bash
      python src/data/split_and_balance.py --data_dir ./data/chest_xray_raw --output_dir ./data/splits
      ```

## Training

Train individual models. You can specify the model architecture:

```bash
python src/train.py --data_dir ./data/chest_xray --model resnet18 --use_amp --epochs 25
```

Supported models: `basic_cnn`, `resnet18`, `resnet50`, `densenet121`, `efficientnet_b0`.

## Evaluation & Inference

To evaluate a model:
```python
# Import evaluate_model from src.evaluate
# See src/evaluate.py for details or run a script calling it.
```

## Running the Application

1.  **Start the API**:
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    ```
    The API loads optimal models from `models_saved/` on startup.

2.  **Start the Frontend**:
    Open `frontend/index.html` in your browser. (Or serve it using `python -m http.server` inside `frontend/` directory).

## Features

- **Preprocessing**: CLAHE contrast enhancement, resizing to 224x224, normalization.
- **Models**: State-of-the-art CNNs + Custom Ensemble.
- **Explainability**: Grad-CAM visualization integrated into the UI.
- **Performance**: Mixed Precision Training (AMP), WeightedRandomSampler for class imbalance.

## Docker

Build and run the container:
```bash
docker build -t xray-api .
docker run -p 8000:8000 xray-api
```
