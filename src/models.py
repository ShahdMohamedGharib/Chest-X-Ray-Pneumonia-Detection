"""
All model architectures
Extracted directly from notebook
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from src.config import *


# ==================== IMPROVED CNN ====================
class ImprovedCNN(nn.Module):
    """
    Enhanced Custom CNN architecture with deeper layers and residual connections
    Improved for better accuracy
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.4):
        super(ImprovedCNN, self).__init__()

        # First block - more filters
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        # Second block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
        )

        # Third block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        # Fourth block - additional depth
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # Global average pooling alternative
            nn.Dropout2d(0.25),
        )

        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x


# ==================== RESNET ====================
class ResNetModel(nn.Module):
    """
    ResNet-based model for X-ray classification
    Extracted directly from notebook - line 1638
    """
    def __init__(self, num_classes=NUM_CLASSES, model_name='resnet18', pretrained=True):
        super(ResNetModel, self).__init__()

        # Load pretrained ResNet
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.base_model = models.resnet18(weights=weights)
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.base_model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Modify first conv layer for grayscale images
        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        # Copy weights from original conv1 (average across RGB channels)
        if pretrained and weights is not None:
            with torch.no_grad():
                self.base_model.conv1.weight = nn.Parameter(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )

        # Modify final layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# ==================== DENSENET ====================
class DenseNetModel(nn.Module):
    """
    DenseNet-based model for X-ray classification
    Extracted directly from notebook - line 1387
    """
    def __init__(self, num_classes=NUM_CLASSES, model_name='densenet121', pretrained=True):
        super(DenseNetModel, self).__init__()

        # Load pretrained DenseNet
        if model_name == 'densenet121':
            self.base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        elif model_name == 'densenet169':
            self.base_model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained else None)
        elif model_name == 'densenet201':
            self.base_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Modify first conv layer for grayscale images
        original_conv0 = self.base_model.features.conv0
        self.base_model.features.conv0 = nn.Conv2d(
            1, 64,
            kernel_size=original_conv0.kernel_size,
            stride=original_conv0.stride,
            padding=original_conv0.padding,
            bias=original_conv0.bias is not None
        )

        # Copy weights from original conv0 (average across RGB channels)
        if pretrained:
            with torch.no_grad():
                self.base_model.features.conv0.weight = nn.Parameter(
                    original_conv0.weight.mean(dim=1, keepdim=True)
                )

        # Modify final layer
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# ==================== EFFICIENTNET ====================
class EfficientNetModel(nn.Module):
    """
    EfficientNet-based model for X-ray classification
    Extracted directly from notebook - line 1434
    """
    def __init__(self, num_classes=NUM_CLASSES, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetModel, self).__init__()

        # Load EfficientNet from timm
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Get number of features
        num_features = self.base_model.num_features

        # Modify classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Handle grayscale input
        original_conv_stem = self.base_model.conv_stem
        self.base_model.conv_stem = nn.Conv2d(
            1, original_conv_stem.out_channels,
            kernel_size=original_conv_stem.kernel_size,
            stride=original_conv_stem.stride,
            padding=original_conv_stem.padding,
            bias=original_conv_stem.bias is not None
        )

        if pretrained:
            with torch.no_grad():
                self.base_model.conv_stem.weight = nn.Parameter(
                    original_conv_stem.weight.mean(dim=1, keepdim=True)
                )
                if original_conv_stem.bias is not None:
                    self.base_model.conv_stem.bias = nn.Parameter(original_conv_stem.bias)

    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)


# ==================== ENSEMBLE MODEL ====================
class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models with weighted averaging
    Extracted directly from notebook - line 1481
    """
    def __init__(self, model_list, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(model_list)

        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(model_list)] * len(model_list)
        else:
            self.weights = weights

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Weighted average of logits
        weighted_sum = None
        for output, weight in zip(outputs, self.weights):
            if weighted_sum is None:
                weighted_sum = output * weight
            else:
                weighted_sum += output * weight

        return weighted_sum

    def predict_proba(self, x):
        """Return class probabilities"""
        with torch.no_grad():
            outputs = self.forward(x)
            return torch.softmax(outputs, dim=1)


# ==================== MODEL FACTORY ====================
def create_model(model_name, pretrained=True):
    """
    Factory function to create models by name
    """
    if model_name == 'improved_cnn':
        return ImprovedCNN(num_classes=NUM_CLASSES)
    elif model_name == 'resnet18':
        return ResNetModel(num_classes=NUM_CLASSES, model_name='resnet18', pretrained=pretrained)
    elif model_name == 'resnet50':
        return ResNetModel(num_classes=NUM_CLASSES, model_name='resnet50', pretrained=pretrained)
    elif model_name == 'densenet121':
        return DenseNetModel(num_classes=NUM_CLASSES, model_name='densenet121', pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        return EfficientNetModel(num_classes=NUM_CLASSES, model_name='efficientnet_b0', pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
