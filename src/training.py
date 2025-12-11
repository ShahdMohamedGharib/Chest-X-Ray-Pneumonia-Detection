"""
Training functions for all models
Extracted directly from notebook
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score
from src.config import *


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    From notebook
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate for one epoch
    From notebook
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return epoch_loss, epoch_acc, precision, recall, f1


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs=25, patience=7, device='cuda', model_name='model'):
    """
    Basic training function
    From notebook - line 1054
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting training for {model_name}...")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        print(f"  Training:   Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Validation: Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  Metrics:    Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, get_model_path(model_name))
            patience_counter = 0
            print(f"  ✓ New best model saved (Accuracy: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

    return model, history


def train_advanced_model(model, train_loader, val_loader, criterion, optimizer,
                         scheduler, num_epochs=30, patience=10, device='cuda',
                         model_name='model', use_amp=True):
    """
    Enhanced training function with mixed precision
    From notebook - line 1518
    """
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rate': []
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\nTraining {model_name}...")
    print("=" * 50)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Metrics: P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, get_model_path(model_name))
            patience_counter = 0
            print(f"  ✓ Saved best model (Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest validation accuracy for {model_name}: {best_val_acc:.2f}%")

    return model, history
