"""
Module d'entraînement et d'évaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple, Optional
import numpy as np


def is_small_dataset(num_samples: int, threshold: int = 10000) -> bool:
    """
    Détermine si un dataset est petit (pas besoin de DataLoader).
    """
    return num_samples < threshold


def get_data_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, 
                    is_small: bool, num_workers: int = 2) -> Optional[DataLoader]:
    """
    Crée un DataLoader si nécessaire, sinon retourne None.
    """
    if is_small:
        return None
    
    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def train_epoch(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: torch.device, train_loader: Optional[DataLoader],
                is_small: bool) -> float:
    """
    Entraîne le modèle pour une époque.
        train_loader: DataLoader (None si dataset petit)
        is_small: Si True, utilise X_train/y_train directement
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if is_small:
        # pas de batch necessaire
        X_train_device = X_train.to(device)
        y_train_device = y_train.to(device)
        
        optimizer.zero_grad()
        logits = model(X_train_device)
        loss = criterion(logits, y_train_device)
        loss.backward()
        optimizer.step()
        
        total_loss = loss.item()
        num_batches = 1
    else:
        if train_loader is not None:
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / num_batches


def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Évalue le modèle sur l'ensemble de test.
        (accuracy, loss)
    """
    model.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        logits = model(X_test)
        loss = criterion(logits, y_test).item()
        preds = logits.argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    
    return accuracy, loss


def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                X_test: torch.Tensor, y_test: torch.Tensor,
                epochs: int, batch_size: int, learning_rate: float = 1e-3,
                device: Optional[torch.device] = None, verbose: bool = False) -> Dict:
    """
    Entraîne le modèle et retourne les métriques.
        Dict avec historique d'entraînement et métriques finales
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure tensors
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.LongTensor(y_train)
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.LongTensor(y_test)
    
    model.to(device)
    
    is_small = is_small_dataset(X_train.shape[0])

    train_loader = get_data_loader(X_train, y_train, batch_size, is_small)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_accuracies = []
    test_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, X_train, y_train, criterion, optimizer,
            device, train_loader, is_small
        )
        train_losses.append(train_loss)
        
        test_acc, test_loss = evaluate(model, X_test, y_test, criterion, device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Acc: {test_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}")
    
    expert_counts = None
    if hasattr(model, 'last_counts') and model.last_counts is not None:
        expert_counts = model.last_counts.cpu().numpy() if isinstance(model.last_counts, torch.Tensor) else model.last_counts
    
    final_acc, final_loss = test_accuracies[-1], test_losses[-1]
    
    if verbose:
        print(f"\n✓ Entraînement terminé!")
        print(f"  Accuracy finale: {final_acc:.4f}")
        print(f"  Loss finale: {final_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'test_losses': test_losses,
        'final_accuracy': final_acc,
        'final_loss': final_loss,
        'expert_counts': expert_counts,
        'epochs': epochs,
        'model': model
    }
