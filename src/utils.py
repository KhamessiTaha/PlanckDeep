import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = float('inf')
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"Restoring model weights from epoch with best validation loss: {self.best_loss:.6f}")
                return True
        return False

class LearningRateScheduler:
    """Custom learning rate scheduler with multiple strategies"""
    def __init__(self, optimizer, strategy='cosine', **kwargs):
        self.optimizer = optimizer
        self.strategy = strategy
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        if strategy == 'cosine':
            self.T_max = kwargs.get('T_max', 50)
            self.eta_min = kwargs.get('eta_min', 1e-6)
        elif strategy == 'reduce_on_plateau':
            self.factor = kwargs.get('factor', 0.5)
            self.patience = kwargs.get('patience', 10)
            self.threshold = kwargs.get('threshold', 1e-4)
            self.best_loss = float('inf')
            self.wait = 0
        elif strategy == 'step':
            self.step_size = kwargs.get('step_size', 30)
            self.gamma = kwargs.get('gamma', 0.1)
    
    def step(self, epoch, val_loss=None):
        if self.strategy == 'cosine':
            lr = self.eta_min + (self.initial_lr - self.eta_min) * (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        elif self.strategy == 'reduce_on_plateau':
            if val_loss is None:
                raise ValueError("val_loss is required for reduce_on_plateau strategy")
            if val_loss < self.best_loss - self.threshold:
                self.best_loss = val_loss
                self.wait = 0
                return  # No change in learning rate
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    lr = self.optimizer.param_groups[0]['lr'] * self.factor
                    self.wait = 0
                else:
                    return  # No change in learning rate
        elif self.strategy == 'step':
            lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        else:
            return  # No scheduling
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class MetricsTracker:
    """Track and compute various metrics during training"""
    def __init__(self, task_names=None):
        self.task_names = task_names or ['main']
        self.reset()
    
    def reset(self):
        self.metrics = {task: {'losses': [], 'accuracies': [], 'predictions': [], 'targets': []} 
                       for task in self.task_names}
    
    def update(self, loss, predictions, targets, task_name='main'):
        if task_name not in self.metrics:
            self.metrics[task_name] = {'losses': [], 'accuracies': [], 'predictions': [], 'targets': []}
        
        self.metrics[task_name]['losses'].append(loss)
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
        
        # Calculate accuracy - FIXED VERSION
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class classification - use argmax
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            # Binary classification - apply sigmoid first if needed
            if predictions.min() < 0 or predictions.max() > 1:
                # These are logits, apply sigmoid
                predictions = torch.sigmoid(predictions)
            pred_classes = (predictions > 0.5).float().squeeze()
        
        # Ensure shapes match
        if len(pred_classes.shape) > len(targets.shape):
            pred_classes = pred_classes.squeeze()
        elif len(pred_classes.shape) < len(targets.shape):
            targets = targets.squeeze()
        
        accuracy = (pred_classes == targets).float().mean().item()
        self.metrics[task_name]['accuracies'].append(accuracy)
        
        # Store for detailed metrics
        self.metrics[task_name]['predictions'].extend(predictions.numpy())
        self.metrics[task_name]['targets'].extend(targets.numpy())
    
    def get_average_metrics(self, task_name='main'):
        if task_name not in self.metrics:
            return {}
        
        task_metrics = self.metrics[task_name]
        return {
            'avg_loss': np.mean(task_metrics['losses']),
            'avg_accuracy': np.mean(task_metrics['accuracies']),
            'total_samples': len(task_metrics['targets'])
        }
    
    def get_detailed_metrics(self, task_name='main'):
        if task_name not in self.metrics:
            return {}
        
        predictions = np.array(self.metrics[task_name]['predictions'])
        targets = np.array(self.metrics[task_name]['targets'])
        
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
            pred_probs = predictions
        else:
            # Binary classification
            pred_classes = (predictions > 0.5).astype(int)
            pred_probs = predictions
        
        metrics = {
            'accuracy': (pred_classes == targets).mean(),
            'confusion_matrix': confusion_matrix(targets, pred_classes),
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(targets)) == 2:
            try:
                if len(pred_probs.shape) > 1 and pred_probs.shape[1] > 1:
                    # Use positive class probability
                    metrics['roc_auc'] = roc_auc_score(targets, pred_probs[:, 1])
                    metrics['average_precision'] = average_precision_score(targets, pred_probs[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(targets, pred_probs)
                    metrics['average_precision'] = average_precision_score(targets, pred_probs)
            except:
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        return metrics


class ModelTrainer:
    """Comprehensive model trainer with advanced features"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, save_dir='checkpoints', task_names=None, binary_classification=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.task_names = task_names or ['main']
        self.binary_classification = binary_classification
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize components
        self.early_stopping = None
        self.lr_scheduler = None
        self.train_metrics = MetricsTracker(task_names)
        self.val_metrics = MetricsTracker(task_names)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Multi-task support
        self.is_multitask = hasattr(model, 'task_heads')
    
    def setup_early_stopping(self, patience=10, min_delta=1e-4):
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    def setup_lr_scheduler(self, strategy='cosine', **kwargs):
        self.lr_scheduler = LearningRateScheduler(self.optimizer, strategy=strategy, **kwargs)
    
    def train_epoch(self):
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)
            
            if self.is_multitask:
                # Multi-task training
                targets = {task: targets[task].to(self.device) for task in targets.keys()}
                outputs = self.model(data)
                
                total_loss = 0
                for task_name in self.task_names:
                    if task_name in targets and task_name in outputs:
                        if self.binary_classification or outputs[task_name].shape[-1] == 1:  # Binary output
                            targets[task_name] = targets[task_name].float().view(-1, 1)
                        loss = self.criterion(outputs[task_name], targets[task_name])
                        total_loss += loss
                        self.train_metrics.update(loss.item(), outputs[task_name], targets[task_name], task_name)
            else:
                # Single task training
                targets = targets.to(self.device)
                outputs = self.model(data)
    
                # Handle binary case
                if self.binary_classification or outputs.shape[-1] == 1:  # Binary output
                    targets = targets.float().view(-1, 1)
                else:
                    targets = targets.long()
                total_loss = self.criterion(outputs, targets)
                self.train_metrics.update(total_loss.item(), outputs, targets)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        return self.train_metrics.get_average_metrics()
    
    def validate_epoch(self):
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data = data.to(self.device)
                
                if self.is_multitask:
                    targets = {task: targets[task].to(self.device) for task in targets.keys()}
                    outputs = self.model(data)
                    
                    total_loss = 0
                    for task_name in self.task_names:
                        if task_name in targets and task_name in outputs:
                            if self.binary_classification or outputs[task_name].shape[-1] == 1:
                                targets[task_name] = targets[task_name].float().view(-1, 1)
                            loss = self.criterion(outputs[task_name], targets[task_name])
                            total_loss += loss
                            self.val_metrics.update(loss.item(), outputs[task_name], targets[task_name], task_name)
                else:
                    targets = targets.to(self.device)
                    outputs = self.model(data)
                    
                    # Handle binary case
                    if self.binary_classification or outputs.shape[-1] == 1:
                        targets = targets.float().view(-1, 1)
                    else:
                        targets = targets.long()
                    total_loss = self.criterion(outputs, targets)
                    self.val_metrics.update(total_loss.item(), outputs, targets)
        
        return self.val_metrics.get_average_metrics()
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'class_name': self.model.__class__.__name__,
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def train(self, epochs, save_every=10, verbose=True):
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_metrics.get('avg_loss', 0))
            self.history['val_loss'].append(val_metrics.get('avg_loss', 0))
            self.history['train_acc'].append(train_metrics.get('avg_accuracy', 0))
            self.history['val_acc'].append(val_metrics.get('avg_accuracy', 0))
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch, val_metrics.get('avg_loss'))
            
            # Check for best model
            current_val_loss = val_metrics.get('avg_loss', float('inf'))
            is_best = current_val_loss < best_val_loss
            if is_best:
                best_val_loss = current_val_loss
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Verbose output
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - "
                      f"Train Loss: {train_metrics.get('avg_loss', 0):.6f}, "
                      f"Train Acc: {train_metrics.get('avg_accuracy', 0):.4f}, "
                      f"Val Loss: {val_metrics.get('avg_loss', 0):.6f}, "
                      f"Val Acc: {val_metrics.get('avg_accuracy', 0):.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(current_val_loss, self.model):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        return self.history
# Visualization and evaluation utilities
def plot_training_history(history, save_path=None):
    """Plot training history with multiple metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Training Accuracy', color='blue', alpha=0.7)
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red', alpha=0.7)
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(history['learning_rates'], color='green', alpha=0.7)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference plot
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(loss_diff, color='purple', alpha=0.7)
    axes[1, 1].set_title('Overfitting Monitor (Val Loss - Train Loss)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def evaluate_model(model, test_loader, device, task_names=None, save_path=None):
    """Comprehensive model evaluation with visualizations"""
    model.eval()
    task_names = task_names or ['main']
    is_multitask = hasattr(model, 'task_heads')
    
    all_predictions = {task: [] for task in task_names}
    all_targets = {task: [] for task in task_names}
    all_probabilities = {task: [] for task in task_names}
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            
            if is_multitask:
                outputs = model(data)
                for task_name in task_names:
                    if task_name in outputs and task_name in targets:
                        probs = F.softmax(outputs[task_name], dim=1)
                        preds = torch.argmax(probs, dim=1)
                        
                        all_predictions[task_name].extend(preds.cpu().numpy())
                        all_targets[task_name].extend(targets[task_name].numpy())
                        all_probabilities[task_name].extend(probs.cpu().numpy())
            else:
                targets = targets.to(device)
                outputs = model(data)
                
                if outputs.shape[1] > 1:  # Multi-class
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                else:  # Binary
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                
                all_predictions['main'].extend(preds.cpu().numpy())
                all_targets['main'].extend(targets.cpu().numpy())
                all_probabilities['main'].extend(probs.cpu().numpy())
    
    # Generate evaluation report
    evaluation_results = {}
    
    for task_name in task_names:
        if len(all_targets[task_name]) == 0:
            continue
            
        predictions = np.array(all_predictions[task_name])
        targets = np.array(all_targets[task_name])
        probabilities = np.array(all_probabilities[task_name])
        
        # Basic metrics
        accuracy = (predictions == targets).mean()
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        report = classification_report(targets, predictions, output_dict=True)
        
        evaluation_results[task_name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities
        }
        
        # ROC curve for binary classification
        if len(np.unique(targets)) == 2:
            if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                pos_probs = probabilities[:, 1]
            else:
                pos_probs = probabilities.flatten()
            
            fpr, tpr, _ = roc_curve(targets, pos_probs)
            roc_auc = roc_auc_score(targets, pos_probs)
            
            evaluation_results[task_name]['roc_curve'] = (fpr, tpr)
            evaluation_results[task_name]['roc_auc'] = roc_auc
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(targets, pos_probs)
            avg_precision = average_precision_score(targets, pos_probs)
            
            evaluation_results[task_name]['pr_curve'] = (precision, recall)
            evaluation_results[task_name]['avg_precision'] = avg_precision
    
    # Plot results
    plot_evaluation_results(evaluation_results, save_path)
    
    return evaluation_results

def plot_evaluation_results(evaluation_results, save_path=None):
    """Plot comprehensive evaluation results"""
    n_tasks = len(evaluation_results)
    fig, axes = plt.subplots(2, n_tasks, figsize=(5 * n_tasks, 10))
    
    if n_tasks == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (task_name, results) in enumerate(evaluation_results.items()):
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, idx], cmap='Blues')
        axes[0, idx].set_title(f'{task_name.capitalize()} - Confusion Matrix\nAccuracy: {results["accuracy"]:.4f}')
        axes[0, idx].set_xlabel('Predicted')
        axes[0, idx].set_ylabel('Actual')
        
        # ROC Curve (if available)
        if 'roc_curve' in results:
            fpr, tpr = results['roc_curve']
            roc_auc = results['roc_auc']
            axes[1, idx].plot(fpr, tpr, color='darkorange', lw=2, 
                            label=f'ROC curve (AUC = {roc_auc:.4f})')
            axes[1, idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, idx].set_xlim([0.0, 1.0])
            axes[1, idx].set_ylim([0.0, 1.05])
            axes[1, idx].set_xlabel('False Positive Rate')
            axes[1, idx].set_ylabel('True Positive Rate')
            axes[1, idx].set_title(f'{task_name.capitalize()} - ROC Curve')
            axes[1, idx].legend(loc="lower right")
            axes[1, idx].grid(True, alpha=0.3)
        else:
            # Plot class distribution if no ROC curve
            unique, counts = np.unique(results['targets'], return_counts=True)
            axes[1, idx].bar(unique, counts)
            axes[1, idx].set_title(f'{task_name.capitalize()} - Class Distribution')
            axes[1, idx].set_xlabel('Class')
            axes[1, idx].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Utility functions
def save_training_config(config, save_path):
    """Save training configuration to JSON file"""
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('history', {})

def setup_reproducibility(seed=42):
    """Setup reproducible training"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_data_loaders(train_dataset, val_dataset, test_dataset=None, 
                       batch_size=32, num_workers=4, pin_memory=True):
    """Create data loaders with proper settings"""
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader

def calculate_class_weights(labels, device, num_classes=2):
    """Calculate class weights for imbalanced datasets"""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    
    # Convert labels to numpy array if they're not already
    if isinstance(labels, list):
        labels = np.array(labels)
    elif isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Get unique classes present in the data
    classes_in_data = np.unique(labels)
    
    # Create array for all expected classes
    all_classes = np.arange(num_classes)
    
    # Calculate weights only for classes present in data
    weights_dict = {}
    
    if len(classes_in_data) == 1:
        # Only one class present, assign equal weights
        weights = np.ones(num_classes)
    else:
        # Calculate balanced weights for present classes
        present_weights = compute_class_weight(
            'balanced', 
            classes=classes_in_data, 
            y=labels
        )
        
        # Map weights to all classes
        weights = np.ones(num_classes)  # Default weight of 1.0
        
        for i, class_idx in enumerate(classes_in_data):
            if class_idx < num_classes:
                weights[class_idx] = present_weights[i]
    
    # Convert to tensor
    return torch.tensor(weights, dtype=torch.float32).to(device)

def print_model_info(model, input_size=(1, 3, 224, 224)):
    """Print comprehensive model information"""
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_all = sum(p.numel() for p in model.parameters())
    
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Total Parameters: {total_params_all:,}")
    print(f"Input Shape: {input_size}")
    
    # Try to get output shape
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_size[1:])
            dummy_output = model(dummy_input)
            if isinstance(dummy_output, dict):
                print("Output Shapes:")
                for key, value in dummy_output.items():
                    print(f"  {key}: {tuple(value.shape)}")
            else:
                print(f"Output Shape: {tuple(dummy_output.shape)}")
    except Exception as e:
        print(f"Could not determine output shape: {e}")
    
    # Memory estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    print(f"Model Size: {total_size / 1024**2:.2f} MB")
    print("=" * 60)
    