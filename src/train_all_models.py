#!/usr/bin/env python3
"""
Complete training script for CMB deep learning models
Train multiple architectures and compare their performance
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
import os
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from dataset import EnhancedCMBDataset, MultiTaskCMBDataset
from models import create_model, get_model_summary
from utils import (
    ModelTrainer, setup_reproducibility, create_data_loaders,
    calculate_class_weights, evaluate_model, plot_training_history,
    save_training_config, print_model_info
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CMB Classification Models')
    
    # Data arguments
    parser.add_argument('--patch_file', type=str, default='data/cmb_patches.npy',
                       help='Path to CMB patches file')
    parser.add_argument('--label_file', type=str, default='data/cmb_labels.npy',
                       help='Path to labels file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--binary_classification', action='store_true',
                   help='Use binary classification mode')
    
    # Model arguments
    parser.add_argument('--models', nargs='+', 
                       default=['improved_cnn', 'resnet', 'physics_informed'],
                       choices=['improved_cnn', 'resnet', 'physics_informed', 
                               'vision_transformer', 'multitask'],
                       help='Models to train')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training configuration
    parser.add_argument('--use_early_stopping', action='store_true',
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                       choices=['cosine', 'reduce_on_plateau', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output during training')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup and return the appropriate device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return device

def create_experiment_dir(output_dir, experiment_name):
    """Create experiment directory with timestamp"""
    if experiment_name is None:
        experiment_name = f"cmb_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['checkpoints', 'results', 'plots', 'logs']:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir

def prepare_datasets(args, device):
    """Prepare train, validation, and test datasets"""
    print("Loading and preparing datasets...")
    
    # Check if files exist
    if not os.path.exists(args.patch_file):
        raise FileNotFoundError(f"Patch file not found: {args.patch_file}")
    if not os.path.exists(args.label_file):
        raise FileNotFoundError(f"Label file not found: {args.label_file}")
    
    # Load data
    patches = np.load(args.patch_file)
    labels = np.load(args.label_file)
    
    # Convert labels to appropriate type for binary classification
    if args.binary_classification:
        # Ensure labels are binary (0 or 1) and convert to float for BCEWithLogitsLoss
        labels = (labels > 0).astype(np.float32)
    else:
        # For multi-class, ensure labels are long integers
        labels = labels.astype(np.int64)
    
    print(f"Loaded {len(patches)} patches with shape {patches.shape[1:]}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Class distribution: {np.bincount(labels.astype(int))}")
    
    # Create dataset
    if 'multitask' in args.models:
        # For multitask learning, we need to create synthetic secondary tasks
        # This is a simplified example - in practice, you'd have real secondary labels
        label_dict = {
            'main': labels,
            'noise_level': np.random.randint(0, 3, len(labels)),  # Low, medium, high noise
            'field_strength': np.random.randint(0, 2, len(labels))  # Weak, strong field
        }
        full_dataset = MultiTaskCMBDataset(patches, label_dict)
    else:
        full_dataset = EnhancedCMBDataset(
            patches, labels,
            transform_strength=0.3,
            augment=True
        )
    
    # Split dataset
    total_size = len(full_dataset)
    test_size = int(args.test_split * total_size)
    val_size = int(args.val_split * total_size)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type=='cuda'
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type=='cuda'
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type=='cuda'
    )
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights and not args.binary_classification:
        train_labels = []
        for _, label in train_dataset:
            if isinstance(label, dict):
                train_labels.append(label['main'])
            else:
                train_labels.append(label)
        class_weights = calculate_class_weights(train_labels, device)
        print(f"Class weights: {class_weights}")
    elif args.use_class_weights and args.binary_classification:
        # For binary classification with BCEWithLogitsLoss, calculate pos_weight
        train_labels = []
        for _, label in train_dataset:
            if isinstance(label, dict):
                train_labels.append(label['main'])
            else:
                train_labels.append(label)
        train_labels = np.array(train_labels)
        pos_count = np.sum(train_labels)
        neg_count = len(train_labels) - pos_count
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        class_weights = torch.tensor([pos_weight], device=device)
        print(f"Positive class weight: {pos_weight:.4f}")
    
    return train_loader, val_loader, test_loader, class_weights

def create_model_and_optimizer(model_type, args, device, class_weights=None):
    """Create model, criterion, and optimizer"""
    print(f"\nCreating {model_type} model...")
    
    # Model configuration
    if args.binary_classification:
        model_config = {'num_classes': 1}
    else:
        model_config = {'num_classes': args.num_classes}
    
    # Get input shape from first batch (assuming 128x128 patches)
    input_shape = (128, 128)
    
    # Only add dropout_rate for models that support it
    if model_type in ['improved_cnn', 'physics_informed', 'vision_transformer']:
        model_config['dropout_rate'] = args.dropout_rate
    
    # Special configurations for specific models
    if model_type == 'physics_informed':
        model_config.update({
            'include_power_spectrum': True,
            'input_size': input_shape[0]  # Add input size for proper dimension calculation
        })
    elif model_type == 'vision_transformer':
        model_config.update({
            'img_size': input_shape[0],
            'patch_size': 16,
            'embed_dim': 384,  # Smaller for efficiency
            'depth': 6,
            'num_heads': 6
        })
    elif model_type == 'multitask':
        task_configs = {
            'main': {'num_classes': args.num_classes},
            'noise_level': {'num_classes': 3},
            'field_strength': {'num_classes': 2}
        }
        model_config = {
            'task_configs': task_configs,
            'shared_backbone': 'resnet',
            'dropout_rate': args.dropout_rate
        }
    
    # Create model
    model = create_model(model_type, **model_config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create criterion
    if args.binary_classification:
        if class_weights is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    return model, criterion, optimizer

def train_single_model(model_type, args, train_loader, val_loader, device, 
                      class_weights, exp_dir):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # Create model and optimizer
    model, criterion, optimizer = create_model_and_optimizer(
        model_type, args, device, class_weights
    )
    
    # Setup trainer
    model_save_dir = os.path.join(exp_dir, 'checkpoints', model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    
    task_names = ['main', 'noise_level', 'field_strength'] if model_type == 'multitask' else None
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=model_save_dir,
        task_names=task_names,
        binary_classification=args.binary_classification  # Pass binary classification flag
    )
    
    # Setup training components
    if args.use_early_stopping:
        trainer.setup_early_stopping(patience=args.patience)
    
    if args.lr_scheduler != 'none':
        scheduler_config = {}
        if args.lr_scheduler == 'cosine':
            scheduler_config['T_max'] = args.epochs
        elif args.lr_scheduler == 'reduce_on_plateau':
            scheduler_config['patience'] = args.patience // 2
        elif args.lr_scheduler == 'step':
            scheduler_config['step_size'] = args.epochs // 3
        
        trainer.setup_lr_scheduler(args.lr_scheduler, **scheduler_config)
    
    # Train model
    start_time = datetime.now()
    history = trainer.train(
        epochs=args.epochs,
        save_every=args.save_every,
        verbose=args.verbose
    )
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save training history
    history_path = os.path.join(exp_dir, 'results', f'{model_type}_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_history = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                json_history[key] = value.tolist()
            elif isinstance(value, list):
                json_history[key] = value
            else:
                json_history[key] = [value] if not isinstance(value, (list, np.ndarray)) else value
        json.dump(json_history, f, indent=2)
    
    # Plot training history
    plot_path = os.path.join(exp_dir, 'plots', f'{model_type}_training_history.png')
    plot_training_history(history, plot_path)
    
    # Results summary
    results = {
        'model_type': model_type,
        'training_time': training_time,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_loss': min(history['val_loss']),
        'best_val_acc': max(history['val_acc']),
    }
    
    return model, results, history

def evaluate_all_models(models_results, test_loader, device, exp_dir, binary_classification):
    """Evaluate all trained models on test set"""
    print(f"\n{'='*60}")
    print("EVALUATING ALL MODELS ON TEST SET")
    print(f"{'='*60}")
    
    test_results = {}
    
    for model_type, (model, train_results, history) in models_results.items():
        print(f"\nEvaluating {model_type}...")
        
        # Load best model weights
        best_model_path = os.path.join(exp_dir, 'checkpoints', model_type, 'best_model.pt')
        if os.path.exists(best_model_path):
            try:
                # Fix for the pickle error - use weights_only=False for now
                checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
            except Exception as e:
                print(f"Warning: Could not load best model checkpoint: {e}")
                print("Using current model state instead...")
        
        # Evaluate model
        task_names = ['main', 'noise_level', 'field_strength'] if model_type == 'multitask' else None
        eval_save_path = os.path.join(exp_dir, 'plots', f'{model_type}_evaluation.png')
        
        evaluation_results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            task_names=task_names,
            save_path=eval_save_path,
            binary_classification=binary_classification
        )
        
        # Store results
        test_results[model_type] = {
            'evaluation_results': evaluation_results,
            'train_results': train_results
        }
        
        # Print summary
        main_results = evaluation_results.get('main', evaluation_results.get(list(evaluation_results.keys())[0], {}))
        print(f"{model_type} Test Results:")
        print(f"  Accuracy: {main_results.get('accuracy', 0):.4f}")
        if 'roc_auc' in main_results:
            print(f"  ROC AUC: {main_results.get('roc_auc', 0):.4f}")
        if 'avg_precision' in main_results:
            print(f"  Avg Precision: {main_results.get('avg_precision', 0):.4f}")
    
    return test_results

def create_comparison_report(test_results, exp_dir):
    """Create comprehensive comparison report"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON REPORT")
    print(f"{'='*60}")
    
    # Prepare comparison data
    comparison_data = []
    
    for model_type, results in test_results.items():
        train_results = results['train_results']
        eval_results = results['evaluation_results']
        main_eval = eval_results.get('main', eval_results.get(list(eval_results.keys())[0], {}))
        
        comparison_data.append({
            'Model': model_type,
            'Parameters': f"{train_results['total_parameters']:,}",
            'Training Time (s)': f"{train_results['training_time']:.1f}",
            'Best Val Acc': f"{train_results['best_val_acc']:.4f}",
            'Test Accuracy': f"{main_eval.get('accuracy', 0):.4f}",
            'ROC AUC': f"{main_eval.get('roc_auc', 0):.4f}",
            'Avg Precision': f"{main_eval.get('avg_precision', 0):.4f}"
        })
    
    # Print table
    print("\nModel Comparison Table:")
    print("-" * 100)
    headers = list(comparison_data[0].keys())
    print(" | ".join(f"{h:>15}" for h in headers))
    print("-" * 100)
    
    for row in comparison_data:
        print(" | ".join(f"{str(row[h]):>15}" for h in headers))
    
    print("-" * 100)
    
    # Save detailed report
    report_path = os.path.join(exp_dir, 'results', 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    # Create performance visualization
    create_performance_plot(comparison_data, exp_dir)
    
    return comparison_data

def create_performance_plot(comparison_data, exp_dir):
    """Create performance comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = [d['Model'] for d in comparison_data]
    test_acc = [float(d['Test Accuracy']) for d in comparison_data]
    params = [int(d['Parameters'].replace(',', '')) for d in comparison_data]
    training_time = [float(d['Training Time (s)']) for d in comparison_data]
    roc_auc = [float(d['ROC AUC']) if d['ROC AUC'] != '0.0000' else 0 for d in comparison_data]
    
    # Test Accuracy Comparison
    axes[0, 0].bar(models, test_acc, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Parameter Count vs Accuracy
    axes[0, 1].scatter(params, test_acc, color='red', alpha=0.7, s=100)
    for i, model in enumerate(models):
        axes[0, 1].annotate(model, (params[i], test_acc[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_title('Parameters vs Accuracy')
    axes[0, 1].set_xlabel('Number of Parameters')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_xscale('log')
    
    # Training Time Comparison
    axes[1, 0].bar(models, training_time, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # ROC AUC Comparison (if available)
    if any(roc_auc):
        axes[1, 1].bar(models, roc_auc, color='orange', alpha=0.7)
        axes[1, 1].set_title('ROC AUC Comparison')
        axes[1, 1].set_ylabel('ROC AUC')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'ROC AUC not available\n(Multi-class or insufficient data)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('ROC AUC Comparison')
    
    plt.tight_layout()
    
    plot_path = os.path.join(exp_dir, 'plots', 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup
    setup_reproducibility(args.seed)
    device = setup_device(args.device)
    exp_dir = create_experiment_dir(args.output_dir, args.experiment_name)
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Binary classification mode: {args.binary_classification}")
    
    # Save configuration
    config_path = os.path.join(exp_dir, 'config.json')
    save_training_config(vars(args), config_path)
    
    # Prepare datasets
    train_loader, val_loader, test_loader, class_weights = prepare_datasets(args, device)
    
    # Train all models
    models_results = {}
    
    for model_type in args.models:
        try:
            model, results, history = train_single_model(
                model_type, args, train_loader, val_loader, 
                device, class_weights, exp_dir
            )
            models_results[model_type] = (model, results, history)
            
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"Skipping {model_type}...")
            continue
    
    if not models_results:
        print("No models were successfully trained!")
        return
    
    # Evaluate all models
    test_results = evaluate_all_models(models_results, test_loader, device, exp_dir, args.binary_classification)
    
    # Create comparison report
    comparison_data = create_comparison_report(test_results, exp_dir)
    
    # Final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Results saved in: {exp_dir}")
    print(f"Trained {len(models_results)} models: {list(models_results.keys())}")
    
    # Find best model
    if comparison_data:
        best_model = max(comparison_data, key=lambda x: float(x['Test Accuracy']))
        print(f"Best performing model: {best_model['Model']} (Accuracy: {best_model['Test Accuracy']})")
    
    print(f"\nExperiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()