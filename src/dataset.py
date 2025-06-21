import torch
from torch.utils.data import Dataset
import numpy as np

class EnhancedCMBDataset(Dataset):
    def __init__(self, patches, labels, num_classes=2, augment=True, 
                 normalize_patches=True, transform_strength=0.3):
        """
        Args:
            num_classes: 1 for binary classification (using BCEWithLogitsLoss)
                        >2 for multi-class classification
                        Set to 0 for regression tasks
        """
        # Store parameters
        self.augment = augment
        self.normalize_patches = normalize_patches
        self.transform_strength = transform_strength
        self.num_classes = num_classes
        self.is_regression = (num_classes == 0)
        
        # Convert patches to tensor
        if isinstance(patches, np.ndarray):
            self.patches = torch.from_numpy(patches).float()
        else:
            self.patches = torch.as_tensor(patches).float()
            
        # Add channel dimension if needed
        if len(self.patches.shape) == 3:
            self.patches = self.patches.unsqueeze(1)  # [N, 1, H, W]
        
        # Convert and validate labels
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels)
        else:
            self.labels = torch.as_tensor(labels)
            
        # Ensure proper label type and shape
        if self.is_regression:
            self.labels = self.labels.float()
        elif num_classes == 1:  # Binary classification
            self.labels = self.labels.float()
            if self.labels.dim() > 1:
                self.labels = self.labels.squeeze(-1)
        else:  # Multi-class classification
            self.labels = self.labels.long()
            if self.labels.dim() > 1:
                self.labels = self.labels.squeeze(-1)
            
            # Validate class indices
            if (self.labels < 0).any() or (self.labels >= num_classes).any():
                raise ValueError("Class indices must be in [0, num_classes-1]")
        
        # Normalize if requested
        if self.normalize_patches:
            self._normalize_patches()
        
        # Print statistics
        print(f"Dataset loaded: {len(self.patches)} patches")
        print(f"Patch shape: {self.patches[0].shape}")
        
        if self.is_regression:
            print(f"Regression labels - Mean: {self.labels.mean():.4f}, Std: {self.labels.std():.4f}")
        elif num_classes == 1:
            print(f"Binary labels - Positive ratio: {self.labels.mean():.4f}")
        else:
            label_counts = torch.bincount(self.labels)
            print(f"Class distribution: {label_counts.tolist()}")
            if len(label_counts) >= 2:
                imbalance_ratio = float(label_counts.max()) / float(label_counts.min())
                if imbalance_ratio > 2.0:
                    print(f"⚠️ Class imbalance detected: ratio = {imbalance_ratio:.2f}")
    
    def _normalize_patches(self):
        """Normalize each patch to have zero mean and unit variance"""
        for i in range(len(self.patches)):
            patch = self.patches[i, 0]  # Remove channel dimension for normalization
            mean = torch.mean(patch)
            std = torch.std(patch)
            if std > 1e-8:  # Avoid division by zero
                self.patches[i, 0] = (patch - mean) / std
            else:
                self.patches[i, 0] = patch - mean
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx].clone()
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augment and torch.rand(1) > 0.5:
            # Random rotation (90, 180, 270 degrees)
            k = torch.randint(1, 4, (1,)).item()
            patch = torch.rot90(patch, k=k, dims=[1, 2])
            
            # Random flips
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[1])
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[2])
            
            # Add Gaussian noise augmentation
            if torch.rand(1) > 0.3:
                noise_std = self.transform_strength * 0.1 * torch.std(patch)
                noise = torch.randn_like(patch) * noise_std
                patch = patch + noise
        
        return patch, label


class MultiTaskCMBDataset(Dataset):
    """
    Dataset for multi-task learning with multiple label types
    """
    def __init__(self, patches, label_dict, augment=True, normalize_patches=True):
        """
        Args:
            patches: Array/tensor containing CMB patches
            label_dict (dict): Dictionary mapping task names to label arrays
                e.g., {'temperature': temp_labels, 'variance': var_labels}
            augment (bool): Whether to apply data augmentation
            normalize_patches (bool): Whether to normalize each patch individually
        """
        self.augment = augment
        self.normalize_patches = normalize_patches
        
        # Convert patches to torch tensors
        if isinstance(patches, np.ndarray):
            self.patches = torch.from_numpy(patches).float()
        else:
            self.patches = torch.tensor(patches).float()
            
        # Add channel dimension if not present
        if len(self.patches.shape) == 3:  # (N, H, W)
            self.patches = self.patches.unsqueeze(1)  # (N, 1, H, W)
        
        # Additional normalization if requested
        if self.normalize_patches:
            self._normalize_patches()
        
        # Load multiple label types
        self.labels = {}
        for task, labels in label_dict.items():
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            else:
                labels = torch.tensor(labels)
            
            # Determine if task is regression or classification
            # Assume classification if labels are integers in a small range
            unique_labels = torch.unique(labels)
            if len(unique_labels) <= 10 and torch.all(labels == labels.long()):
                self.labels[task] = labels.long()
            else:
                self.labels[task] = labels.float()
        
        self.task_names = list(self.labels.keys())
        
        print(f"Multi-task dataset loaded: {len(self.patches)} patches")
        print(f"Tasks: {self.task_names}")
        for task_name in self.task_names:
            task_labels = self.labels[task_name]
            if task_labels.dtype == torch.long:
                print(f"  {task_name} (classification): {torch.bincount(task_labels)}")
            else:
                print(f"  {task_name} (regression): Mean={task_labels.mean():.4f}, Std={task_labels.std():.4f}")
    
    def _normalize_patches(self):
        """Normalize each patch to have zero mean and unit variance"""
        for i in range(len(self.patches)):
            patch = self.patches[i, 0]  # Remove channel dimension for normalization
            mean = torch.mean(patch)
            std = torch.std(patch)
            if std > 1e-8:  # Avoid division by zero
                self.patches[i, 0] = (patch - mean) / std
            else:
                self.patches[i, 0] = patch - mean
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx].clone()
        
        # Apply augmentation during training
        if self.augment and torch.rand(1) > 0.5:
            # Random rotation (90, 180, 270 degrees)
            k = torch.randint(1, 4, (1,)).item()
            patch = torch.rot90(patch, k=k, dims=[1, 2])
            
            # Random flips
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[1])
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[2])
        
        # Return patch and all labels
        labels = {task_name: self.labels[task_name][idx] for task_name in self.task_names}
        
        return patch, labels