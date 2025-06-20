import torch
from torch.utils.data import Dataset
import numpy as np

class EnhancedCMBDataset(Dataset):
    """
    Enhanced CMB Dataset with data augmentation and multiple label types
    """
    def __init__(self, patches, labels, augment=True, normalize_patches=True, transform_strength=0.3):
        """
        Args:
            patches: Array/tensor containing CMB patches
            labels: Array/tensor containing labels
            augment (bool): Whether to apply data augmentation
            normalize_patches (bool): Whether to normalize each patch individually
            transform_strength (float): Strength of augmentation transforms
        """
        # Store parameters
        self.augment = augment
        self.normalize_patches = normalize_patches
        self.transform_strength = transform_strength
        
        # Convert to torch tensors
        self.patches = torch.tensor(patches).unsqueeze(1).float()
        self.labels = torch.tensor(labels).long()
        
        # Additional normalization if requested
        if self.normalize_patches:
            self._normalize_patches()
        
        print(f"Dataset loaded: {len(self.patches)} patches")
        print(f"Patch shape: {self.patches[0].shape}")
        print(f"Label distribution: {torch.bincount(self.labels)}")
        
        # Check for class imbalance
        label_counts = torch.bincount(self.labels)
        if len(label_counts) == 2:
            imbalance_ratio = float(label_counts.max()) / float(label_counts.min())
            if imbalance_ratio > 2.0:
                print(f"⚠️  Class imbalance detected: ratio = {imbalance_ratio:.2f}")
    
    def _normalize_patches(self):
        """Normalize each patch to have zero mean and unit variance"""
        for i in range(len(self.patches)):
            patch = self.patches[i, 0]  # Remove channel dimension for normalization
            mean = torch.mean(patch)
            std = torch.std(patch)
            if std > 0:
                self.patches[i, 0] = (patch - mean) / std
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx].clone()
        label = self.labels[idx]
        
        # Apply augmentation
        if self.augment and torch.rand(1) > 0.5:
            # Random rotation
            k = torch.randint(1, 4, (1,)).item()
            patch = torch.rot90(patch, k=k, dims=[1, 2])
            
            # Random flips
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[1])
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[2])
            
            # Add noise augmentation
            if torch.rand(1) > 0.3:
                noise = torch.randn_like(patch) * self.transform_strength * 0.1
                patch = patch + noise
        
        return patch, label


class MultiTaskCMBDataset(Dataset):
    """
    Dataset for multi-task learning with multiple label types
    """
    def __init__(self, patches, label_dict, augment=True):
        """
        Args:
            patches: Array/tensor containing CMB patches
            label_dict (dict): Dictionary mapping task names to label arrays
                e.g., {'temperature': temp_labels, 'variance': var_labels}
            augment (bool): Whether to apply data augmentation
        """
        self.patches = torch.tensor(patches).unsqueeze(1).float()
        self.augment = augment
        
        # Load multiple label types
        self.labels = {task: torch.tensor(labels).long() for task, labels in label_dict.items()}
        
        self.task_names = list(self.labels.keys())
        
        print(f"Multi-task dataset loaded: {len(self.patches)} patches")
        print(f"Tasks: {self.task_names}")
        for task_name in self.task_names:
            print(f"  {task_name}: {torch.bincount(self.labels[task_name])}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx].clone()
        
        # Apply augmentation
        if self.augment and torch.rand(1) > 0.5:
            k = torch.randint(1, 4, (1,)).item()
            patch = torch.rot90(patch, k=k, dims=[1, 2])
            
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[1])
            if torch.rand(1) > 0.5:
                patch = torch.flip(patch, dims=[2])
        
        # Return patch and all labels
        labels = {task_name: self.labels[task_name][idx] for task_name in self.task_names}
        
        return patch, labels