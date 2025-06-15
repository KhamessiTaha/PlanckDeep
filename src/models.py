import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ImprovedCMBClassifier(nn.Module):
    """
    Improved CNN architecture for CMB classification with enhanced feature extraction
    """
    def __init__(self, num_classes=2, dropout_rate=0.3, input_size=128):
        super().__init__()
        
        self.input_size = input_size
        
        # Convolutional layers with batch normalization and residual connections
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers with skip connections
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 5
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.adaptive_pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

class ResidualBlock(nn.Module):
    """Enhanced Residual block with attention mechanism"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Channel attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.channel_attention(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CMBResNet(nn.Module):
    """
    Enhanced ResNet-style architecture optimized for CMB analysis with attention
    """
    def __init__(self, num_classes=2, block_counts=[2, 2, 2, 2], use_attention=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers with progressive attention
        self.layer1 = self._make_layer(64, 64, block_counts[0], use_attention=use_attention)
        self.layer2 = self._make_layer(64, 128, block_counts[1], stride=2, use_attention=use_attention)
        self.layer3 = self._make_layer(128, 256, block_counts[2], stride=2, use_attention=use_attention)
        self.layer4 = self._make_layer(256, 512, block_counts[3], stride=2, use_attention=use_attention)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_attention=False):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_attention))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_attention=use_attention))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class PhysicsInformedCMBNet(nn.Module):
    """
    Physics-Informed Neural Network for CMB analysis
    Incorporates domain knowledge about CMB power spectra and angular correlations
    """
    def __init__(self, num_classes=2, dropout_rate=0.3, include_power_spectrum=True):
        super().__init__()
        
        self.include_power_spectrum = include_power_spectrum
        
        # Standard CNN backbone for spatial features
        self.spatial_backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Physics-informed features extraction
        self.physics_branch = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Power spectrum analysis branch
        if self.include_power_spectrum:
            self.power_spectrum_branch = nn.Sequential(
                nn.Linear(32, 256),  # Power spectrum features
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            feature_size = 128 * 4 * 4 + 64  # Physics + power spectrum features
        else:
            feature_size = 128 * 4 * 4  # Only physics features
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def extract_power_spectrum_features(self, x):
        """Extract power spectrum features using 2D FFT"""
        batch_size, _, h, w = x.size()
        device = x.device
        power_features_list = []
        
        for i in range(batch_size):
            patch = x[i, 0]  # Remove channel dimension
            
            # Compute 2D FFT
            fft_patch = torch.fft.fft2(patch)
            power_spectrum = torch.abs(fft_patch) ** 2
            
            # Extract radial power spectrum
            h, w = patch.shape
            center_y, center_x = h // 2, w // 2
            
            # Create coordinate grids
            y, x_coords = torch.meshgrid(torch.arange(h, device=device), 
                                       torch.arange(w, device=device), 
                                       indexing='ij')
            
            # Calculate radial distances
            r = torch.sqrt((x_coords - center_x).float()**2 + (y - center_y).float()**2)
            
            # Bin the power spectrum radially (32 bins)
            max_r = min(h, w) // 2
            r_bins = torch.linspace(0, max_r, 33, device=device)
            radial_power = torch.zeros(32, device=device)
            
            for j in range(32):
                mask = (r >= r_bins[j]) & (r < r_bins[j+1])
                if mask.sum() > 0:
                    radial_power[j] = power_spectrum[mask].mean()
            
            power_features_list.append(radial_power)
        
        # Stack all power spectra and take log (as in observational cosmology)
        power_features = torch.stack(power_features_list)
        power_features = torch.log(power_features + 1e-10)  # Avoid log(0)
        
        # Reduce to 64 features using linear transformation
        power_features_reduced = power_features[:, :32]  # Take first 32 features
        
        # Pad to 64 features if needed
        if power_features_reduced.shape[1] < 64:
            padding = torch.zeros(batch_size, 64 - power_features_reduced.shape[1], device=device)
            power_features_reduced = torch.cat([power_features_reduced, padding], dim=1)
        else:
            power_features_reduced = power_features_reduced[:, :64]
        
        return power_features_reduced
    
    def forward(self, x):
        # Extract spatial features
        spatial_features = self.spatial_backbone(x)
        physics_features = self.physics_branch(spatial_features)
        physics_features_flat = physics_features.view(physics_features.size(0), -1)
        
        if self.include_power_spectrum:
            # Extract power spectrum features
            power_features = self.extract_power_spectrum_features(x)
            power_features = self.power_spectrum_branch(power_features)
            
            # Combine features
            combined_features = torch.cat([physics_features_flat, power_features], dim=1)
        else:
            combined_features = physics_features_flat
        
        # Final classification
        output = self.classifier(combined_features)
        
        return output

class VisionTransformerCMB(nn.Module):
    """
    Vision Transformer adapted for CMB analysis
    """
    def __init__(self, img_size=128, patch_size=16, num_classes=2, embed_dim=768, 
                 depth=12, num_heads=12, dropout_rate=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification head (use only class token)
        x = self.norm(x[:, 0])
        x = self.dropout(x)
        x = self.head(x)
        
        return x

class MultiTaskCMBNet(nn.Module):
    """
    Multi-task learning network for various CMB classification tasks
    """
    def __init__(self, task_configs, shared_backbone='resnet', dropout_rate=0.3):
        super().__init__()
        
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # Shared backbone
        if shared_backbone == 'resnet':
            self.backbone = self._build_resnet_backbone()
            backbone_features = 512
        elif shared_backbone == 'cnn':
            self.backbone = self._build_cnn_backbone()
            backbone_features = 256
        else:
            raise ValueError(f"Unknown backbone: {shared_backbone}")
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            num_classes = config['num_classes']
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(backbone_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes)
            )
    
    def _build_resnet_backbone(self):
        return nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def _build_cnn_backbone(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
    
    def forward(self, x):
        # Extract shared features
        shared_features = self.backbone(x)
        
        # Apply task-specific heads
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_heads[task_name](shared_features)
        
        return outputs

# Factory function to create models
def create_model(model_type, num_classes=2, **kwargs):
    """
    Factory function to create different model types
    """
    if model_type == 'improved_cnn':
        return ImprovedCMBClassifier(num_classes=num_classes, **kwargs)
    elif model_type == 'resnet':
        return CMBResNet(num_classes=num_classes, **kwargs)
    elif model_type == 'physics_informed':
        return PhysicsInformedCMBNet(num_classes=num_classes, **kwargs)
    elif model_type == 'vision_transformer':
        return VisionTransformerCMB(num_classes=num_classes, **kwargs)
    elif model_type == 'multitask':
        return MultiTaskCMBNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model, input_size=(1, 1, 128, 128)):
    """Get a summary of model architecture"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    total_params = count_parameters(model)
    
    try:
        output = model(dummy_input)
        if isinstance(output, dict):
            output_shapes = {k: v.shape for k, v in output.items()}
        else:
            output_shapes = output.shape
    except Exception as e:
        output_shapes = f"Error: {e}"
    
    return {
        'total_parameters': total_params,
        'input_shape': input_size,
        'output_shape': output_shapes,
        'model_type': type(model).__name__
    }