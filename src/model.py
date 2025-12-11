"""Multi-modal Transformer model for gesture spotting from RGB + mask sequences"""

import math
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class VisualFeatureExtractor(nn.Module):
    """Feature extractor for RGB and mask images using configurable backbone"""
    
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
        """Initialize visual feature extractor"""
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == 'resnet18':
            # Load ResNet18 with pretrained ImageNet weights
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception as e:
                print(f"Warning: Failed to load pretrained weights ({e}), using random initialization")
                model = models.resnet18(weights=None)
            
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512  
            
        elif backbone == 'squeezenet':
            # Load SqueezeNet with pretrained weights
            try:
                model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception as e:
                print(f"Warning: Failed to load pretrained weights ({e}), using random initialization")
                model = models.squeezenet1_1(weights=None)
            
            # Remove final classifier
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            # Add adaptive avg pooling
            self.backbone.add_module('adaptive_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
            self.feature_dim = 512 
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'squeezenet'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images"""
        features = self.backbone(x)
        return features.squeeze(-1).squeeze(-1)


class MultiModalFusion(nn.Module):
    """Fusion layer for RGB and mask features"""
    def __init__(self, rgb_dim: int = 512, mask_dim: int = 512, fusion_dim: int = 256):
        """Initialize multi-modal fusion"""
        super().__init__()
        
        # Project RGB features to fusion space
        self.rgb_proj = nn.Linear(rgb_dim, fusion_dim)
        
        # Project mask features to fusion space
        self.mask_proj = nn.Linear(mask_dim, fusion_dim)
        
        # Fuse via MLP
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, rgb_feat: torch.Tensor, mask_feat: torch.Tensor) -> torch.Tensor:
        """Fuse RGB and mask features"""
        rgb_proj = self.rgb_proj(rgb_feat)  
        mask_proj = self.mask_proj(mask_feat)  
        concat = torch.cat([rgb_proj, mask_proj], dim=-1)  
        fused = self.fusion(concat)  
        return fused


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding"""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GestureSpottingTransformer(nn.Module):
    """Multi-modal gesture spotting transformer"""
    
    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = 'resnet18',
        rgb_pretrained: bool = True,
        mask_pretrained: bool = False,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.3,
        max_seq_len: int = 60
    ):
        """Initialize GestureSpottingTransformer"""
        super().__init__()
        
        # Visual encoders
        self.rgb_encoder = VisualFeatureExtractor(backbone=backbone, pretrained=rgb_pretrained)
        self.mask_encoder = VisualFeatureExtractor(backbone=backbone, pretrained=mask_pretrained)
        
        # Get feature dimension from encoder
        feature_dim = self.rgb_encoder.feature_dim
        
        # Validate dimensions for batch normalization compatibility
        assert feature_dim == 512, f"Expected feature_dim=512 from ResNet18, got {feature_dim}"
        assert fusion_dim > 0, f"fusion_dim must be positive, got {fusion_dim}"
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(feature_dim, feature_dim, fusion_dim)
        
        # Batch normalization layers (Quest 3)
        self.rgb_bn = nn.BatchNorm1d(feature_dim)
        self.mask_bn = nn.BatchNorm1d(feature_dim)
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)
        
        # Project to transformer hidden dimension
        self.input_proj = nn.Linear(fusion_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),  
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in [self.input_proj, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def encode_images(self, rgb_images: torch.Tensor, mask_images: torch.Tensor) -> torch.Tensor:
        """Encode RGB and mask image sequences into fused features"""
        B, T = rgb_images.shape[:2]
        
        # Reshape for batch processing of all frames
        rgb_flat = rgb_images.view(B * T, 3, *rgb_images.shape[-2:])  
        mask_flat = mask_images.view(B * T, 1, *mask_images.shape[-2:])  
        
        # Convert mask to 3-channel for ResNet
        mask_flat = mask_flat.repeat(1, 3, 1, 1)  
        
        # Extract features
        rgb_feat = self.rgb_encoder(rgb_flat)  
        mask_feat = self.mask_encoder(mask_flat)  
        
        # Apply batch normalization (Quest 3)
        rgb_feat = self.rgb_bn(rgb_feat)  
        mask_feat = self.mask_bn(mask_feat)  
        
        rgb_feat = rgb_feat.view(B, T, -1)
        mask_feat = mask_feat.view(B, T, -1)
        
        # Fuse
        fused = self.fusion(rgb_feat, mask_feat) 
        return fused
    
    def forward(
        self,
        rgb_images: torch.Tensor,
        mask_images: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for gesture spotting"""
        # Extract and fuse visual features
        features = self.encode_images(rgb_images, mask_images)  
        
        # Project to hidden dimension
        x = self.input_proj(features)  
        
        # Apply batch normalization after fusion (Quest 3)
        B, T, D = x.shape
        x = x.view(B * T, D)  
        x = self.fusion_bn(x)  
        x = x.view(B, T, D)  
        
        # Add positional encoding
        x = self.pos_encoder(x)  
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)  
        
        # Frame-level classification
        logits = self.classifier(x)  
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients by global norm"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


def create_model(
    num_classes: int = 8,
    backbone: str = 'resnet18',
    rgb_pretrained: bool = True,
    mask_pretrained: bool = False,
    fusion_dim: int = 256,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    feedforward_dim: int = 512,
    dropout: float = 0.3,
    device: Optional[torch.device] = None,
    max_seq_len: int = 30
) -> GestureSpottingTransformer:
    """Create a gesture spotting model"""
    model = GestureSpottingTransformer(
        num_classes=num_classes,
        backbone=backbone,
        rgb_pretrained=rgb_pretrained,
        mask_pretrained=mask_pretrained,
        fusion_dim=fusion_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    
    if device is not None:
        model = model.to(device)
    
    return model
