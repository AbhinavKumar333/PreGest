"""
Transformer model architecture for gesture recognition.
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from .config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        
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
        
        # Register as buffer (not trainable)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class GestureTransformer(nn.Module):
    """Transformer model for hand gesture classification."""
    
    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        num_classes: int = 25,
        max_seq_len: int = 60,
        dropout: float = 0.1
    ):
        """Initialize Transformer model.
        
        Args:
            input_dim: Input feature dimension (63 = 21 joints * 3 coords)
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of gesture classes
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, 63]
            
        Returns:
            Output logits of shape [batch, num_classes]
        """
        # Embed input: [batch, seq_len, 63] → [batch, seq_len, hidden_dim]
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoder: [batch, seq_len, hidden_dim]
        x = self.transformer(x)
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Global average pooling: [batch, seq_len, hidden_dim] → [batch, hidden_dim]
        x = x.mean(dim=1)
        
        # Classification: [batch, hidden_dim] → [batch, num_classes]
        logits = self.classifier(x)
        
        return logits


def create_model(
    input_dim: int = 63,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    num_classes: int = 25,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> GestureTransformer:
    """Create and initialize a Transformer model.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_classes: Number of gesture classes
        dropout: Dropout probability
        device: Device to move model to
        
    Returns:
        Initialized GestureTransformer model
    """
    model = GestureTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )
    
    if device is not None:
        model = model.to(device)
    
    return model
