"""
Dueling Double DQN with Patch-based Self-Attention.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                 PATCH + SELF-ATTENTION ARCHITECTURE             │
    │                                                                 │
    │   Input: (B, 4, 64, 64) frames     4 stacked grayscale          │
    │           ↓                                                     │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │  Patch Embedding: Conv(4→32, kernel=16, stride=16)      │   │
    │   │                                                         │   │
    │   │    64×64 image → 4×4 grid = 16 patches                  │   │
    │   │    Each patch: 16×16 pixels → 32-dim embedding          │   │
    │   │    Output: (B, 32, 4, 4) = (B, 16 patches, 32 dims)     │   │
    │   │                                                         │   │
    │   │    ┌────┬────┬────┬────┐                                │   │
    │   │    │ P0 │ P1 │ P2 │ P3 │  Top row (sky/score)           │   │
    │   │    ├────┼────┼────┼────┤                                │   │
    │   │    │ P4 │ P5 │ P6 │ P7 │  Upper play area               │   │
    │   │    ├────┼────┼────┼────┤                                │   │
    │   │    │ P8 │ P9 │P10 │P11 │  Main play area (Mario here)   │   │
    │   │    ├────┼────┼────┼────┤                                │   │
    │   │    │P12 │P13 │P14 │P15 │  Ground level                  │   │
    │   │    └────┴────┴────┴────┘                                │   │
    │   └─────────────────────────────────────────────────────────┘   │
    │           ↓                                                     │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │  Self-Attention over 16 patches                         │   │
    │   │                                                         │   │
    │   │  Each patch attends to all other patches:               │   │
    │   │  • P9 (Mario) attends to P10 (ahead) for enemies        │   │
    │   │  • P13 (ground) attends to P14 (gap detection)          │   │
    │   │                                                         │   │
    │   │  Attention = softmax(Q @ K.T / √d) @ V                  │   │
    │   │  Output = Input + Attention (residual)                  │   │
    │   └─────────────────────────────────────────────────────────┘   │
    │           ↓                                                     │
    │   Flatten(16×32=512) → LayerNorm → FC(512) → GELU               │
    │           ↓                                                     │
    │   ┌────────────┴────────────┐                                   │
    │   ↓                         ↓                                   │
    │   Value Stream          Advantage Stream                        │
    │   FC(512→256)→GELU      FC(512→256)→GELU                        │
    │   FC(256→1)=V(s)        FC(256→A)=A(s,a)                        │
    │           ↓                         ↓                           │
    │           └────────────┬────────────┘                           │
    │                        ↓                                        │
    │         Q(s,a) = V(s) + (A(s,a) - mean(A))                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Key Design Choices:
- Single conv layer for patch embedding (simpler than 3-layer CNN)
- 16×16 patches give 4×4 grid = 16 tokens (manageable for attention)
- 32 channels keep model small but expressive
- Self-attention learns spatial relationships between patches
- Residual connection for stable gradient flow

Inspired by:
- Vision Transformer (ViT): patch embedding + self-attention
- Decision Transformer: attention for RL
"""

from typing import Tuple

import torch
import numpy as np
from torch import nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# Self-Attention Module for Spatial Features
# =============================================================================

class SpatialAttention(nn.Module):
    """
    Self-Attention Module for Spatial Features - Classical Transformer-style attention.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SELF-ATTENTION FOR VISION                            │
    │                                                                         │
    │  Problem with CBAM-style attention:                                     │
    │  • Per-position sigmoid can ALL saturate to 1.0 (no competition)        │
    │  • Doesn't learn relationships BETWEEN positions                        │
    │                                                                         │
    │  Self-attention solution:                                               │
    │  • Softmax over ALL positions: attention sums to 1.0 (competition!)     │
    │  • Each position can attend to every other position                     │
    │  • Learns: "When Mario is HERE, look at enemies THERE"                  │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  Feature Map 4×4 = 16 positions                                 │    │
    │  │                                                                 │    │
    │  │    Position 0   Position 1   ...   Position 15                  │    │
    │  │       ↓            ↓                   ↓                        │    │
    │  │    ┌─────┐      ┌─────┐           ┌─────┐                       │    │
    │  │    │ Q₀  │      │ Q₁  │    ...    │ Q₁₅ │  Queries              │    │
    │  │    │ K₀  │      │ K₁  │    ...    │ K₁₅ │  Keys                 │    │
    │  │    │ V₀  │      │ V₁  │    ...    │ V₁₅ │  Values               │    │
    │  │    └─────┘      └─────┘           └─────┘                       │    │
    │  │                                                                 │    │
    │  │    Attention[i,j] = softmax(Qᵢ · Kⱼ / √d)                       │    │
    │  │                                                                 │    │
    │  │    ┌──────────────────────────────────────┐                     │    │
    │  │    │  Attention Matrix (16×16)            │                     │    │
    │  │    │  ┌─────────────────────────────────┐ │                     │    │
    │  │    │  │ 0.6  0.1  0.1  0.1  0.05 ...    │ │ Position 0 attends  │    │
    │  │    │  │ 0.1  0.5  0.2  0.1  0.05 ...    │ │ mostly to itself    │    │
    │  │    │  │ 0.05 0.1  0.7  0.05 0.05 ...    │ │ but also to         │    │
    │  │    │  │ ...                             │ │ nearby positions    │    │
    │  │    │  └─────────────────────────────────┘ │                     │    │
    │  │    │  Each ROW sums to 1.0 (softmax!)     │                     │    │
    │  │    └──────────────────────────────────────┘                     │    │
    │  │                                                                 │    │
    │  │    Output[i] = Σⱼ Attention[i,j] × Vⱼ                           │    │
    │  │              = weighted combination of all values               │    │
    │  │                                                                 │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Architecture:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │   Input: Feature maps (B, C, H, W) from conv3                           │
    │          e.g., (32, 64, 4, 4) = batch 32, 64 channels, 4×4 spatial      │
    │                                                                         │
    │   Step 1: Reshape to sequence                                           │
    │   ────────────────────────────                                          │
    │   (B, C, H, W) → (B, H*W, C) = (B, 16, 64)                              │
    │   Each of the 16 positions becomes a "token" with 64 features           │
    │                                                                         │
    │   Step 2: Project to Q, K, V                                            │
    │   ───────────────────────────                                           │
    │   Q = X @ Wq   (B, 16, d)  "What am I looking for?"                     │
    │   K = X @ Wk   (B, 16, d)  "What do I contain?"                         │
    │   V = X @ Wv   (B, 16, d)  "What is my value?"                          │
    │                                                                         │
    │   Step 3: Compute attention scores                                      │
    │   ──────────────────────────────                                        │
    │   scores = Q @ K.T / √d    (B, 16, 16)                                  │
    │   attention = softmax(scores, dim=-1)  ← SUMS TO 1 PER ROW!             │
    │                                                                         │
    │   Step 4: Apply attention to values                                     │
    │   ─────────────────────────────────                                     │
    │   output = attention @ V   (B, 16, d)                                   │
    │                                                                         │
    │   Step 5: Project back and reshape                                      │
    │   ────────────────────────────────                                      │
    │   output = output @ Wo     (B, 16, C)                                   │
    │   output → (B, C, H, W)    reshape back to spatial                      │
    │                                                                         │
    │   Step 6: Residual connection                                           │
    │   ───────────────────────────                                           │
    │   output = input + output  (helps gradient flow)                        │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Why self-attention works better:
    - Softmax FORCES competition between positions (can't all be 1.0)
    - Learns spatial relationships (Mario position ↔ Enemy position)
    - Proven effective in Vision Transformers, Decision Transformer, Gato
    - For 4×4 feature map: only 16×16=256 attention weights (cheap!)
    
    References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    - "Decision Transformer" (Chen et al., 2021)
    """
    
    def __init__(self, in_channels: int = 64, num_heads: int = 4, dropout: float = 0.0):
        """
        Initialize self-attention module.
        
        Args:
            in_channels: Number of input channels (from conv3, typically 64)
            num_heads: Number of attention heads (parallel attention patterns)
                      More heads = more diverse attention patterns
                      Must divide in_channels evenly
            dropout: Dropout rate for attention weights (0 = no dropout)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5  # 1/√d for scaled dot-product attention
        
        assert in_channels % num_heads == 0, f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        # Q, K, V projections (combined into one linear for efficiency)
        # Projects from C to 3*C, then split into Q, K, V
        self.qkv = nn.Linear(in_channels, 3 * in_channels, bias=False)
        
        # Output projection
        self.proj = nn.Linear(in_channels, in_channels, bias=False)
        
        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)
        
        # Initialize with small weights for stable start
        nn.init.xavier_normal_(self.qkv.weight, gain=0.1)
        nn.init.xavier_normal_(self.proj.weight, gain=0.1)
        
        # Store attention map for visualization (shape: B, num_heads, H*W, H*W)
        self.last_attention_map: torch.Tensor | None = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to spatial features.
        
        Args:
            x: Input feature maps (B, C, H, W)
        
        Returns:
            Attended features (B, C, H, W) - same shape as input
        
        Side effect:
            Stores attention weights in self.last_attention_map for visualization
        """
        B, C, H, W = x.shape
        N = H * W  # Number of spatial positions (tokens)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Reshape to sequence (B, C, H, W) → (B, N, C)
        # ─────────────────────────────────────────────────────────────────────
        x_seq = x.flatten(2).transpose(1, 2)  # (B, N, C)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Project to Q, K, V
        # ─────────────────────────────────────────────────────────────────────
        qkv = self.qkv(x_seq)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Compute attention scores
        # ─────────────────────────────────────────────────────────────────────
        # Scaled dot-product attention: softmax(Q @ K.T / √d)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Softmax over keys (last dim)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Store for visualization (average over heads for simpler viz)
        self.last_attention_map = attn_weights.mean(dim=1).detach()  # (B, N, N)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Apply attention to values
        # ─────────────────────────────────────────────────────────────────────
        attended = attn_weights @ v  # (B, num_heads, N, head_dim)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Combine heads and project
        # ─────────────────────────────────────────────────────────────────────
        attended = attended.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        output = self.proj(attended)  # (B, N, C)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 6: Reshape back to spatial and add residual
        # ─────────────────────────────────────────────────────────────────────
        output = output.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        output = x + output  # Residual connection
        
        return output
    
    def get_attention_map(self) -> torch.Tensor | None:
        """
        Get the last computed attention map for visualization.
        
        Returns:
            Attention map (B, N, N) where N = H*W (e.g., 16 for 4×4).
            Values in [0, 1], each row sums to 1.0.
            
            To visualize as spatial heatmap for a specific query position:
            >>> attn = attention.get_attention_map()  # (B, 16, 16)
            >>> # How much does position 8 (center) attend to other positions?
            >>> center_attn = attn[0, 8, :].reshape(4, 4)  # (4, 4)
            >>> plt.imshow(center_attn, cmap='hot')
            
            For overall attention "importance" (how much each position is attended to):
            >>> importance = attn[0].sum(dim=0).reshape(4, 4)  # Sum over queries
            >>> plt.imshow(importance, cmap='hot')
        """
        return self.last_attention_map
    
    def get_spatial_importance(self, h: int = 4, w: int = 4) -> torch.Tensor | None:
        """
        Get spatial importance map (how much each position is attended to overall).
        
        This is more intuitive for visualization than the full attention matrix.
        Positions that are attended to by many other positions are "important".
        
        Args:
            h, w: Spatial dimensions to reshape to
        
        Returns:
            Importance map (B, 1, H, W) suitable for overlay visualization.
            Higher values = position is attended to by more queries.
        """
        if self.last_attention_map is None:
            return None
        
        # Sum attention received by each key position (column sum)
        # Shape: (B, N, N) → sum over dim=1 (queries) → (B, N)
        importance = self.last_attention_map.sum(dim=1)  # (B, N)
        
        # Normalize to [0, 1] for visualization
        importance = importance - importance.min(dim=1, keepdim=True)[0]
        importance = importance / (importance.max(dim=1, keepdim=True)[0] + 1e-8)
        
        # Reshape to spatial
        B = importance.shape[0]
        return importance.reshape(B, 1, h, w)


class DDQNBackbone(nn.Module):
    """
    CNN + Self-Attention backbone for DDQN with 8×8 attention grid.
    
    Uses 3 conv layers to get 8×8 spatial resolution, then applies
    self-attention over 64 positions for finer-grained spatial reasoning.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │          FINER-GRAINED 8×8 SELF-ATTENTION BACKBONE                      │
    │                                                                         │
    │   Input: (B, 4, 64, 64) - 4 stacked grayscale frames                   │
    │          │                                                              │
    │          ▼                                                              │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │  Conv1: 4→32, kernel=3, stride=2, padding=1                     │   │
    │   │         64×64 → 32×32                                           │   │
    │   │         GELU activation                                         │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │          │                                                              │
    │          ▼                                                              │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │  Conv2: 32→32, kernel=3, stride=2, padding=1                    │   │
    │   │         32×32 → 16×16                                           │   │
    │   │         GELU activation                                         │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │          │                                                              │
    │          ▼                                                              │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │  Conv3: 32→32, kernel=3, stride=2, padding=1                    │   │
    │   │         16×16 → 8×8                                             │   │
    │   │         GELU activation                                         │   │
    │   │                                                                 │   │
    │   │  Output: (B, 32, 8, 8) = 64 spatial positions × 32 features    │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │          │                                                              │
    │          ▼                                                              │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │  SELF-ATTENTION over 64 positions (8×8 grid)                    │   │
    │   │                                                                 │   │
    │   │  Each position covers 8×8 pixels of input (vs 16×16 before)    │   │
    │   │  → Can track individual enemies/obstacles more precisely        │   │
    │   │                                                                 │   │
    │   │  ┌──┬──┬──┬──┬──┬──┬──┬──┐                                     │   │
    │   │  │  │  │  │  │  │  │  │  │  8×8 = 64 positions                 │   │
    │   │  ├──┼──┼──┼──┼──┼──┼──┼──┤                                     │   │
    │   │  │  │  │  │  │  │  │  │  │  Each covers 8×8 input pixels       │   │
    │   │  ├──┼──┼──┼──┼──┼──┼──┼──┤                                     │   │
    │   │  │  │  │██│  │  │  │  │  │  ← Mario fits in ~1-2 cells         │   │
    │   │  ├──┼──┼──┼──┼──┼──┼──┼──┤                                     │   │
    │   │  │  │  │  │  │☠ │  │  │  │  ← Enemy in distinct cell           │   │
    │   │  ├──┼──┼──┼──┼──┼──┼──┼──┤                                     │   │
    │   │  │  │  │  │  │  │  │  │  │                                     │   │
    │   │  ├──┼──┼──┼──┼──┼──┼──┼──┤                                     │   │
    │   │  │  │  │  │  │  │  │  │  │                                     │   │
    │   │  ├──┼──┼──┼──┼──┼──┼──┼──┤                                     │   │
    │   │  │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│  ← Ground level                     │   │
    │   │  └──┴──┴──┴──┴──┴──┴──┴──┘                                     │   │
    │   │                                                                 │   │
    │   │  Attention cost: 64² = 4096 (still fast!)                      │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │          │                                                              │
    │          ▼                                                              │
    │   Flatten(64×32=2048) → FC(512) → LayerNorm → [+action] → FC(512)      │
    │          │                                                              │
    │          ▼                                                              │
    │   Output: (B, 512) feature vector for dueling heads                    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Why 8×8 is better than 4×4:
    ───────────────────────────
    • 4× more spatial positions (64 vs 16)
    • Each cell covers 8×8 pixels (vs 16×16)
    • A goomba (~8-12 pixels) now fits in 1-2 cells instead of being smeared
    • Can learn "look at cell (3,5) for that enemy" vs "somewhere in quadrant"
    
    Receptive Field (with 3×3 kernel, stride 2):
    ────────────────────────────────────────────
    After Conv1: 3×3 pixels
    After Conv2: 7×7 pixels
    After Conv3: 15×15 pixels
    
    Each of 64 positions sees ~15×15 pixel context - enough for local features
    while attention provides global reasoning.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        feature_dim: int = 512,
        dropout: float = 0.1,
        action_history_dim: int = 0,
        embed_dim: int = 32,
        num_heads: int = 4,
    ):
        """
        Initialize CNN + self-attention backbone with 8×8 attention.
        
        Args:
            input_shape: (C, H, W) input shape, e.g., (4, 64, 64)
            feature_dim: Output feature dimension (default 512)
            dropout: Dropout rate (default 0.1)
            action_history_dim: Dimension of action history to concatenate
            embed_dim: Number of channels throughout (default 32)
            num_heads: Number of attention heads (default 4)
        """
        super().__init__()
        c, h, w = input_shape
        self.action_history_dim = action_history_dim
        self.embed_dim = embed_dim
        
        # ─────────────────────────────────────────────────────────────────────
        # Convolutional stack: 3 layers with kernel=3, stride=2
        # 64×64 → 32×32 → 16×16 → 8×8 (stop here for finer attention)
        # ─────────────────────────────────────────────────────────────────────
        self.conv1 = layer_init(nn.Conv2d(c, embed_dim, kernel_size=3, stride=2, padding=1))
        self.conv2 = layer_init(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1))
        self.conv3 = layer_init(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1))
        # No conv4 - we want 8×8 spatial resolution for finer attention
        
        # Calculate output spatial size: 64 / 8 = 8
        self.spatial_h = h // 8  # 64 // 8 = 8
        self.spatial_w = w // 8  # 64 // 8 = 8
        self.num_positions = self.spatial_h * self.spatial_w  # 64
        
        # ─────────────────────────────────────────────────────────────────────
        # Self-Attention over 64 spatial positions (8×8 grid)
        # Attention cost: 64² = 4096 (still very fast)
        # ─────────────────────────────────────────────────────────────────────
        self.attention = SpatialAttention(in_channels=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # ─────────────────────────────────────────────────────────────────────
        # Output layers: 8×8×32 = 2048 → 512
        # ─────────────────────────────────────────────────────────────────────
        flat_size = self.num_positions * embed_dim  # 64 × 32 = 2048
        
        # Two-stage projection: 2048 → 512 → 512
        self.fc1 = layer_init(nn.Linear(flat_size, feature_dim))
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        fc_input_dim = feature_dim + action_history_dim
        self.fc2 = layer_init(nn.Linear(fc_input_dim, feature_dim))
        self.feature_dim = feature_dim
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through CNN + 8×8 attention backbone.
        
        Args:
            x: Input frames (B, C, H, W), MUST be uint8 [0, 255]
            action_history: Optional action history (B, history_len, num_actions)
        
        Returns:
            Feature vector (B, feature_dim) for dueling heads
        """
        # ─────────────────────────────────────────────────────────────────────
        # Normalize input: uint8 [0, 255] -> float32 [0, 1]
        # ─────────────────────────────────────────────────────────────────────
        assert x.dtype == torch.uint8, f"Input must be uint8, got {x.dtype}"
        x = x.float() / 255.0
        
        # ─────────────────────────────────────────────────────────────────────
        # Convolutional feature extraction: 3 layers to 8×8
        # ─────────────────────────────────────────────────────────────────────
        x = torch.nn.functional.gelu(self.conv1(x))  # (B, 32, 32, 32)
        x = torch.nn.functional.gelu(self.conv2(x))  # (B, 32, 16, 16)
        x = torch.nn.functional.gelu(self.conv3(x))  # (B, 32, 8, 8)
        # No conv4 - keep 8×8 for finer attention
        
        # ─────────────────────────────────────────────────────────────────────
        # Self-attention over 64 spatial positions (8×8 grid)
        # ─────────────────────────────────────────────────────────────────────
        x = self.attention(x)  # (B, 32, 8, 8) with residual connection
        
        # ─────────────────────────────────────────────────────────────────────
        # Flatten and project: 2048 → 512 → 512
        # ─────────────────────────────────────────────────────────────────────
        x = x.flatten(1)  # (B, 2048)
        x = torch.nn.functional.gelu(self.fc1(x))  # (B, 512)
        x = self.layer_norm(x)
        x = self.dropout_layer(x)
        
        # Concatenate action history if provided
        if action_history is not None:
            if action_history.dim() == 3:
                action_history = action_history.flatten(1)
            x = torch.cat([x, action_history], dim=1)
        elif self.action_history_dim > 0:
            x = torch.cat([x, torch.zeros(x.shape[0], self.action_history_dim, device=x.device)], dim=1)
        
        x = torch.nn.functional.gelu(self.fc2(x))
        return x
    
    def get_attention_map(self) -> torch.Tensor | None:
        """
        Get the spatial importance map for visualization.
        
        Returns:
            Importance map (B, 1, 8, 8) showing which positions are attended to.
            Higher values = position is attended to by more queries.
        """
        return self.attention.get_spatial_importance(
            h=self.spatial_h, 
            w=self.spatial_w
        )


class DuelingHead(nn.Module):
    """
    Dueling architecture head for Q-value estimation.

    Separates value V(s) and advantage A(s,a) streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

    This makes it easier to identify valuable states regardless
    of action choice, improving stability when many actions are similar.
    """

    def __init__(self, feature_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            layer_init(nn.Linear(feature_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Advantage stream: estimates A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            layer_init(nn.Linear(feature_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(features)  # (N, 1)
        advantages = self.advantage_stream(features)  # (N, num_actions)

        # Dueling aggregation: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DDQNNet(nn.Module):
    """
    Single Q-network for DDQN with dueling architecture.

    Features:
    - Nature DQN-style CNN backbone with modern improvements
    - Dueling architecture (separate value and advantage streams)
    - GELU activation, LayerNorm, Dropout, Orthogonal init
    - Raw linear Q-value output (unbounded)
    - Optional action history input for temporal context
    - Optional auxiliary danger prediction head
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        action_history_len: int = 4,
        danger_prediction_bins: int = 16,
    ):
        """
        Args:
            input_shape: Shape of frame stack (C, H, W)
            num_actions: Number of actions in action space
            feature_dim: Dimension of backbone feature output
            hidden_dim: Dimension of dueling head hidden layers
            dropout: Dropout rate for regularization
            action_history_len: Length of action history to use (0 = disabled).
                               When enabled, expects action_history tensor of shape
                               (N, action_history_len, num_actions) as second input.
            danger_prediction_bins: Number of bins for danger prediction auxiliary task (0 = disabled).
                                   When enabled, network predicts danger probability for each bin.
        """
        super().__init__()
        self.action_history_len = action_history_len
        self.danger_prediction_bins = danger_prediction_bins
        self.num_actions = num_actions
        
        # Calculate action history dimension (flattened one-hot)
        action_history_dim = action_history_len * num_actions if action_history_len > 0 else 0
        
        self.backbone = DDQNBackbone(
            input_shape, feature_dim, dropout, action_history_dim
        )
        self.head = DuelingHead(feature_dim, num_actions, hidden_dim)
        
        # Auxiliary danger prediction head (predicts danger probability per distance bin)
        if danger_prediction_bins > 0:
            self.danger_head = nn.Sequential(
                layer_init(nn.Linear(feature_dim, hidden_dim)),
                nn.GELU(),
                layer_init(nn.Linear(hidden_dim, danger_prediction_bins), std=0.01),
            )
        else:
            self.danger_head = None

    def forward(
        self,
        x: torch.Tensor,
        action_history: torch.Tensor | None = None,
        return_danger: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning Q-values for all actions.

        Args:
            x: Observation tensor (N, C, H, W) where C is frame stack
            action_history: Optional action history tensor (N, history_len, num_actions)
                           One-hot encoded previous actions.
            return_danger: If True, also return danger predictions.

        Returns:
            q_values: Q-values for each action (N, num_actions)
            danger_pred: (if return_danger=True) Danger predictions (N, danger_bins)
        """
        features = self.backbone(x, action_history)
        q_values = self.head(features)

        if return_danger and self.danger_head is not None:
            danger_pred = torch.sigmoid(self.danger_head(features))
            return q_values, danger_pred
        
        return q_values

    def get_action(
        self,
        x: torch.Tensor,
        epsilon: float = 0.0,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get action using epsilon-greedy policy.

        Args:
            x: Observation tensor (N, C, H, W)
            epsilon: Exploration rate (0 = greedy, 1 = random)
            action_history: Optional action history tensor

        Returns:
            action: Selected action indices (N,)
        """
        if np.random.random() < epsilon:
            # Random action
            batch_size = x.shape[0]
            return torch.randint(0, self.num_actions, (batch_size,), device=x.device)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(x, action_history)
                return q_values.argmax(dim=1)


class DoubleDQN(nn.Module):
    """
    Double DQN with online and target networks.

    Double DQN decouples action selection from Q-value estimation:
    - Action selection: argmax_a Q_online(s', a)
    - Q-value estimation: Q_target(s', argmax_a Q_online(s', a))

    This reduces overestimation bias common in standard DQN.

    Target network is updated via:
    - Soft update (polyak averaging): θ_target = τ * θ_online + (1-τ) * θ_target
    - Hard sync: copy online weights to target every N steps

    Features:
    - Raw linear Q-value output (unbounded)
    - Optional action history input for temporal context
    - Optional auxiliary danger prediction head
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        action_history_len: int = 4,
        danger_prediction_bins: int = 16,
    ):
        super().__init__()
        self.online = DDQNNet(
            input_shape, num_actions, feature_dim, hidden_dim, dropout,
            action_history_len, danger_prediction_bins
        )
        self.target = DDQNNet(
            input_shape, num_actions, feature_dim, hidden_dim, dropout,
            action_history_len, danger_prediction_bins
        )
        self.num_actions = num_actions
        self.action_history_len = action_history_len
        self.danger_prediction_bins = danger_prediction_bins

        # Copy online weights to target and freeze target
        self.sync_target()

    def forward(
        self,
        x: torch.Tensor,
        network: str = "online",
        action_history: torch.Tensor | None = None,
        return_danger: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through specified network.

        Args:
            x: Observation tensor (N, C, H, W)
            network: "online" or "target"
            action_history: Optional action history tensor (N, history_len, num_actions)
            return_danger: If True, also return danger predictions (online only)

        Returns:
            q_values: Q-values for each action (N, num_actions)
            danger_pred: (if return_danger=True) Danger predictions (N, danger_bins)
        """
        if network == "online":
            return self.online(x, action_history, return_danger)
        elif network == "target":
            return self.target(x, action_history, return_danger=False)
        raise ValueError(f"Unknown network: {network}")

    def get_action(
        self,
        x: torch.Tensor,
        epsilon: float = 0.0,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get action using online network with epsilon-greedy."""
        return self.online.get_action(x, epsilon, action_history)

    def sync_target(self) -> None:
        """Hard sync: copy online weights to target network."""
        self.target.load_state_dict(self.online.state_dict())
        # Freeze target parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def soft_update(self, tau: float = 0.005) -> None:
        """
        Soft update target network (polyak averaging).

        θ_target = τ * θ_online + (1-τ) * θ_target
        """
        for online_param, target_param in zip(
            self.online.parameters(),
            self.target.parameters(),
            strict=True,
        ):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        action_history: torch.Tensor | None = None,
        next_action_history: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Double DQN loss with Huber loss for robustness.

        Double DQN target:
            y = r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)

        Args:
            states: Current states (N, C, H, W)
            actions: Actions taken (N,)
            rewards: Rewards received (N,)
            next_states: Next states (N, C, H, W)
            dones: Episode done flags (N,)
            gamma: Discount factor
            action_history: Optional action history for current states (N, history_len, num_actions)
            next_action_history: Optional action history for next states

        Returns:
            loss: Huber loss
            info: Dict with metrics
        """
        states.shape[0]

        # Current Q-values for taken actions
        current_q = self.online(states, action_history)  # (N, num_actions)
        current_q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (N,)

        # Double DQN target
        with torch.no_grad():
            # Select best actions using online network
            next_q_online = self.online(next_states, next_action_history)  # (N, num_actions)
            best_actions = next_q_online.argmax(dim=1)  # (N,)

            # Evaluate using target network
            next_q_target = self.target(next_states, next_action_history)  # (N, num_actions)
            next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # (N,)

            # TD target
            target_q = rewards + gamma * next_q_selected * (1.0 - dones)

        # Huber loss (more robust to outliers than MSE)
        loss = torch.nn.functional.huber_loss(current_q_selected, target_q, delta=1.0)

        # Compute TD errors for prioritized replay (absolute error)
        td_errors = (current_q_selected - target_q).abs().detach()

        # Metrics for logging
        info = {
            "loss": loss.item(),
            "q_mean": current_q_selected.mean().item(),
            "q_max": current_q_selected.max().item(),
            "td_error_mean": td_errors.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

        return loss, info
