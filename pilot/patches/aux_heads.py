"""Auxiliary heads for the dual-pathway tactile method (Pilot).

RegionHead (Path-1): position-blind contact-region reconstruction.
  - 64 learnable queries cross-attend to prefix (image+language) tokens,
    predict the SigLIP latent of the contact-region crop (49x768).
  - Trained with normalized MSE + InfoNCE; weight=0 frames contribute nothing.

ForceHead (Path-2 anti-starvation): predicts the CHANGE in fingertip pressure
  over the next decision cycle from action-expert features after state injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionHead(nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int = 768, n_latent: int = 49,
                 n_queries: int = 49, depth: int = 3, hidden: int = 512, n_heads: int = 8):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden) * 0.02)
        self.query_proj = nn.Linear(hidden, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(hidden, latent_dim)

    def forward(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        """cond_tokens: (B, N, cond_dim) fused prefix tokens. Returns (B, n_queries, latent_dim)."""
        b = cond_tokens.shape[0]
        q = self.query_proj(self.queries).expand(b, -1, -1)
        cond = self.cond_proj(cond_tokens)
        for blk in self.blocks.layers:               # per-layer: self-attn then cross-read memory
            q = blk(q)                               # queries self-attend
            combined = torch.cat([q, cond], dim=1)   # append memory tokens, let self-attn mix
            combined = blk(combined)
            q = combined[:, : q.shape[1]]            # keep query slots
        return self.out(q)


def region_loss(latent_pred: torch.Tensor, latent_gt: torch.Tensor,
                weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Normalized MSE (per-dim standardized by GT stats within batch) + InfoNCE, weighted."""
    w = weight.view(-1)                                   # (B,)
    mask = w > 0
    if mask.sum() < 2:
        zero = latent_pred.sum() * 0.0
        return zero, {"region_mse": 0.0, "region_nce": 0.0, "region_n": int(mask.sum())}

    pred, gt = latent_pred[mask], latent_gt[mask]
    # normalize per-dim with batch stats (makes MSE scale-free)
    mu, sd = gt.mean(dim=0, keepdim=True), gt.std(dim=0, keepdim=True).clamp(min=1e-3)
    mse = F.mse_loss(pred, gt, reduction="none").mean(dim=(1, 2))          # (Bk,)

    # InfoNCE: sample-level discrimination against other GTs in batch
    p = pred.flatten(1)
    g = gt.flatten(1)
    logits = p @ g.T / (p.norm(dim=1, keepdim=True) * g.norm(dim=1, keepdim=True) + 1e-6).sqrt()
    labels = torch.arange(p.shape[0], device=p.device)
    nce = F.cross_entropy(logits, labels)

    loss = (mse + 0.1 * nce) * w[mask]
    return loss.mean(), {
        "region_mse": float(mse.mean()), "region_nce": float(nce),
        "region_n": int(mask.sum()),
    }


class ForceHead(nn.Module):
    """Predicts dF (next-cycle force change) from action-expert features."""

    def __init__(self, feat_dim: int, n_out: int = 6, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def force_loss(df_pred: torch.Tensor, df_gt: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """df_pred: (B, n_out); df_gt: (B, n_out). Plain MSE on the change quantity."""
    mse = F.mse_loss(df_pred, df_gt, reduction="none").mean(dim=1)         # (B,)
    return mse.mean(), {"force_mse": float(mse.mean())}
