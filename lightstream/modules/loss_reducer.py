from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class _StreamingWSLState:
    sum_p_r_terms: list[Tensor]
    valid_count: int
    seen_indices: Tensor | None


class GlobalWSLossReducer(nn.Module):
    """Weakly-supervised global reducer.

    Aggregates per-pixel sigmoid outputs using:
        y_hat = ((1 / |X|) * sum_{i,j} p(i,j)^r)^(1/r)
    and computes BCE(y_hat, slide_label).
    """

    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)

    def aggregate(self, logits: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(logits.shape)}")
        probs = torch.sigmoid(logits)
        mean_p_r = probs.pow(self.r).mean(dim=(-2, -1))
        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)

    def forward(self, logits: Tensor, slide_label: Tensor) -> Tensor:
        pooled_score = self.aggregate(logits)
        target = slide_label.to(device=pooled_score.device, dtype=pooled_score.dtype)
        return F.binary_cross_entropy(pooled_score, target)


class StreamingGlobalWSLossReducer(nn.Module):
    """Tile-wise weakly-supervised global reducer.

    The reducer accumulates sum(p^r) and valid spatial counts tile-by-tile, then
    computes the same pooled score/loss as `GlobalWSLossReducer`.

    Backward is handled by regular autograd: each `update(...)` stores a differentiable
    scalar contribution (`sum(p^r)` over valid pixels). `finalize(...)` combines these
    contributions into one scalar BCE loss, and `loss.backward()` propagates gradients
    through all tile contributions back to their originating logits.

    Optional deduplication support:
      - call `reset(spatial_shape=(H, W))`
      - pass `tile_origin=(y, x)` for each update
      - optionally pass `lost=(top, left, bottom, right)` to crop invalid tile borders
    This prevents double-counting on overlapping tiles.
    """

    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)
        self.reset()

    def reset(self, spatial_shape: tuple[int, int] | None = None) -> None:
        seen_indices = None
        if spatial_shape is not None:
            seen_indices = torch.zeros(spatial_shape, dtype=torch.bool)
        self.state = _StreamingWSLState(sum_p_r_terms=[], valid_count=0, seen_indices=seen_indices)

    def _crop_lost(self, logits: Tensor, lost: tuple[int, int, int, int] | None) -> Tensor:
        if lost is None:
            return logits
        top, left, bottom, right = lost
        h, w = logits.shape[-2:]
        y1 = max(top, 0)
        x1 = max(left, 0)
        y2 = h - max(bottom, 0)
        x2 = w - max(right, 0)
        if y1 >= y2 or x1 >= x2:
            raise ValueError("Lost-cropping removed the entire tile.")
        return logits[:, :, y1:y2, x1:x2]

    def update(
        self,
        logits: Tensor,
        tile_origin: tuple[int, int] | None = None,
        lost: tuple[int, int, int, int] | None = None,
    ) -> None:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(logits.shape)}")

        tile_logits = self._crop_lost(logits, lost)
        probs_r = torch.sigmoid(tile_logits).pow(self.r)
        tile_h, tile_w = probs_r.shape[-2:]

        if self.state.seen_indices is None:
            self.state.sum_p_r_terms.append(probs_r.sum(dim=(-2, -1)))
            self.state.valid_count += tile_h * tile_w
            return

        if tile_origin is None:
            raise ValueError("tile_origin must be provided when using deduplication (reset with spatial_shape).")

        y, x = tile_origin
        y = int(y)
        x = int(x)
        y2 = y + tile_h
        x2 = x + tile_w

        seen = self.state.seen_indices
        if y < 0 or x < 0 or y2 > seen.shape[0] or x2 > seen.shape[1]:
            raise ValueError("tile_origin and tile shape exceed reducer spatial_shape.")

        unseen_mask_2d = ~seen[y:y2, x:x2]
        seen[y:y2, x:x2] = True

        unseen_mask = unseen_mask_2d.to(device=probs_r.device, dtype=probs_r.dtype)[None, None, :, :]
        self.state.sum_p_r_terms.append((probs_r * unseen_mask).sum(dim=(-2, -1)))
        self.state.valid_count += int(unseen_mask_2d.sum().item())

    def pooled_score(self) -> Tensor:
        if self.state.valid_count == 0:
            raise ValueError("StreamingGlobalWSLossReducer received no valid pixels.")
        if not self.state.sum_p_r_terms:
            raise ValueError("StreamingGlobalWSLossReducer has no accumulated tiles. Call update() first.")

        sum_p_r = torch.stack(self.state.sum_p_r_terms, dim=0).sum(dim=0)
        mean_p_r = sum_p_r / self.state.valid_count
        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)

    def finalize(self, slide_label: Tensor) -> Tensor:
        pooled_score = self.pooled_score()
        target = slide_label.to(device=pooled_score.device, dtype=pooled_score.dtype)
        return F.binary_cross_entropy(pooled_score, target)


# Backwards-compatible aliases from the previous draft implementation.
GlobalLossReducer = GlobalWSLossReducer
StreamingGlobalLossReducer = StreamingGlobalWSLossReducer
