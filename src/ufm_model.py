"""Research implementation of Proposal.pdf equations (5), (8)-(15).

This is a model core, NOT a trained model or an official SASRec reproduction.
Foundation embeddings are extracted with a frozen, revision-pinned encoder;
only the adapter and recommendation modules are updated using train examples.
All item IDs are one-based, with zero reserved for right-side padding.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.nn import functional as F


class FrozenCLIPEncoder(nn.Module):
    """Optional real text/image encoder; importing this module needs only torch.

    Use transformers 4.57.x. A Hub commit is mandatory to make caches reproducible.
    The CLIP text window is 77 tokens; long product descriptions are truncated.
    """

    def __init__(self, revision: str, device: str = "cpu", cache_dir=None):
        super().__init__()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pin CLIP to a 40-character Hub commit, not main.")
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", revision=revision, trust_remote_code=False,
            cache_dir=cache_dir,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", revision=revision, trust_remote_code=False,
            cache_dir=cache_dir, use_fast=False,
        )
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # A caller's model.train() must never activate the frozen encoder.
        return super().train(False)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts, padding=True, truncation=True,
            max_length=self.model.config.text_config.max_position_embeddings,
            return_tensors="pt",
        ).to(self.model.device)
        return F.normalize(self.model.get_text_features(**inputs).float(), dim=-1)

    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        return F.normalize(self.model.get_image_features(**inputs).float(), dim=-1)


@dataclass(frozen=True)
class UFMConfig:
    items: int
    feature_dim: int = 512
    dim: int = 128
    history_size: int = 20
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    id_dropout: float = 0.5
    variant: str = "full"
    fixed_alpha: float = 0.5

    def __post_init__(self):
        if min(self.items, self.feature_dim, self.dim, self.history_size, self.heads, self.layers) < 1:
            raise ValueError("All sizes must be positive.")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads.")
        if not 0 <= self.dropout < 1 or not 0 <= self.id_dropout <= 1:
            raise ValueError("Invalid dropout.")
        if not 0 <= self.fixed_alpha <= 1:
            raise ValueError("fixed_alpha must be in [0,1].")
        if self.variant not in {
            "full", "no_uncertainty", "fixed_fusion", "no_cross_align",
            "semantic_only", "collaborative_only", "text_only", "image_only",
        }:
            raise ValueError("Unknown ablation variant.")


def mlp(input_dim: int, dim: int, output_dim: int | None = None):
    return nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(),
                         nn.Linear(dim, output_dim if output_dim is not None else dim))


def reliability_weights(uncertainty: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Stable exp(-u)/sum(exp(-u)), in [CF,semantic] order (eq. 10-11)."""
    if uncertainty.shape != available.shape or uncertainty.shape[-1] != 2:
        raise ValueError("Expected matching [...,2] uncertainty and availability.")
    logits = (-uncertainty.float()).masked_fill(~available, -torch.inf)
    empty = ~available.any(-1, keepdim=True)
    weights = torch.softmax(logits.masked_fill(empty, 0), dim=-1)
    return weights.masked_fill(~available, 0)


class UFMRec(nn.Module):
    """Candidate-conditioned semantic/causal-sequential fusion with two heads.

    Foundation tables are supplied at call time, not learned parameters. Train
    counts are computed from train only; do not pass metadata rating_number.
    Uncertainty is a learned reliability proxy, not a Bayesian posterior.
    """

    def __init__(self, config: UFMConfig):
        super().__init__()
        self.config = config
        d = config.dim
        self.adapter = nn.Sequential(
            nn.Linear(2 * config.feature_dim + 2, d * 2), nn.GELU(),
            nn.Linear(d * 2, d), nn.LayerNorm(d),
        )
        self.item_ids = nn.Embedding(config.items + 1, d, padding_idx=0)
        nn.init.normal_(self.item_ids.weight, std=0.02)
        with torch.no_grad():
            self.item_ids.weight[0].zero_()
        self.position = nn.Embedding(config.history_size, d)
        layer = nn.TransformerEncoderLayer(
            d, config.heads, dim_feedforward=d * 4, dropout=config.dropout,
            batch_first=True, norm_first=True,
        )
        self.sequential = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.cf_pair = mlp(3 * d + 2, d)
        self.sem_pair = mlp(3 * d + 2, d)
        self.cf_uncertainty = mlp(d, d, 1)
        self.sem_uncertainty = mlp(d, d, 1)
        self.learned_gate = mlp(2 * d, d, 2)
        self.align_gate = mlp(2 * d, d)
        self.align_projection = nn.Linear(2 * d, d, bias=False)
        self.recommendation_head = mlp(d, d, 1)

    def semantic_items(self, ids, text, image, modalities):
        if text.shape != image.shape or text.shape != (self.config.items + 1, self.config.feature_dim):
            raise ValueError("Feature tables must have shape [items+1,feature_dim].")
        if modalities.dtype != torch.bool or modalities.shape != (self.config.items + 1, 2):
            raise ValueError("modalities must be boolean [items+1,2], text then image.")
        mask = modalities[ids].clone()
        mask &= ids.ne(0).unsqueeze(-1)
        if self.config.variant == "text_only":
            mask[..., 1] = False
        if self.config.variant == "image_only":
            mask[..., 0] = False
        # Missing modalities are zeroed before the adapter, not represented by noise.
        features = torch.cat([
            text[ids].detach().float().masked_fill(~mask[..., :1], 0),
            image[ids].detach().float().masked_fill(~mask[..., 1:], 0),
            mask.float(),
        ], dim=-1)
        present = mask.any(-1)
        vectors = F.normalize(self.adapter(features).float(), dim=-1)
        return vectors * present.unsqueeze(-1), present

    def collaborative_items(self, ids, counts):
        known = ids.ne(0) & counts[ids].gt(0)
        if self.training and self.config.id_dropout:
            known &= torch.rand_like(known.float()).ge(self.config.id_dropout)
        # Unknown IDs map to the padding row BEFORE lookup, so their weights and
        # gradients cannot leak into a prediction even when they contain NaN.
        safe_ids = ids.masked_fill(~known, 0)
        vectors = self.item_ids(safe_ids) * known.unsqueeze(-1)
        return vectors, known

    def encode_cf_history(self, vectors, ids, known):
        occupied = ids.ne(0)
        valid = occupied & known
        # Preserve positions of unknown but observed past items. Mask pads only.
        padding = ~occupied
        safe_padding = padding.clone()
        safe_padding[:, 0] = False
        x = vectors + self.position(torch.arange(ids.shape[1], device=ids.device))
        causal = torch.ones((ids.shape[1], ids.shape[1]), device=ids.device, dtype=torch.bool).triu(1)
        x = self.sequential(x, mask=causal, src_key_padding_mask=safe_padding)
        lengths = occupied.sum(-1)
        last = lengths.clamp_min(1) - 1
        user = x[torch.arange(len(ids), device=ids.device), last]
        available = valid.any(-1)
        return F.normalize(user.float(), dim=-1) * available.unsqueeze(-1), available

    def forward(self, histories, candidates, text, image, modalities, train_counts):
        if histories.ndim != 2 or candidates.ndim != 2 or len(histories) != len(candidates):
            raise ValueError("Expected histories [B,H] and candidates [B,K].")
        if histories.shape[1] != self.config.history_size or len(histories) == 0 or candidates.shape[1] == 0:
            raise ValueError("Wrong history length or empty batch/candidate set.")
        if histories.dtype != torch.long or candidates.dtype != torch.long:
            raise ValueError("Item IDs must be torch.long.")
        if (histories < 0).any() or (histories > self.config.items).any() or (candidates < 1).any() or (candidates > self.config.items).any():
            raise ValueError("Item ID out of range; candidates cannot contain padding.")
        if (histories[:, 1:].ne(0) & histories[:, :-1].eq(0)).any():
            raise ValueError("Histories must be right-padded.")
        if train_counts.dtype != torch.long or train_counts.shape != (self.config.items + 1,) or train_counts[0] != 0 or (train_counts < 0).any():
            raise ValueError("Expected non-negative train-only counts and a zero padding row.")
        hs, hs_present = self.semantic_items(histories, text, image, modalities)
        cs, cs_present = self.semantic_items(candidates, text, image, modalities)
        sem_user = hs.sum(1) / hs_present.sum(1, keepdim=True).clamp_min(1)
        sem_user = F.normalize(sem_user, dim=-1)[:, None].expand_as(cs)
        hc, hc_present = self.collaborative_items(histories, train_counts)
        cc, cc_present = self.collaborative_items(candidates, train_counts)
        cf_user, cf_available = self.encode_cf_history(hc, histories, hc_present)
        cf_user = cf_user[:, None].expand_as(cc)
        lengths = histories.ne(0).sum(-1).float().log1p()[:, None, None].expand(len(histories), candidates.shape[1], 1)
        frequencies = train_counts[candidates].float().log1p().unsqueeze(-1)
        cf = self.cf_pair(torch.cat([cf_user, cc, cf_user * cc, lengths, frequencies], -1))
        modality_flags = modalities[candidates].clone()
        if self.config.variant == "text_only":
            modality_flags[..., 1] = False
        if self.config.variant == "image_only":
            modality_flags[..., 0] = False
        sem = self.sem_pair(torch.cat([sem_user, cs, sem_user * cs, modality_flags.float()], -1))
        cf_ok = cf_available[:, None] & cc_present
        sem_ok = hs_present.any(-1)[:, None] & cs_present
        if self.config.variant == "semantic_only":
            cf_ok = torch.zeros_like(cf_ok)
        if self.config.variant == "collaborative_only":
            sem_ok = torch.zeros_like(sem_ok)
        available = torch.stack([cf_ok, sem_ok], -1)
        cf = cf * cf_ok.unsqueeze(-1)
        sem = sem * sem_ok.unsqueeze(-1)
        u_cf = F.softplus(self.cf_uncertainty(cf).float()).squeeze(-1)
        u_sem = F.softplus(self.sem_uncertainty(sem).float()).squeeze(-1)
        uncertainty = torch.stack([u_cf, u_sem], -1)
        pair = torch.cat([cf, sem], -1)
        variant = self.config.variant
        if variant == "no_uncertainty":
            gate_logits = self.learned_gate(pair).float().masked_fill(~available, -torch.inf)
            gate_logits = gate_logits.masked_fill(~available.any(-1, keepdim=True), 0)
            weights = torch.softmax(gate_logits, -1).masked_fill(~available, 0)
        elif variant == "fixed_fusion":
            weights = torch.tensor([self.config.fixed_alpha, 1 - self.config.fixed_alpha], device=cf.device).expand_as(uncertainty)
            # If only one branch exists, use it even for alpha=0 or alpha=1.
            weights = torch.where(available.sum(-1, keepdim=True).eq(1), available.float(), weights * available)
        else:
            weights = reliability_weights(uncertainty, available)
        align = torch.sigmoid(self.align_gate(pair)) * self.align_projection(pair)
        align = align * available.all(-1, keepdim=True)
        if variant == "no_cross_align":
            align = torch.zeros_like(align)
        fused = weights[..., :1] * cf + weights[..., 1:] * sem + align
        score = self.recommendation_head(fused).squeeze(-1).float()
        # Common train-popularity fallback, never future counts, for no-signal pairs.
        score = torch.where(available.any(-1), score, train_counts[candidates].float().log1p())
        return {
            "scores": score, "weights": weights, "uncertainty": uncertainty,
            "available": available, "cf": cf, "semantic": sem,
            "cf_scores": self.recommendation_head(cf).squeeze(-1).float(),
            "semantic_scores": self.recommendation_head(sem).squeeze(-1).float(),
            "alignment": align,
            "uncertainty_enabled": variant != "no_uncertainty",
        }


def ufm_loss(output, valid_negatives, *, lambda_align=0.01, lambda_unc=0.01, lambda_cal=0.1,
             uncertainty_enabled=None):
    """Positive at column 0, remaining columns are train-only sampled negatives.

    Concrete objectives where the proposal leaves details unspecified: alignment
    is cosine consistency on available pairs; uncertainty fits each branch's
    detached sigmoid residual (a bounded reliability proxy); calibration uses
    pointwise BCE on the sampled candidates, NOT real-world purchase probability.
    """
    scores = output["scores"]
    if scores.ndim != 2 or scores.shape[1] < 2 or valid_negatives.shape != scores[:, 1:].shape or valid_negatives.dtype != torch.bool:
        raise ValueError("Expected scores [B,1+N] and boolean valid_negatives [B,N].")
    if min(lambda_align, lambda_unc, lambda_cal) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if not valid_negatives.any():
        raise ValueError("No valid negatives; skip this batch instead of fake learning.")
    ranking = F.softplus(scores[:, 1:] - scores[:, :1])[valid_negatives].mean()
    both = output["available"].all(-1)
    active = torch.cat([valid_negatives.any(-1, keepdim=True), valid_negatives], -1)
    both = both & active
    alignment = (1 - F.cosine_similarity(output["cf"], output["semantic"], dim=-1))[both].mean() if both.any() else scores.sum() * 0
    labels = torch.zeros_like(scores)
    labels[:, 0] = 1
    calibration = F.binary_cross_entropy_with_logits(scores[active], labels[active])
    uncertainty = scores.sum() * 0
    if uncertainty_enabled is None:
        uncertainty_enabled = output["uncertainty_enabled"]
    if uncertainty_enabled:
        branch_scores = torch.stack([output["cf_scores"], output["semantic_scores"]], -1)
        residual = (branch_scores.sigmoid().detach() - labels.unsqueeze(-1)).square()
        allowed = output["available"] & active.unsqueeze(-1)
        if allowed.any():
            # Bounded by the observed sampled residual; not a free confidence head.
            uncertainty = F.mse_loss(output["uncertainty"][allowed], residual[allowed])
    total = ranking + lambda_align * alignment + lambda_unc * uncertainty + lambda_cal * calibration
    return {"total": total, "bpr": ranking, "align": alignment, "unc": uncertainty, "cal": calibration}


@torch.no_grad()
def candidate_calibration(scores, positive_indices=None, *, temperature=1.0, bins=15):
    """Top-1 ECE/NLL/Brier conditional on a fixed candidate set, not catalog-wide."""
    if scores.ndim != 2 or min(scores.shape) < 1 or bins < 1 or temperature <= 0 or not torch.isfinite(scores).all():
        raise ValueError("Finite non-empty scores, positive temperature and bins required.")
    if positive_indices is None:
        positive_indices = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
    if positive_indices.shape != (len(scores),) or positive_indices.dtype != torch.long or (positive_indices < 0).any() or (positive_indices >= scores.shape[1]).any():
        raise ValueError("Invalid positive indices.")
    log_probs = F.log_softmax(scores.double() / temperature, -1)
    probs = log_probs.exp()
    confidence = probs.max(-1).values
    # Positive is often column zero: argmax's first-index tie rule would then
    # spuriously make a uniform predictor 100% correct. Use expected correctness
    # under uniform tie-breaking, independent of positive-column placement.
    tied = probs.eq(confidence.unsqueeze(-1))
    correct = tied.gather(1, positive_indices[:, None]).squeeze(-1).double() / tied.sum(-1)
    assignments = (confidence * bins).long().clamp_max(bins - 1)
    ece = 0.0
    reliability = []
    for i in range(bins):
        mask = assignments.eq(i)
        n = int(mask.sum())
        accuracy = float(correct[mask].mean()) if n else None
        mean_conf = float(confidence[mask].mean()) if n else None
        if n:
            ece += n / len(scores) * abs(accuracy - mean_conf)
        reliability.append({"bin": i, "n": n, "accuracy": accuracy, "confidence": mean_conf})
    labels = F.one_hot(positive_indices, num_classes=scores.shape[1]).double()
    return {"scope": "top1_given_sampled_candidate_set", "n": len(scores),
            "tie_policy": "uniform_among_equal_maxima",
            "ece": ece, "nll": float(F.nll_loss(log_probs, positive_indices)),
            "brier_multiclass": float((probs - labels).square().sum(-1).mean()),
            "temperature": temperature, "bins": reliability}
