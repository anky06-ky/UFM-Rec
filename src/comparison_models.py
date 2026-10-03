"""Additional adapted baselines for the project's temporal candidate protocol.

BERT4Rec uses bidirectional Cloze masking (Sun et al., arXiv:1904.06690),
with sampled softmax for this 767k-item catalog. It is not an exact reproduction.
Concat uses the same frozen CLIP tables as UFM, without uncertainty or alignment.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BERT4Rec(nn.Module):
    def __init__(self, items, history, dim=128, heads=4, layers=2, dropout=.1):
        super().__init__()
        self.history, self.mask_id = history, items + 1
        self.items = nn.Embedding(items + 2, dim, padding_idx=0)
        self.position = nn.Embedding(history, dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        nn.init.normal_(self.items.weight, std=.02)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def encode_tokens(self, tokens):
        occupied = tokens.ne(0)
        padding = ~occupied
        padding = padding.clone()
        padding[:, 0] = False
        vectors = (self.items(tokens) + self.position(torch.arange(
            self.history, device=tokens.device))) * occupied.unsqueeze(-1)
        return self.head(self.encoder(vectors, src_key_padding_mask=padding))

    def logits(self, history, candidates):
        tokens = torch.zeros_like(history)
        positions = []
        for j, row in enumerate(history):
            prior = row[row != 0][-max(0, self.history - 1):] if self.history > 1 else row[:0]
            tokens[j, :len(prior)] = prior
            tokens[j, len(prior)] = self.mask_id
            positions.append(len(prior))
        hidden = self.encode_tokens(tokens)[torch.arange(len(tokens), device=tokens.device), positions]
        return (hidden[:, None] * self.items(candidates)).sum(-1)


def cloze_batch(histories, targets, mask_id, eligible, rng):
    """Construct train-only sequences and mask 15%; 10% use last-item prediction."""
    tokens = np.zeros_like(histories)
    rows, columns, labels = [], [], []
    for row, history in enumerate(histories):
        seq = np.concatenate((history[history != 0], [targets[row]]))[-histories.shape[1]:]
        tokens[row, :len(seq)] = seq
        force_last = rng.random() < .1
        selected = np.array([len(seq)-1]) if force_last else np.flatnonzero(rng.random(len(seq)) < .15)
        if not len(selected):
            selected = np.array([rng.integers(len(seq))])
        for column in selected:
            rows.append(row); columns.append(column); labels.append(seq[column])
            draw = 0. if force_last else rng.random()
            if draw < .8:
                tokens[row, column] = mask_id
            elif draw < .9:
                tokens[row, column] = rng.choice(eligible)
    return tokens, np.asarray(rows), np.asarray(columns), np.asarray(labels)


class ConcatHybrid(nn.Module):
    can_score_unseen = True

    def __init__(self, sequential, tables, counts, dim):
        super().__init__()
        self.sequential = sequential
        self.register_buffer('counts', torch.as_tensor(np.array(counts), dtype=torch.long), persistent=False)
        for name, table in zip(('text', 'image', 'modalities'), tables):
            self.register_buffer(name, torch.as_tensor(np.array(table)), persistent=False)
        self.adapter = nn.Sequential(nn.Linear(1026, dim*2), nn.GELU(), nn.Linear(dim*2, dim), nn.LayerNorm(dim))
        self.score = nn.Sequential(nn.Linear(dim*6+2, dim), nn.GELU(), nn.Linear(dim, 1))

    def semantic(self, ids):
        mask = self.modalities[ids] & ids.ne(0).unsqueeze(-1)
        feature = torch.cat((self.text[ids].float()*mask[..., :1],
                             self.image[ids].float()*mask[..., 1:], mask.float()), -1)
        return F.normalize(self.adapter(feature), dim=-1)*mask.any(-1).unsqueeze(-1), mask

    def logits(self, history, candidates):
        known_h = history.masked_fill(self.counts[history] == 0, 0)
        cf_user = self.sequential.encode(known_h)
        known_c = self.counts[candidates] > 0
        cf_item = self.sequential.items(candidates.masked_fill(~known_c, 0))*known_c.unsqueeze(-1)
        sem_h, h_mask = self.semantic(history)
        sem_user = F.normalize(sem_h.sum(1)/h_mask.any(-1).sum(1).clamp_min(1).unsqueeze(-1), dim=-1)
        sem_item, masks = self.semantic(candidates)
        cf_user = cf_user[:, None].expand_as(cf_item)
        sem_user = sem_user[:, None].expand_as(sem_item)
        scores = self.score(torch.cat((cf_user, cf_item, cf_user*cf_item,
                                      sem_user, sem_item, sem_user*sem_item, masks.float()), -1)).squeeze(-1)
        no_signal = ~((known_h.ne(0).any(1)[:, None] & known_c) | masks.any(-1))
        return torch.where(no_signal, self.counts[candidates].float().log1p(), scores)
