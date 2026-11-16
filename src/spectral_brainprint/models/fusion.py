"""Task-conditioned fusion from the Brainprint_ML team handoff.

Module names and parameter layout match the historical FusionIdentityNet.
Callers provide the normalized feature tensors, availability masks, and task IDs.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class BranchEncoder(nn.Module):
    """Encode one vector, sequence, grid, or unordered set of features."""

    def __init__(self, spec, width, dropout):
        super().__init__()
        self.kind = spec['kind']
        shape = spec['shape']
        self.net = nn.Sequential(
            nn.Linear(shape[-1], width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.score = nn.Linear(width, 1)
        self.position = None
        self.context = None
        if self.kind in ('sequence', 'grid'):
            self.position = nn.Parameter(torch.zeros(1, math.prod(shape[:-1]), width))
            nn.init.normal_(self.position, std=.02)
            self.context = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        if self.kind == 'vector':
            x = x[:, None, :]
            mask = mask.reshape(batch_size, 1)
        else:
            x = x.reshape(batch_size, -1, x.shape[-1])
            mask = mask.reshape(batch_size, -1)

        mask = mask.bool()
        present = mask.any(1)
        encoded = self.net(x)
        if self.position is not None:
            encoded = encoded + self.position
            # Keep one key available to attention for rows with no valid token.
            # The original mask clears those rows during pooling.
            safe = mask.clone()
            safe[~present, 0] = True
            context, _ = self.context(
                encoded, encoded, encoded, key_padding_mask=~safe, need_weights=False
            )
            encoded = encoded + context

        logits = self.score(torch.tanh(encoded)).squeeze(-1).masked_fill(~mask, -1e4)
        weights = logits.softmax(1) * mask
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = (encoded * weights[:, :, None]).sum(1)
        return pooled * present[:, None], present


class FusionIdentityNet(nn.Module):
    """Predict class cosine scores and an embedding from masked feature branches."""

    def __init__(
        self,
        specs,
        num_classes,
        width=96,
        embedding_dim=96,
        dropout=.2,
        modality_dropout=.2,
        expert_mode='soft',
    ):
        super().__init__()
        self.names = sorted(specs)
        self.modality_dropout = modality_dropout
        self.expert_mode = expert_mode
        if expert_mode not in ('soft', 'hard', 'shared'):
            raise ValueError('expert_mode must be soft, hard or shared')
        if width < 8 or width % 4:
            raise ValueError('Model width must be >=8 and divisible by 4')

        self.branches = nn.ModuleDict({
            name: BranchEncoder(specs[name], width, dropout) for name in self.names
        })
        self.task_embedding = nn.Embedding(4, 16)
        self.gates = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(width + 16, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
            for name in self.names
        })
        self.fuse = nn.Sequential(nn.Linear(width + len(self.names), width), nn.LayerNorm(width), nn.GELU())

        def expert():
            return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))

        self.shared = expert()
        self.experts = nn.ModuleList([expert() for _ in range(4)])
        self.router = nn.Linear(width + 16, 4)
        self.projection = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, embedding_dim)
        )
        self.class_centers = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.class_centers)

    def forward(self, features, masks, task):
        task_embedding = self.task_embedding(task)
        encoded, available, gate_logits = [], [], []
        for name in self.names:
            branch, present = self.branches[name](features[name], masks[name])
            encoded.append(branch)
            available.append(present)
            gate_logits.append(self.gates[name](torch.cat([branch, task_embedding], -1)).squeeze(-1))

        encoded = torch.stack(encoded, 1)
        available = torch.stack(available, 1)
        used = available.clone()
        if self.training and self.modality_dropout > 0:
            used &= torch.rand_like(used.float()) >= self.modality_dropout
            lost = ~used.any(1) & available.any(1)
            rows = lost.nonzero(as_tuple=True)[0]
            if len(rows):
                used[rows, available[rows].float().argmax(1)] = True
        if (~available.any(1)).any():
            raise ValueError('Every sample must have at least one observed feature branch')

        logits = torch.stack(gate_logits, 1).masked_fill(~used, -1e4)
        gates = logits.softmax(1) * used
        gates = gates / gates.sum(1, keepdim=True).clamp_min(1e-8)
        fused = (encoded * gates[:, :, None]).sum(1)
        fused = self.fuse(torch.cat([fused, used.float()], -1))
        shared = self.shared(fused)

        if self.expert_mode == 'shared':
            routing = torch.zeros(len(task), 4, device=task.device)
            mixed = shared
        else:
            if self.expert_mode == 'hard':
                routing = F.one_hot(task, 4).to(fused.dtype)
            else:
                router_input = torch.cat([fused, task_embedding], -1)
                routing = (self.router(router_input) + 2 * F.one_hot(task, 4)).softmax(-1)
            experts = torch.stack([expert(fused) for expert in self.experts], 1)
            mixed = shared + .5 * (experts * routing[:, :, None]).sum(1)

        embedding = F.normalize(self.projection(mixed), dim=-1)
        cosine = embedding @ F.normalize(self.class_centers, dim=-1).T
        return {'embedding': embedding, 'cosine': cosine, 'branch_weights': gates, 'expert_weights': routing}
