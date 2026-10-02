import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TopKRouter(nn.Module):
    """Top-K Router with load-balancing auxiliary loss computation."""

    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, hidden_dim) or (total_tokens, hidden_dim)
        logits = self.gate(x)  # (..., num_experts)
        scores = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        # Normalize weights so sum over top-k is 1.0
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Auxiliary load balancing loss (Switch Transformer / GShard formulation)
        # aux_loss = num_experts * sum_i(f_i * P_i)
        # f_i: fraction of tokens dispatched to expert i
        # P_i: average probability assigned to expert i
        flat_scores = scores.view(-1, self.num_experts)
        flat_indices = top_k_indices.view(-1, self.top_k)

        # Compute fraction dispatched
        one_hot_dispatched = F.one_hot(flat_indices, num_classes=self.num_experts).sum(dim=1).float()
        f_i = one_hot_dispatched.mean(dim=0)
        p_i = flat_scores.mean(dim=0)

        aux_loss = self.num_experts * torch.sum(f_i * p_i)

        return top_k_weights, top_k_indices, aux_loss


class MoEBlock(nn.Module):
    """Sparse MoE block containing top-k router and multiple expert FFNs."""

    def __init__(self, experts: nn.ModuleList, router: TopKRouter):
        super().__init__()
        self.experts = experts
        self.router = router
        self.num_experts = len(experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        flat_x = x.view(-1, orig_shape[-1])  # (N, D)
        weights, indices, aux_loss = self.router(flat_x)  # weights: (N, K), indices: (N, K)

        output = torch.zeros_like(flat_x)

        for expert_idx, expert_layer in enumerate(self.experts):
            # Mask for tokens where expert_idx was selected in top_k
            for k in range(self.router.top_k):
                mask = indices[:, k] == expert_idx
                if mask.any():
                    selected_x = flat_x[mask]
                    expert_out = expert_layer(selected_x)
                    w = weights[mask, k].unsqueeze(-1)
                    output[mask] += w * expert_out

        return output.view(orig_shape), aux_loss
