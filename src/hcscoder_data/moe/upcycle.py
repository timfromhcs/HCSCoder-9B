import copy
import logging
from typing import List, Tuple
import torch
import torch.nn as nn
from hcscoder_data.moe.router import MoEBlock, TopKRouter

logger = logging.getLogger("MoEUpcycler")


class MoEUpcycler:
    """Upcycles a dense FFN/MLP module into a sparse MoE module with cloned experts and safe memory offload."""

    def __init__(self, num_experts: int = 4, top_k: int = 2, offload_to_cpu: bool = True):
        self.num_experts = num_experts
        self.top_k = top_k
        self.offload_to_cpu = offload_to_cpu

    def upcycle_mlp(self, dense_mlp: nn.Module, hidden_dim: int) -> MoEBlock:
        logger.info(f"Upcycling dense MLP (dim={hidden_dim}) to {self.num_experts} experts (top_k={self.top_k}, cpu_offload={self.offload_to_cpu})...")
        experts = nn.ModuleList([copy.deepcopy(dense_mlp) for _ in range(self.num_experts)])
        if self.offload_to_cpu and not next(dense_mlp.parameters()).is_cuda:
            experts = experts.to("cpu")
        router = TopKRouter(hidden_dim=hidden_dim, num_experts=self.num_experts, top_k=self.top_k)
        return MoEBlock(experts=experts, router=router)


    def verify_upcycle(self, dense_mlp: nn.Module, moe_block: MoEBlock, hidden_dim: int) -> Tuple[bool, float]:
        """Verify that before training, MoE output with cloned experts closely matches dense output."""
        test_input = torch.randn(2, 8, hidden_dim)
        with torch.no_grad():
            dense_out = dense_mlp(test_input)
            moe_out, aux_loss = moe_block(test_input)
            # Since experts are clones and weights sum to 1, output should be identical
            diff = torch.norm(dense_out - moe_out).item()
            is_valid = diff < 1e-4
            logger.info(f"Upcycle verification diff: {diff:.6f}, aux_loss: {aux_loss.item():.4f}, valid: {is_valid}")
            return is_valid, diff
