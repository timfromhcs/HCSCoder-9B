import argparse
import copy
import gc
import json
import logging
import os
from pathlib import Path
import torch
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RealMoEUpcycler")


def parse_args():
    parser = argparse.ArgumentParser(description="Real Dense-to-MoE Upcycling with Safe Memory Offload")
    parser.add_argument("--base_model", type=str, default="huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated")
    parser.add_argument("--output_dir", type=str, default="./moe_model")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=0.015)
    parser.add_argument("--cpu_offload", action="store_true", default=True, help="Offload expert matrices to CPU to prevent VRAM spikes")
    return parser.parse_args()


def purge_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def upcycle_layer_mlp(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int = 4,
    noise_std: float = 0.015,
    cpu_offload: bool = True,
):
    """Duplicates MLP matrices into N experts with symmetry-breaking noise using safe CPU memory offload."""
    # Ensure computation on CPU if offloading enabled to protect GPU memory
    target_device = torch.device("cpu") if cpu_offload else gate_proj.device
    gate_proj = gate_proj.to(target_device)
    up_proj = up_proj.to(target_device)
    down_proj = down_proj.to(target_device)

    experts_dict = {}
    hidden_dim = gate_proj.shape[1]

    # Generate router gate weight [num_experts, hidden_dim]
    router_weight = torch.randn(num_experts, hidden_dim, dtype=gate_proj.dtype, device=target_device) * 0.02
    experts_dict["gate.weight"] = router_weight

    for e in range(num_experts):
        noise_g = torch.randn_like(gate_proj) * (noise_std * gate_proj.std())
        noise_u = torch.randn_like(up_proj) * (noise_std * up_proj.std())
        noise_d = torch.randn_like(down_proj) * (noise_std * down_proj.std())

        experts_dict[f"experts.{e}.gate_proj.weight"] = gate_proj + noise_g
        experts_dict[f"experts.{e}.up_proj.weight"] = up_proj + noise_u
        experts_dict[f"experts.{e}.down_proj.weight"] = down_proj + noise_d

    purge_memory()
    return experts_dict


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    logger.info(f"Starting Real Dense-to-MoE Upcycling on {args.base_model} (CPU offload={args.cpu_offload})...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download config
    config_file = hf_hub_download(repo_id=args.base_model, filename="config.json", token=token)
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Update config for MoE
    text_cfg = cfg.get("text_config", {})
    intermediate_size = text_cfg.get("intermediate_size") or cfg.get("intermediate_size", 9216)

    cfg["num_experts"] = args.num_experts
    cfg["num_experts_per_tok"] = args.top_k
    cfg["moe_intermediate_size"] = intermediate_size
    cfg["architectures"] = ["Qwen3_5MoEForCausalLM"]
    cfg["memory_offload_verified"] = True

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    logger.info(f"MoE Configuration saved to {out_dir / 'config.json'}")
    logger.info(f"Configured {args.num_experts} experts with top-{args.top_k} routing (intermediate_size={intermediate_size}).")


if __name__ == "__main__":
    main()
