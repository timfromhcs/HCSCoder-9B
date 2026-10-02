import gc
import logging
import os
import psutil
from typing import Any, Callable, Dict, Optional, Tuple
import torch

logger = logging.getLogger("SafeMemoryGuardian")


def setup_cuda_allocator() -> None:
    """Configures PyTorch CUDA memory allocator to eliminate fragmentation and OOM spikes."""
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    flags = [
        "expandable_segments:True",
        "garbage_collection_threshold:0.8",
        "max_split_size_mb:128",
    ]
    for flag in flags:
        key = flag.split(":")[0]
        if key not in alloc_conf:
            alloc_conf = f"{alloc_conf},{flag}".strip(",")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    logger.info(f"Configured PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def purge_memory() -> None:
    """Performs aggressive, synchronized garbage collection and CUDA VRAM cache purging."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_memory_status() -> Dict[str, Any]:
    """Returns real-time GPU VRAM and CPU RAM statistics."""
    vm = psutil.virtual_memory()
    status = {
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "ram_percent": vm.percent,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        status.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            "vram_max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 2),
            "vram_total_device_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        })
    return status


def get_safe_bnb_config(
    load_in_4bit: bool = True,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
    cpu_offload: bool = True,
) -> Any:
    """Creates a memory-safe BitsAndBytesConfig optimized for Colab Free (T4 16GB) and low-memory environments."""
    try:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
        )
    except ImportError:
        logger.warning("bitsandbytes or transformers not available for quantization config.")
        return None


def is_oom_error(exc: Exception) -> bool:
    """Detects if an exception represents a CUDA or system out-of-memory error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "cuda oom" in msg


def safe_memory_execute(
    func: Callable,
    *args,
    max_retries: int = 3,
    fallback_cleanup: Optional[Callable[[], None]] = None,
    **kwargs,
) -> Any:
    """Executes a callable with automatic self-healing memory recovery on OOM."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if is_oom_error(e):
                logger.warning(f"OOM caught on attempt {attempt}/{max_retries}: {e}")
                purge_memory()
                if fallback_cleanup:
                    try:
                        fallback_cleanup()
                    except Exception as clean_err:
                        logger.error(f"Fallback cleanup error: {clean_err}")
                if attempt == max_retries:
                    logger.error(f"Self-healing memory recovery failed after {max_retries} attempts.")
                    raise
                logger.info(f"VRAM purged. Retrying operation (attempt {attempt + 1}/{max_retries})...")
            else:
                raise


class SafeMemoryManager:
    """Context and lifecycle manager for safe self-healing memory offload."""

    def __init__(self, offload_dir: str = "./offload", max_gpu_mem: str = "12GiB", max_cpu_mem: str = "30GiB"):
        self.offload_dir = offload_dir
        self.max_gpu_mem = max_gpu_mem
        self.max_cpu_mem = max_cpu_mem
        setup_cuda_allocator()
        os.makedirs(self.offload_dir, exist_ok=True)

    def get_max_memory_map(self) -> Dict[Any, str]:
        """Calculates device max memory mapping for Hugging Face Accelerate offloading."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            mem_map = {i: self.max_gpu_mem for i in range(num_gpus)}
            mem_map["cpu"] = self.max_cpu_mem
            return mem_map
        return {"cpu": self.max_cpu_mem}

    def safe_load_causal_lm(
        self,
        model_id: str,
        token: Optional[str] = None,
        use_4bit: bool = True,
        trust_remote_code: bool = True,
    ):
        """Loads a causal LM with self-healing 4-bit quantization and CPU memory offload."""
        from transformers import AutoModelForCausalLM

        purge_memory()
        bnb_config = get_safe_bnb_config(load_in_4bit=use_4bit, cpu_offload=True) if use_4bit else None
        max_mem = self.get_max_memory_map()

        try:
            logger.info(f"Loading {model_id} with 4-bit QLoRA and CPU offload ({self.offload_dir})...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_mem,
                offload_folder=self.offload_dir,
                offload_state_dict=True,
                torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
                trust_remote_code=trust_remote_code,
                token=token,
            )
            return model
        except Exception as e:
            if is_oom_error(e):
                logger.warning(f"OOM during primary model load: {e}. Activating aggressive offload fallback...")
                purge_memory()
                # Fallback: force cpu offload for non-essential layers
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    offload_folder=self.offload_dir,
                    offload_state_dict=True,
                    low_cpu_mem_usage=True,
                    trust_remote_code=trust_remote_code,
                    token=token,
                )
                return model
            raise

    def safe_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """Executes LLM inference with self-healing OOM recovery and memory purge."""
        purge_memory()

        def _gen(tok_len: int):
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=tok_len,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            generated = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated, skip_special_tokens=True)

        try:
            return _gen(max_new_tokens)
        except Exception as e:
            if is_oom_error(e):
                logger.warning(f"Inference OOM: {e}. Purging cache and retrying with reduced token length...")
                purge_memory()
                # Halve token length on retry
                fallback_tokens = max(64, max_new_tokens // 2)
                return _gen(fallback_tokens)
            raise
