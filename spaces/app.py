import gc
import logging
import os
import spaces  # ZeroGPU dynamic GPU allocation (must precede torch)
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HCSSpacesApp")

# 1. Eliminate memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

MODEL_ID = "huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated"
OFFLOAD_DIR = "./offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

tokenizer = None
model = None


def purge_memory():
    """Forces garbage collection and CUDA cache release."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_model_if_needed():
    global tokenizer, model
    if tokenizer is None:
        logger.info(f"Loading tokenizer for {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        purge_memory()
        logger.info(f"Loading model {MODEL_ID} with 4-bit QLoRA and CPU offload...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        max_mem = {0: "12GiB", "cpu": "24GiB"} if torch.cuda.is_available() else {"cpu": "24GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_mem,
            offload_folder=OFFLOAD_DIR,
            offload_state_dict=True,
            trust_remote_code=True,
        )


@spaces.GPU(duration=120)
def generate_agent_response(prompt: str, system_prompt: str, max_new_tokens: int = 512, temperature: float = 0.2):
    """Executes agent turn on Hugging Face ZeroGPU with Safe Self-Healing Memory and CPU Offloading."""
    load_model_if_needed()

    messages = [
        {"role": "system", "content": system_prompt or "You are HCSCoder, an autonomous coding agent."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # Safe Self-Healing Inference Execution
    def _execute_gen(token_limit: int):
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=token_limit,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    try:
        return _execute_gen(max_new_tokens)
    except Exception as e:
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        if is_oom:
            logger.warning(f"[SELF-HEALING MEMORY] Inference OOM: {e}. Purging VRAM cache and retrying with reduced token budget...")
            purge_memory()
            fallback_tokens = max(64, max_new_tokens // 2)
            try:
                return _execute_gen(fallback_tokens)
            except Exception as e2:
                purge_memory()
                return f"Error: Inference memory limit reached ({e2}). VRAM safely purged."
        raise e


demo = gr.Interface(
    fn=generate_agent_response,
    inputs=[
        gr.Textbox(lines=5, label="User Coding / Benchmark Prompt", placeholder="Fix the failing unit test..."),
        gr.Textbox(lines=2, label="System Prompt", value="You are HCSCoder, an autonomous coding agent. Inspect before modifying and verify evidence."),
        gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Max New Tokens"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature"),
    ],
    outputs=gr.Textbox(lines=10, label="HCSCoder Response / Tool Call"),
    title="HCSCoder 4B — ZeroGPU Evaluation & Inference",
    description="Running on Hugging Face ZeroGPU (Nvidia A100 compute pool, $0 cost for Pro users) with Safe Self-Healing Memory and CPU Offload.",
)

if __name__ == "__main__":
    demo.launch()
