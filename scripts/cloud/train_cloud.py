import argparse
import gc
import logging
import os
import sys
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HCSCloudTrainer")


def setup_cuda_allocator():
    """Eliminates memory fragmentation for safe zero-OOM execution."""
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    flags = ["expandable_segments:True", "garbage_collection_threshold:0.8", "max_split_size_mb:128"]
    for flag in flags:
        k = flag.split(":")[0]
        if k not in alloc_conf:
            alloc_conf = f"{alloc_conf},{flag}".strip(",")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def purge_cuda_memory():
    """Forces aggressive garbage collection and CUDA cache flush."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Real Cloud Training for HCSCoder 4B with Safe Memory Offload")
    parser.add_argument("--base_model", type=str, default="huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated")
    parser.add_argument("--dataset_repo", type=str, default="timfromhcs/HCSCoder-9B-Training-Data")
    parser.add_argument("--output_dir", type=str, default="./output_model")
    parser.add_argument("--hub_model_id", type=str, default="timfromhcs/HCSCoder-Qwen3.5-4B-SFT")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Load base model in 4-bit QLoRA")
    parser.add_argument("--offload_dir", type=str, default="./offload", help="Directory for CPU memory offload")
    parser.add_argument("--enable_self_healing", action="store_true", default=True, help="Auto-recover from CUDA OOM")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_cuda_allocator()
    purge_cuda_memory()
    token = os.environ.get("HF_TOKEN")

    logger.info(f"Starting Real Cloud SFT Run for {args.base_model} (Self-Healing Memory + CPU Offload enabled)...")
    os.makedirs(args.offload_dir, exist_ok=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset from Hub repo
    logger.info(f"Loading dataset from {args.dataset_repo}...")
    dataset = load_dataset(args.dataset_repo, split="train", token=token)

    # Format into chat template if required
    def format_chat(sample):
        msgs = sample.get("messages", [])
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    formatted_dataset = dataset.map(format_chat)

    # LoRA PEFT Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Safe Quantization & Memory Offload Setup
    max_memory = {0: "12GiB", "cpu": "30GiB"} if torch.cuda.is_available() else {"cpu": "30GiB"}

    if args.use_4bit:
        logger.info(f"Configuring 4-bit QLoRA with CPU offloading ({args.offload_dir})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=args.offload_dir,
            offload_state_dict=True,
            trust_remote_code=True,
            token=token,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info(f"Loading base model {args.base_model} in bfloat16 with CPU offload...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=args.offload_dir,
            offload_state_dict=True,
            trust_remote_code=True,
            token=token,
        )

    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def build_training_args(bs: int, accum: int, seq_len: int) -> SFTConfig:
        return SFTConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=accum,
            num_train_epochs=args.epochs,
            max_seq_length=seq_len,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            logging_steps=10,
            save_strategy="epoch",
            push_to_hub=True,
            hub_model_id=args.hub_model_id,
            hub_token=token,
            dataset_text_field="text",
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_pin_memory=False,
        )

    # Safe Self-Healing Training Loop
    current_bs = args.batch_size
    current_accum = args.gradient_accumulation_steps
    current_seq_len = args.max_seq_length
    max_retries = 3 if args.enable_self_healing else 1

    for attempt in range(1, max_retries + 1):
        try:
            purge_cuda_memory()
            t_args = build_training_args(current_bs, current_accum, current_seq_len)
            trainer = SFTTrainer(
                model=model,
                args=t_args,
                train_dataset=formatted_dataset,
                processing_class=tokenizer,
            )

            logger.info(f"Executing training (Attempt {attempt}, batch_size={current_bs}, grad_accum={current_accum}, max_seq_length={current_seq_len})...")
            trainer.train()
            break
        except Exception as e:
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
            if is_oom and attempt < max_retries:
                logger.warning(f"[SELF-HEALING MEMORY] CUDA OOM caught on attempt {attempt}: {e}")
                purge_cuda_memory()
                current_accum = current_accum * 2
                current_seq_len = max(1024, current_seq_len // 2)
                logger.info(f"[SELF-HEALING MEMORY] Downscaled seq_len to {current_seq_len}, grad_accum to {current_accum}. Retrying...")
            else:
                logger.error(f"Fatal training error: {e}")
                raise

    logger.info(f"Saving final adapter and pushing to Hub {args.hub_model_id}...")
    trainer.save_model(args.output_dir)
    trainer.push_to_hub()
    logger.info("Cloud SFT training run successfully completed!")


if __name__ == "__main__":
    main()
