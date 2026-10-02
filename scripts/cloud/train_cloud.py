import argparse
import logging
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HCSCloudTrainer")


def parse_args():
    parser = argparse.ArgumentParser(description="Real Cloud Training for HCSCoder 9B")
    parser.add_argument("--base_model", type=str, default="wangzhang/Qwen3.5-9B-abliterated")
    parser.add_argument("--dataset_repo", type=str, default="timfromhcs/HCSCoder-9B-Training-Data")
    parser.add_argument("--output_dir", type=str, default="./output_model")
    parser.add_argument("--hub_model_id", type=str, default="timfromhcs/HCSCoder-Qwen3.5-9B-SFT")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    logger.info(f"Starting Real Cloud SFT Run for {args.base_model}...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset from private Hub repo
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

    # Model in BF16
    logger.info(f"Loading base model {args.base_model} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        hub_token=token,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
    )

    logger.info("Executing training...")
    trainer.train()

    logger.info(f"Saving final adapter and pushing to Hub {args.hub_model_id}...")
    trainer.save_model(args.output_dir)
    trainer.push_to_hub()
    logger.info("Cloud SFT training run successfully completed!")


if __name__ == "__main__":
    main()
