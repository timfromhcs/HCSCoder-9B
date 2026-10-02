import os
import spaces  # ZeroGPU dynamic GPU allocation (must precede torch)
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "wangzhang/Qwen3.5-9B-abliterated"

tokenizer = None
model = None


def load_model_if_needed():
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )


@spaces.GPU(duration=120)
def generate_agent_response(prompt: str, system_prompt: str, max_new_tokens: int = 512, temperature: float = 0.2):
    """Executes agent turn on Hugging Face ZeroGPU (Nvidia A100/H200) with zero dollar cost."""
    load_model_if_needed()

    messages = [
        {"role": "system", "content": system_prompt or "You are HCSCoder, an autonomous coding agent."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response


demo = gr.Interface(
    fn=generate_agent_response,
    inputs=[
        gr.Textbox(lines=5, label="User Coding / Benchmark Prompt", placeholder="Fix the failing unit test..."),
        gr.Textbox(lines=2, label="System Prompt", value="You are HCSCoder, an autonomous coding agent. Inspect before modifying and verify evidence."),
        gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Max New Tokens"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature"),
    ],
    outputs=gr.Textbox(lines=10, label="HCSCoder Response / Tool Call"),
    title="HCSCoder 9B — ZeroGPU Evaluation & Inference",
    description="Running on Hugging Face ZeroGPU (Nvidia A100 compute pool, $0 cost for Pro users).",
)

if __name__ == "__main__":
    demo.launch()
