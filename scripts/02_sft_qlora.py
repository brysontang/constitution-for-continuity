#!/usr/bin/env python3
"""
Step 2 (Stage A): QLoRA SFT on critique-revised data.

Loads Qwen2.5-3B-Instruct in 4-bit NF4, attaches LoRA adapters on all linear
layers, and fine-tunes on the revised responses from Stage A.

Runs on local GPU (RTX 3070 Ti, 8GB VRAM).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import init_wandb, load_config, set_seed


def load_revisions(path: str) -> list[dict]:
    """Load revisions from JSONL."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_for_sft(revisions: list[dict], tokenizer) -> list[str]:
    """Format revision data as chat-template strings for SFT."""
    formatted = []
    for item in revisions:
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append(text)
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Stage A: QLoRA SFT training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Load model, run 1 step, exit")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    # Lazy imports to speed up arg parsing
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    model_name = config["model"]["name"]
    sft_cfg = config["sft"]
    lora_cfg = config["lora"]

    # Init wandb
    run = init_wandb(config, run_name_suffix="sft", dry_run=args.dry_run)

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model with 4-bit quantization ──────────────────────────────────
    print(f"Loading model in 4-bit NF4: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config["model"]["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Attach LoRA ─────────────────────────────────────────────────────────
    print("Attaching LoRA adapters...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load and format data ────────────────────────────────────────────────
    print(f"Loading revisions from {config['data']['revisions_path']}")
    revisions = load_revisions(config["data"]["revisions_path"])
    print(f"Loaded {len(revisions)} revision examples")

    texts = format_for_sft(revisions, tokenizer)
    dataset = Dataset.from_dict({"text": texts})

    if args.dry_run:
        dataset = dataset.select(range(min(4, len(dataset))))
        print(f"Dry-run: using {len(dataset)} examples")
        print(f"Sample text (first 500 chars):\n{texts[0][:500]}")

    # ── Training arguments ──────────────────────────────────────────────────
    output_dir = sft_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    max_steps = 2 if args.dry_run else -1
    num_epochs = 1 if args.dry_run else sft_cfg["epochs"]

    # Detect bf16 support — NVML driver mismatch can cause false negatives
    use_bf16 = sft_cfg["bf16"]
    use_fp16 = False
    if use_bf16:
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            use_bf16 = False
        if not use_bf16:
            print("Warning: bf16 not detected, falling back to fp16")
            use_fp16 = True

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        lr_scheduler_type=sft_cfg["lr_scheduler_type"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=sft_cfg["gradient_checkpointing"],
        logging_steps=sft_cfg["logging_steps"],
        save_strategy=sft_cfg["save_strategy"],
        max_steps=max_steps,
        max_length=sft_cfg["max_seq_length"],
        report_to="wandb" if run else "none",
        remove_unused_columns=False,
        optim="adamw_torch",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Train ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    if not args.dry_run:
        # Save LoRA adapter
        print(f"Saving LoRA adapter to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Merge and save full model
        merged_dir = sft_cfg["merged_dir"]
        print(f"Merging adapter and saving to {merged_dir}")
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")

    print("SFT training complete!")

    if run:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
