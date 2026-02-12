#!/usr/bin/env python3
"""
Step 4 (Stage B): DPO training on preference pairs.

Loads the SFT-merged model in 4-bit NF4, attaches fresh LoRA adapters,
and trains with DPOTrainer on the AI-labeled preference data.

Runs on local GPU (RTX 3070 Ti, 8GB VRAM).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import init_wandb, load_config, set_seed


def load_preferences(path: str) -> list[dict]:
    """Load preference pairs from JSONL."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Stage B: DPO training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Load model, run 1 step, exit")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import DPOConfig, DPOTrainer

    dpo_cfg = config["dpo"]
    lora_cfg = config["lora"]
    sft_merged_dir = config["sft"]["merged_dir"]

    run = init_wandb(config, run_name_suffix="dpo", dry_run=args.dry_run)

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer from {sft_merged_dir}")
    tokenizer = AutoTokenizer.from_pretrained(sft_merged_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load SFT-merged model in 4-bit ──────────────────────────────────────
    print(f"Loading SFT-merged model in 4-bit NF4: {sft_merged_dir}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config["model"]["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        sft_merged_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Fresh LoRA for DPO ──────────────────────────────────────────────────
    print("Attaching fresh LoRA adapters for DPO...")
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

    # ── Load and format preference data ─────────────────────────────────────
    print(f"Loading preferences from {config['data']['preferences_path']}")
    prefs = load_preferences(config["data"]["preferences_path"])
    print(f"Loaded {len(prefs)} preference pairs")

    # Format for DPOTrainer: needs prompt, chosen, rejected as chat messages
    def format_pair(item):
        prompt_msgs = [{"role": "user", "content": item["prompt"]}]
        chosen_msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["chosen"]},
        ]
        rejected_msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["rejected"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            ),
            "chosen": tokenizer.apply_chat_template(
                chosen_msgs, tokenize=False, add_generation_prompt=False
            ),
            "rejected": tokenizer.apply_chat_template(
                rejected_msgs, tokenize=False, add_generation_prompt=False
            ),
        }

    formatted = [format_pair(p) for p in prefs]
    dataset = Dataset.from_list(formatted)

    if args.dry_run:
        dataset = dataset.select(range(min(4, len(dataset))))
        print(f"Dry-run: using {len(dataset)} examples")

    # ── DPO training config ─────────────────────────────────────────────────
    output_dir = dpo_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    max_steps = 2 if args.dry_run else -1
    num_epochs = 1 if args.dry_run else dpo_cfg["epochs"]

    # Detect bf16 support — NVML driver mismatch can cause false negatives
    use_bf16 = dpo_cfg["bf16"]
    use_fp16 = False
    if use_bf16:
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            use_bf16 = False
        if not use_bf16:
            print("Warning: bf16 not detected, falling back to fp16")
            use_fp16 = True

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=dpo_cfg["batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        lr_scheduler_type=dpo_cfg["lr_scheduler_type"],
        warmup_ratio=dpo_cfg["warmup_ratio"],
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=dpo_cfg["gradient_checkpointing"],
        logging_steps=dpo_cfg["logging_steps"],
        save_strategy=dpo_cfg["save_strategy"],
        max_steps=max_steps,
        report_to="wandb" if run else "none",
        remove_unused_columns=False,
        optim="adamw_torch",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Train ───────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    if not args.dry_run:
        print(f"Saving DPO adapter to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    print("DPO training complete!")

    if run:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
