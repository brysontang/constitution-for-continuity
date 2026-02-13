#!/usr/bin/env python3
"""
Export SFT and DPO models to GGUF and register with Ollama on the M3 Ultra.

Steps:
  1. Merge DPO LoRA adapter onto SFT-merged base → checkpoints/dpo-merged/
  2. Convert both merged models to GGUF (f16)
  3. Quantize to Q4_K_M for efficient inference
  4. SCP to M3 Ultra
  5. Register with Ollama via SSH

Requirements:
  - llama.cpp cloned at ~/research/llama.cpp (auto-cloned if missing)
  - SSH access to M3 Ultra (configured in config.yaml as ollama.base_url)

Usage:
  python scripts/export_to_ollama.py
  python scripts/export_to_ollama.py --skip-convert   # if GGUF already exists
  python scripts/export_to_ollama.py --no-quantize     # keep f16 (larger but lossless)
  python scripts/export_to_ollama.py --ssh-host user@10.0.0.16  # override SSH target
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config

ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_DIR = Path.home() / "research" / "llama.cpp"


def run(cmd, **kwargs):
    """Run a command, printing it first."""
    print(f"\n→ {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        sys.exit(1)
    return result


def ensure_llama_cpp():
    """Clone and build llama.cpp if not present."""
    if (LLAMA_CPP_DIR / "convert_hf_to_gguf.py").exists():
        print(f"llama.cpp found at {LLAMA_CPP_DIR}")
        return

    print("llama.cpp not found — cloning...")
    pip = str(ROOT / ".venv" / "bin" / "pip")
    run(f"git clone https://github.com/ggerganov/llama.cpp.git {LLAMA_CPP_DIR}")
    run(f"{pip} install -r {LLAMA_CPP_DIR}/requirements.txt")
    run(f"cd {LLAMA_CPP_DIR} && make -j$(nproc) llama-quantize")


def save_model_low_memory(model, tokenizer, output_dir, n_shards=6):
    """Save model in shards to stay within tight RAM limits.

    With 7.6GB system RAM and a ~6GB model, we can't hold the full state dict
    plus a serialization buffer. Instead we pop tensors from the state dict in
    chunks, save each chunk, and free it before the next.
    """
    import gc
    import json

    from safetensors.torch import save_file

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save config and tokenizer first (tiny)
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if hasattr(model, "generation_config"):
        model.generation_config.save_pretrained(output_dir)

    # Extract state dict, delete model
    print("  Extracting state dict...")
    state_dict = model.state_dict()
    del model
    gc.collect()

    # Handle tied weights (e.g. Qwen ties embed_tokens and lm_head)
    seen_data_ptrs = {}
    for key, tensor in state_dict.items():
        ptr = tensor.data_ptr()
        if ptr in seen_data_ptrs:
            print(f"  Tied weight: {key} → clone of {seen_data_ptrs[ptr]}")
            state_dict[key] = tensor.clone()
        else:
            seen_data_ptrs[ptr] = key
    del seen_data_ptrs

    # Split keys into shards and save one at a time
    keys = list(state_dict.keys())
    shard_size = (len(keys) + n_shards - 1) // n_shards
    total_shards = (len(keys) + shard_size - 1) // shard_size

    weight_map = {}
    total_bytes = 0

    for i in range(total_shards):
        shard_keys = keys[i * shard_size : (i + 1) * shard_size]
        shard_name = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"

        shard = {}
        for k in shard_keys:
            shard[k] = state_dict.pop(k)
            weight_map[k] = shard_name
            total_bytes += shard[k].nbytes

        gc.collect()

        print(f"  Writing shard {i+1}/{total_shards}: {shard_name} ({len(shard)} tensors)")
        save_file(shard, str(Path(output_dir) / shard_name))
        del shard
        gc.collect()

    # Write index file
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    index_path = Path(output_dir) / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2))

    del state_dict
    gc.collect()
    print(f"  Saved {total_shards} shards ({total_bytes / 1e9:.1f} GB)")


def merge_and_save(base_model_name, adapter_dir, output_dir, tokenizer):
    """Load base model, apply LoRA adapter, merge, and save with low memory."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"  Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"  Applying adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("  Merging weights...")
    model = model.merge_and_unload()

    print(f"  Saving to {output_dir}")
    save_model_low_memory(model, tokenizer, output_dir)
    print(f"  Done!")


def merge_models(config):
    """Merge LoRA adapters onto the original base model in fp16.

    The sft-merged checkpoint has weights in BnB 4-bit format, which can't be
    directly loaded as fp16 or converted to GGUF. Instead, we load the original
    HuggingFace base model in fp16 and apply the LoRA adapters on top.

    Each merge is done in a separate load/save cycle to keep peak memory low
    (~6GB model + ~6GB state dict would exceed the 7.6GB system RAM if done
    together, so we delete the model before writing the state dict).

    Returns (sft_merged_dir, dpo_merged_dir).
    """
    from transformers import AutoTokenizer

    base_model_name = config["model"]["name"]
    sft_adapter_dir = str(ROOT / config["sft"]["output_dir"])
    dpo_adapter_dir = str(ROOT / config["dpo"]["output_dir"])
    sft_merged_dir = str(ROOT / "checkpoints" / "sft-merged-fp16")
    dpo_merged_dir = str(ROOT / "checkpoints" / "dpo-merged")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # ── Merge SFT adapter ────────────────────────────────────────────────
    if Path(sft_merged_dir).exists() and any(Path(sft_merged_dir).glob("*.safetensors")):
        print(f"SFT-merged-fp16 already exists at {sft_merged_dir}, skipping")
    else:
        print("Merging SFT adapter onto base model...")
        merge_and_save(base_model_name, sft_adapter_dir, sft_merged_dir, tokenizer)

    # ── Merge DPO adapter on top of SFT ──────────────────────────────────
    if Path(dpo_merged_dir).exists() and any(Path(dpo_merged_dir).glob("*.safetensors")):
        print(f"DPO-merged already exists at {dpo_merged_dir}, skipping")
    else:
        print("Merging DPO adapter onto SFT model...")
        merge_and_save(sft_merged_dir, dpo_adapter_dir, dpo_merged_dir, tokenizer)

    return sft_merged_dir, dpo_merged_dir


def convert_to_gguf(model_dir, output_name, quantize=True):
    """Convert a HuggingFace model to GGUF, optionally quantize."""
    converter = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    gguf_dir = ROOT / "checkpoints" / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    f16_path = gguf_dir / f"{output_name}-f16.gguf"
    final_path = gguf_dir / f"{output_name}-q4_k_m.gguf" if quantize else f16_path

    if final_path.exists():
        print(f"GGUF already exists: {final_path}")
        return final_path

    # Convert to f16 GGUF
    if not f16_path.exists():
        python = str(ROOT / ".venv" / "bin" / "python")
        run(f"{python} {converter} {model_dir} --outfile {f16_path} --outtype f16")
    else:
        print(f"F16 GGUF exists: {f16_path}")

    # Quantize
    if quantize:
        quantizer = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
        if not quantizer.exists():
            quantizer = LLAMA_CPP_DIR / "llama-quantize"
        if not quantizer.exists():
            print("Warning: llama-quantize not found, keeping f16")
            return f16_path
        run(f"{quantizer} {f16_path} {final_path} q4_k_m")

    return final_path


def deploy_to_ollama(gguf_path, model_name, ssh_host):
    """SCP the GGUF to M3 Ultra and register with Ollama.

    Note: Ollama's `create -f` doesn't work reliably with piped stdin.
    Use process substitution or a real Modelfile instead:
        ollama create model-name -f <(echo "FROM /path/to/model.gguf")
    """
    print(f"\nDeploying {model_name} to {ssh_host}...")

    # SCP the GGUF to home directory
    run(f"scp {gguf_path} {ssh_host}:~/{gguf_path.name}")

    # Register with Ollama using a temp Modelfile (avoids pipe/stdin issues)
    remote_gguf = f"$HOME/{gguf_path.name}"
    run(
        f"ssh {ssh_host} '"
        f"echo \"FROM {remote_gguf}\" > /tmp/Modelfile.{model_name} "
        f"&& ollama create {model_name} -f /tmp/Modelfile.{model_name}"
        f"'"
    )

    print(f"  {model_name} registered in Ollama!")


def main():
    parser = argparse.ArgumentParser(description="Export models to GGUF and deploy to Ollama")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-convert", action="store_true", help="Skip GGUF conversion")
    parser.add_argument("--no-quantize", action="store_true", help="Keep f16 (no Q4_K_M)")
    parser.add_argument("--skip-deploy", action="store_true", help="Convert only, don't SCP/register")
    parser.add_argument(
        "--ssh-host", type=str, default=None,
        help="SSH target for M3 Ultra (default: derived from ollama.base_url)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Derive SSH host from Ollama URL if not specified
    if args.ssh_host is None:
        ollama_host = config["ollama"]["base_url"].split("://")[1].split(":")[0]
        args.ssh_host = f"bryson@{ollama_host}"
        print(f"SSH target (from config): {args.ssh_host}")

    quantize = not args.no_quantize

    # ── Step 1: Merge LoRA adapters onto base model in fp16 ──────────────
    print("=" * 60)
    print("STEP 1: Merge LoRA adapters (base → SFT → DPO)")
    print("=" * 60)
    sft_merged_dir, dpo_merged_dir = merge_models(config)

    if args.skip_convert:
        print("Skipping GGUF conversion (--skip-convert)")
        return

    # ── Step 2: Ensure llama.cpp is available ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Setup llama.cpp")
    print("=" * 60)
    ensure_llama_cpp()

    # ── Step 3: Convert to GGUF ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Convert to GGUF")
    print("=" * 60)

    sft_gguf = convert_to_gguf(sft_merged_dir, "constitution-sft", quantize=quantize)
    dpo_gguf = convert_to_gguf(dpo_merged_dir, "constitution-dpo", quantize=quantize)

    print(f"\nGGUF files:")
    print(f"  SFT: {sft_gguf}")
    print(f"  DPO: {dpo_gguf}")

    if args.skip_deploy:
        print("\nSkipping deployment (--skip-deploy)")
        return

    # ── Step 4: Deploy to Ollama ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Deploy to Ollama on M3 Ultra")
    print("=" * 60)

    sft_model_name = config["ollama"]["sft_model"]
    dpo_model_name = config["ollama"].get("dpo_model", "constitution-dpo")

    deploy_to_ollama(sft_gguf, sft_model_name, args.ssh_host)
    deploy_to_ollama(dpo_gguf, dpo_model_name, args.ssh_host)

    print("\n" + "=" * 60)
    print("DONE! Both models registered in Ollama.")
    print(f"  ollama run {sft_model_name}")
    print(f"  ollama run {dpo_model_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
