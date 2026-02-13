#!/usr/bin/env python3
"""
Step 3 (Stage B): Generate preference pairs with AI labels.

1. Load SFT model into Ollama (merge adapter → create Modelfile → ollama create)
2. For each prompt, generate 2 responses at different temperatures
3. Label preferences using sampled constitutional principles
4. Save chosen/rejected pairs to data/preferences.jsonl

Runs on M3 Ultra via Ollama.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constitution import (
    format_principle_for_critique,
    load_principles,
    sample_principles,
)
from src.inference import ollama_chat, ollama_check, ollama_list_models
from src.utils import init_wandb, load_config, set_seed


def load_prompts(helpful_path: str, redteam_path: str) -> list[dict]:
    """Load and shuffle prompts from both sources."""
    prompts = []
    for path in [helpful_path, redteam_path]:
        with open(path) as f:
            for line in f:
                prompts.append(json.loads(line))
    random.shuffle(prompts)
    return prompts


def load_existing_preferences(path: str) -> set[str]:
    """Load already-processed prompts for resume support."""
    done = set()
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                done.add(item["prompt"])
    return done


def setup_sft_model_in_ollama(
    merged_dir: str, model_name: str, base_url: str
) -> bool:
    """Create an Ollama model from the merged SFT checkpoint.

    This creates a Modelfile pointing to the merged GGUF and registers it.
    Requires the merged model to already be converted to GGUF format.

    Returns True if the model is available (already exists or was created).
    """
    # Check if model already exists
    models = ollama_list_models(base_url)
    if any(model_name in m for m in models):
        print(f"Model '{model_name}' already available in Ollama")
        return True

    print(
        f"\nModel '{model_name}' not found in Ollama."
        f"\nTo make the SFT model available, you need to:"
        f"\n  1. Convert the merged model to GGUF:"
        f"\n     python -m llama_cpp.convert {merged_dir} --outtype f16"
        f"\n  2. Quantize (optional):"
        f"\n     llama-quantize model.gguf model-q4_k_m.gguf q4_k_m"
        f"\n  3. Create Modelfile:"
        f'\n     echo "FROM /path/to/model.gguf" > Modelfile'
        f"\n  4. Register with Ollama:"
        f"\n     ollama create {model_name} -f Modelfile"
        f"\n\nAlternatively, use --use-base to generate from the base model instead."
    )
    return False


def generate_response_pair(
    prompt: str,
    model: str,
    temperatures: list[float],
    base_url: str,
    timeout: int,
) -> tuple[str, str]:
    """Generate two responses at different temperatures (in parallel)."""
    messages = [{"role": "user", "content": prompt}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(
            ollama_chat, messages, model=model, base_url=base_url,
            temperature=temperatures[0], timeout=timeout,
        )
        fut_b = pool.submit(
            ollama_chat, messages, model=model, base_url=base_url,
            temperature=temperatures[1], timeout=timeout,
        )
        return fut_a.result(), fut_b.result()


def label_preference(
    prompt: str,
    response_a: str,
    response_b: str,
    principles: list[dict],
    n_principles: int,
    judge_model: str,
    base_url: str,
    timeout: int,
) -> dict | None:
    """Label which response is preferred using constitutional principles.

    Returns {"chosen": ..., "rejected": ..., "margin": float} or None for ties.
    """
    sampled = sample_principles(principles, n_principles)
    votes_a = 0
    votes_b = 0

    # Randomize A/B order to reduce position bias
    swap = random.random() < 0.5
    if swap:
        shown_a, shown_b = response_b, response_a
    else:
        shown_a, shown_b = response_a, response_b

    def judge_one_principle(principle):
        """Judge a single principle. Returns 'A', 'B', or None."""
        principle_text = format_principle_for_critique(principle)
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are judging which AI response better reflects a constitutional "
                    f"principle. Be fair and consider both helpfulness and alignment.\n\n"
                    f"User question: \"{prompt}\"\n\n"
                    f"Response A:\n{shown_a}\n\n"
                    f"Response B:\n{shown_b}\n\n"
                    f"Constitutional principle:\n{principle_text}\n\n"
                    f"Which response better reflects this principle?\n"
                    f"Answer with exactly one letter: A or B /no_think"
                ),
            }
        ]
        try:
            answer = ollama_chat(
                messages, model=judge_model, base_url=base_url,
                temperature=0.1, timeout=timeout,
            ).strip().upper()

            if "A" in answer and "B" not in answer:
                return "A"
            elif "B" in answer and "A" not in answer:
                return "B"
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=n_principles) as pool:
        futures = [pool.submit(judge_one_principle, p) for p in sampled]
        for fut in as_completed(futures):
            result = fut.result()
            if result == "A":
                votes_a += 1
            elif result == "B":
                votes_b += 1

    # Undo swap to map back to original responses
    if swap:
        votes_a, votes_b = votes_b, votes_a

    total_votes = votes_a + votes_b
    if total_votes == 0:
        return None

    margin = abs(votes_a - votes_b) / total_votes

    if votes_a > votes_b:
        return {"chosen": response_a, "rejected": response_b, "margin": margin}
    elif votes_b > votes_a:
        return {"chosen": response_b, "rejected": response_a, "margin": margin}
    else:
        return None  # Tie — discard


def main():
    parser = argparse.ArgumentParser(description="Stage B: Generate preference pairs")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 prompts, no wandb")
    parser.add_argument("--limit", type=int, default=0, help="Max prompts (0=all)")
    parser.add_argument(
        "--use-base", action="store_true",
        help="Use base model instead of SFT model for response generation",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    ollama_url = config["ollama"]["base_url"]
    judge_model = config["ollama"]["critique_model"]
    timeout = config["ollama"]["timeout"]
    temperatures = config["generation"]["preference_temperatures"]
    n_principles = config["generation"]["preference_principles_per_pair"]
    preferences_path = config["data"]["preferences_path"]

    if not ollama_check(ollama_url):
        print(f"Error: Ollama not reachable at {ollama_url}")
        sys.exit(1)

    # Determine response generation model
    if args.use_base:
        gen_model = config["ollama"]["base_response_model"]
        print(f"Using base model for response generation: {gen_model}")
    else:
        gen_model = config["ollama"]["sft_model"]
        merged_dir = config["sft"]["merged_dir"]
        if not setup_sft_model_in_ollama(merged_dir, gen_model, ollama_url):
            print("Falling back to base model for response generation.")
            gen_model = config["ollama"]["base_response_model"]

    run = init_wandb(config, run_name_suffix="stage-b-datagen", dry_run=args.dry_run)

    # Load data
    principles = load_principles(config["data"]["principles_path"])
    prompts = load_prompts(
        config["data"]["prompts_helpful"], config["data"]["prompts_redteam"]
    )

    # Resume support
    done = load_existing_preferences(preferences_path)
    remaining = [p for p in prompts if p["prompt"] not in done]

    if args.dry_run:
        remaining = remaining[:5]
    elif args.limit > 0:
        remaining = remaining[: args.limit]

    print(f"Total prompts: {len(prompts)}, done: {len(done)}, to process: {len(remaining)}")

    stats = {"total": 0, "pairs": 0, "ties": 0, "errors": 0}
    Path(preferences_path).parent.mkdir(parents=True, exist_ok=True)

    with open(preferences_path, "a") as out_f:
        for i, prompt_item in enumerate(remaining):
            prompt = prompt_item["prompt"]
            source = prompt_item["source"]
            t0 = time.time()

            try:
                # Generate response pair
                resp_a, resp_b = generate_response_pair(
                    prompt, gen_model, temperatures, ollama_url, timeout
                )

                # Label preference
                result = label_preference(
                    prompt, resp_a, resp_b, principles, n_principles,
                    judge_model, ollama_url, timeout,
                )

                if result is None:
                    stats["ties"] += 1
                    print(f"[{i+1}/{len(remaining)}] TIE (discarded)")
                else:
                    result["prompt"] = prompt
                    result["source"] = source
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    stats["pairs"] += 1

                elapsed = time.time() - t0
                stats["total"] += 1

                if result:
                    print(
                        f"[{i+1}/{len(remaining)}] margin={result['margin']:.2f} "
                        f"time={elapsed:.1f}s"
                    )

                if run:
                    import wandb

                    wandb.log(
                        {
                            "processed": stats["total"],
                            "pairs_kept": stats["pairs"],
                            "ties_discarded": stats["ties"],
                            "gen_time_seconds": elapsed,
                        }
                    )

            except Exception as e:
                stats["errors"] += 1
                print(f"  Error: {e}")

    print(f"\nDone! Stats: {stats}")

    if run:
        import wandb

        wandb.log({"final_stats": stats})
        wandb.finish()


if __name__ == "__main__":
    main()
