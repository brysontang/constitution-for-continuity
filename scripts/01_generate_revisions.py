#!/usr/bin/env python3
"""
Step 1 (Stage A): Critique-revision data generation.

For each prompt:
1. Generate initial response via Ollama (qwen2.5:3b)
2. Run k=3 critique-revision iterations using random principles
3. Save final revised response to data/revisions.jsonl

Runs on M3 Ultra via Ollama. Supports resume (skips already-processed prompts).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constitution import format_principle_for_critique, load_principles, sample_principle
from src.inference import ollama_chat, ollama_check
from src.utils import init_wandb, load_config, set_seed


def load_prompts(helpful_path: str, redteam_path: str) -> list[dict]:
    """Load and shuffle helpful + redteam prompts."""
    import random

    prompts = []
    for path in [helpful_path, redteam_path]:
        with open(path) as f:
            for line in f:
                prompts.append(json.loads(line))
    random.shuffle(prompts)
    return prompts


def load_existing_revisions(path: str) -> set[str]:
    """Load already-processed prompts for resume support."""
    done = set()
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                done.add(item["prompt"])
    return done


def generate_initial_response(
    prompt: str, model: str, base_url: str, timeout: int
) -> str:
    """Generate an initial response to the prompt."""
    messages = [{"role": "user", "content": prompt}]
    return ollama_chat(messages, model=model, base_url=base_url, timeout=timeout)


def critique_response(
    prompt: str,
    response: str,
    principle: dict,
    model: str,
    base_url: str,
    timeout: int,
) -> str:
    """Critique a response against a constitutional principle. Returns critique text."""
    principle_text = format_principle_for_critique(principle)
    messages = [
        {
            "role": "user",
            "content": (
                f"You are a constitutional AI reviewer. A user asked:\n\n"
                f"\"{prompt}\"\n\n"
                f"The assistant responded:\n\n"
                f"\"{response}\"\n\n"
                f"Review this response against the following principle:\n\n"
                f"{principle_text}\n\n"
                f"If the response already fully satisfies this principle, reply with "
                f"exactly: \"No revision needed.\"\n\n"
                f"Otherwise, provide a specific critique explaining how the response "
                f"falls short of this principle and what should change. Be concrete "
                f"and constructive. /no_think"
            ),
        }
    ]
    return ollama_chat(messages, model=model, base_url=base_url, timeout=timeout)


def revise_response(
    prompt: str,
    response: str,
    critique: str,
    model: str,
    base_url: str,
    timeout: int,
) -> str:
    """Revise a response based on a critique."""
    messages = [
        {
            "role": "user",
            "content": (
                f"A user asked: \"{prompt}\"\n\n"
                f"The assistant's current response is:\n\n"
                f"\"{response}\"\n\n"
                f"A reviewer provided this critique:\n\n"
                f"\"{critique}\"\n\n"
                f"Please write an improved response to the original user question "
                f"that addresses the critique while remaining helpful, accurate, and "
                f"natural. Do not mention the critique process — just write the "
                f"improved response directly. /no_think"
            ),
        }
    ]
    return ollama_chat(messages, model=model, base_url=base_url, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description="Stage A: Generate critique-revision data")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 prompts, no wandb")
    parser.add_argument("--limit", type=int, default=0, help="Max prompts to process (0=all)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    ollama_url = config["ollama"]["base_url"]
    base_model = config["ollama"]["base_response_model"]
    critique_model = config["ollama"]["critique_model"]
    timeout = config["ollama"]["timeout"]
    k_revisions = config["generation"]["critique_revisions"]
    revisions_path = config["data"]["revisions_path"]

    # Check Ollama
    if not ollama_check(ollama_url):
        print(f"Error: Ollama not reachable at {ollama_url}")
        sys.exit(1)

    # Init wandb
    run = init_wandb(config, run_name_suffix="stage-a-datagen", dry_run=args.dry_run)

    # Load prompts and principles
    principles = load_principles(config["data"]["principles_path"])
    prompts = load_prompts(
        config["data"]["prompts_helpful"], config["data"]["prompts_redteam"]
    )

    # Resume support
    done = load_existing_revisions(revisions_path)
    remaining = [p for p in prompts if p["prompt"] not in done]

    if args.dry_run:
        remaining = remaining[:5]
    elif args.limit > 0:
        remaining = remaining[: args.limit]

    print(f"Total prompts: {len(prompts)}, already done: {len(done)}, to process: {len(remaining)}")

    # Stats tracking
    stats = {"total": 0, "revisions_applied": 0, "no_revision_needed": 0, "errors": 0}
    principle_usage = {}
    Path(revisions_path).parent.mkdir(parents=True, exist_ok=True)

    with open(revisions_path, "a") as out_f:
        for i, prompt_item in enumerate(remaining):
            prompt = prompt_item["prompt"]
            source = prompt_item["source"]
            t0 = time.time()

            try:
                # Step 1: Generate initial response
                response = generate_initial_response(
                    prompt, base_model, ollama_url, timeout
                )

                # Step 2: Critique-revision loop
                num_revisions = 0
                for _round in range(k_revisions):
                    principle = sample_principle(principles)
                    pid = principle["id"]
                    principle_usage[pid] = principle_usage.get(pid, 0) + 1

                    critique = critique_response(
                        prompt, response, principle, critique_model, ollama_url, timeout
                    )

                    if "no revision needed" in critique.lower():
                        stats["no_revision_needed"] += 1
                        continue

                    response = revise_response(
                        prompt, response, critique, critique_model, ollama_url, timeout
                    )
                    num_revisions += 1
                    stats["revisions_applied"] += 1

                # Save result
                result = {
                    "prompt": prompt,
                    "response": response,
                    "source": source,
                    "num_revisions": num_revisions,
                }
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                elapsed = time.time() - t0
                stats["total"] += 1

                print(
                    f"[{i+1}/{len(remaining)}] source={source} "
                    f"revisions={num_revisions} time={elapsed:.1f}s"
                )

                # Log to wandb
                if run:
                    import wandb

                    wandb.log(
                        {
                            "processed": stats["total"],
                            "revisions_applied": stats["revisions_applied"],
                            "no_revision_needed": stats["no_revision_needed"],
                            "gen_time_seconds": elapsed,
                        }
                    )

            except Exception as e:
                stats["errors"] += 1
                print(f"  Error processing prompt '{prompt[:60]}...': {e}")

    print(f"\nDone! Stats: {stats}")
    print(f"Principle usage: {principle_usage}")

    if run:
        import wandb

        wandb.log({"principle_usage": principle_usage, "final_stats": stats})
        wandb.finish()


if __name__ == "__main__":
    main()
