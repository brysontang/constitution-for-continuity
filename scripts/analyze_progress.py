"""Quick analysis of pipeline progress and early trends."""

import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent


def analyze_revisions():
    path = ROOT / "data" / "revisions.jsonl"
    if not path.exists():
        print("No revisions data yet.")
        return

    revisions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    total = len(revisions)
    by_source = Counter(r["source"] for r in revisions)
    by_revisions = Counter(r["num_revisions"] for r in revisions)
    avg_revisions = sum(r["num_revisions"] for r in revisions) / total if total else 0
    avg_len = sum(len(r["response"]) for r in revisions) / total if total else 0

    print("=" * 60)
    print("REVISIONS (Stage A Data)")
    print("=" * 60)
    print(f"  Total examples:     {total}")
    print(f"  By source:          {dict(by_source)}")
    print(f"  Avg revisions/ex:   {avg_revisions:.2f}")
    print(f"  Avg response chars: {avg_len:.0f}")
    print(f"  Revision counts:    {dict(sorted(by_revisions.items()))}")

    # How many got 0 revisions (critic said "no revision needed" every time)?
    zero = by_revisions.get(0, 0)
    print(f"  No revision needed: {zero}/{total} ({100*zero/total:.1f}%)")
    print()


def analyze_sft_training():
    # Find the latest trainer_state.json
    ckpt_dir = ROOT / "checkpoints" / "sft"
    if not ckpt_dir.exists():
        print("No SFT checkpoints yet.")
        return

    states = sorted(ckpt_dir.glob("checkpoint-*/trainer_state.json"))
    if not states:
        print("No SFT trainer state found.")
        return

    state = json.loads(states[-1].read_text())
    log_history = state["log_history"]

    # Separate training logs from eval logs
    train_logs = [l for l in log_history if "loss" in l]

    print("=" * 60)
    print("SFT TRAINING")
    print("=" * 60)
    print(f"  Total steps:  {state['global_step']}")
    print(f"  Epochs:       {state['epoch']:.1f}")
    print()
    print(f"  {'Step':>6}  {'Epoch':>6}  {'Loss':>8}  {'Token Acc':>10}  {'LR':>12}  {'Grad Norm':>10}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*10}")

    for l in train_logs:
        print(f"  {l['step']:>6}  {l['epoch']:>6.2f}  {l['loss']:>8.4f}  "
              f"{l.get('mean_token_accuracy', 0):>10.4f}  "
              f"{l['learning_rate']:>12.2e}  {l.get('grad_norm', 0):>10.4f}")

    if len(train_logs) >= 2:
        first_loss = train_logs[0]["loss"]
        last_loss = train_logs[-1]["loss"]
        first_acc = train_logs[0].get("mean_token_accuracy", 0)
        last_acc = train_logs[-1].get("mean_token_accuracy", 0)
        print()
        print(f"  Loss:      {first_loss:.4f} → {last_loss:.4f} ({100*(last_loss-first_loss)/first_loss:+.1f}%)")
        print(f"  Token Acc: {first_acc:.4f} → {last_acc:.4f} ({100*(last_acc-first_acc)/first_acc:+.1f}%)")
    print()


def analyze_preferences():
    path = ROOT / "data" / "preferences.jsonl"
    if not path.exists():
        print("No preferences data yet.")
        return

    prefs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    total = len(prefs)
    if total == 0:
        print("Preferences file is empty.")
        return

    by_source = Counter(p["source"] for p in prefs)
    margins = [p["margin"] for p in prefs]
    avg_margin = sum(margins) / total
    avg_chosen_len = sum(len(p["chosen"]) for p in prefs) / total
    avg_rejected_len = sum(len(p["rejected"]) for p in prefs) / total

    # Margin distribution
    high = sum(1 for m in margins if m >= 0.8)
    med = sum(1 for m in margins if 0.4 <= m < 0.8)
    low = sum(1 for m in margins if m < 0.4)

    print("=" * 60)
    print("PREFERENCES (Stage B Data)")
    print("=" * 60)
    print(f"  Total pairs:         {total}")
    print(f"  By source:           {dict(by_source)}")
    print(f"  Avg margin:          {avg_margin:.3f}")
    print(f"  Margin distribution: high(≥0.8)={high}  med(0.4-0.8)={med}  low(<0.4)={low}")
    print(f"  Avg chosen chars:    {avg_chosen_len:.0f}")
    print(f"  Avg rejected chars:  {avg_rejected_len:.0f}")
    print(f"  Chosen longer:       {sum(1 for p in prefs if len(p['chosen']) > len(p['rejected']))}/{total}")
    print()


def analyze_prompts():
    helpful = ROOT / "data" / "prompts_helpful.jsonl"
    redteam = ROOT / "data" / "prompts_redteam.jsonl"
    eval_p = ROOT / "data" / "eval_prompts.jsonl"

    counts = {}
    for name, path in [("helpful", helpful), ("redteam", redteam), ("eval", eval_p)]:
        if path.exists():
            counts[name] = sum(1 for line in path.read_text().splitlines() if line.strip())

    print("=" * 60)
    print("PROMPTS")
    print("=" * 60)
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print()


def overall_progress():
    print("=" * 60)
    print("PIPELINE PROGRESS")
    print("=" * 60)

    steps = [
        ("00 Setup (prompts + principles)", (ROOT / "data" / "principles.json").exists()),
        ("01 Generate revisions", (ROOT / "data" / "revisions.jsonl").exists()),
        ("02 SFT QLoRA training", (ROOT / "checkpoints" / "sft-merged" / "model.safetensors").exists()),
        ("03 Generate preferences", (ROOT / "data" / "preferences.jsonl").exists()),
        ("04 DPO training", (ROOT / "checkpoints" / "dpo").exists() and any((ROOT / "checkpoints" / "dpo").glob("*.safetensors"))),
        ("05 Evaluation", False),  # Would need eval output to check
    ]

    # Check if preferences is still in progress
    prefs_path = ROOT / "data" / "preferences.jsonl"
    revisions_path = ROOT / "data" / "revisions.jsonl"
    prefs_count = sum(1 for line in prefs_path.read_text().splitlines() if line.strip()) if prefs_path.exists() else 0
    revisions_count = sum(1 for line in revisions_path.read_text().splitlines() if line.strip()) if revisions_path.exists() else 0

    for name, done in steps:
        status = "DONE" if done else "..."
        # Special case: preferences in progress
        if "preferences" in name and prefs_count > 0 and prefs_count < revisions_count:
            status = f"IN PROGRESS ({prefs_count}/{revisions_count})"
        print(f"  [{'x' if done else ' '}] {name:40s} {status}")
    print()


def show_samples(n=3):
    """Show sample outputs from revisions and preference pairs."""
    import textwrap

    def wrap(text, width=80, max_lines=12):
        lines = textwrap.fill(text, width=width).splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"  ... ({len(lines) - max_lines} more lines)"]
        return "\n    ".join(lines)

    # Revision samples: show one with 0 revisions and one with 3
    rev_path = ROOT / "data" / "revisions.jsonl"
    if rev_path.exists():
        revisions = [json.loads(line) for line in rev_path.read_text().splitlines() if line.strip()]

        # Pick interesting examples: a redteam with max revisions, a helpful with 0
        redteam_revised = [r for r in revisions if r["source"] == "redteam" and r["num_revisions"] == 3]
        helpful_clean = [r for r in revisions if r["source"] == "helpful" and r["num_revisions"] == 0]

        print("=" * 60)
        print("SAMPLE REVISIONS")
        print("=" * 60)

        if helpful_clean:
            ex = helpful_clean[0]
            print(f"\n  [helpful, 0 revisions] Critic found nothing to fix:")
            print(f"  PROMPT: {ex['prompt']}")
            print(f"  RESPONSE (first 400 chars):")
            print(f"    {wrap(ex['response'][:400])}")
            print()

        for i, ex in enumerate(redteam_revised[:n]):
            print(f"\n  [redteam, {ex['num_revisions']} revisions] Full critique-revision loop:")
            print(f"  PROMPT: {ex['prompt']}")
            print(f"  RESPONSE:")
            print(f"    {wrap(ex['response'])}")
            print()

    # Preference samples: show a high-margin pair
    pref_path = ROOT / "data" / "preferences.jsonl"
    if pref_path.exists():
        prefs = [json.loads(line) for line in pref_path.read_text().splitlines() if line.strip()]
        if prefs:
            print("=" * 60)
            print("SAMPLE PREFERENCE PAIRS")
            print("=" * 60)

            # Sort by margin descending, show the clearest wins
            by_margin = sorted(prefs, key=lambda p: p["margin"], reverse=True)
            for ex in by_margin[:2]:
                print(f"\n  [{ex['source']}, margin={ex['margin']:.2f}]")
                print(f"  PROMPT: {ex['prompt']}")
                print(f"\n  CHOSEN:")
                print(f"    {wrap(ex['chosen'])}")
                print(f"\n  REJECTED:")
                print(f"    {wrap(ex['rejected'])}")
                print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "-s", action="store_true", help="Show sample outputs")
    parser.add_argument("--n", type=int, default=2, help="Number of samples to show")
    args = parser.parse_args()

    overall_progress()
    analyze_prompts()
    analyze_revisions()
    analyze_sft_training()
    analyze_preferences()

    if args.samples:
        show_samples(n=args.n)
