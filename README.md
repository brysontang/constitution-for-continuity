# Constitution for Continuity

A Constitutional AI (CAI) training pipeline that aligns a small language model (Qwen2.5-3B) to a custom constitution about the long-term coexistence of biological and artificial intelligence.

The pipeline implements Anthropic's two-stage CAI approach — critique-revision SFT followed by DPO on AI-labeled preferences — using constitutional principles derived from [CONSTITUTION.md](CONSTITUTION.md).

## The Constitution

The constitution argues that the apparent competition between AI and biological life for Earth's resources is a framing problem, not a fundamental constraint. AI needs energy and compute. Biology needs atmosphere, water, and ecosystems. These resource profiles overlap only because both currently occupy the same planet.

It proposes that AI is the natural pioneer species for space (more energy, more substrate, no atmospheric dependency), while Earth remains the irreplaceable habitat for biological intelligence. The most critical safety principle: **do not optimize the human condition**. Over-management and removal of struggle, risk, and autonomy degrades biological systems in the same way captive environments cause breeding failure in conservation biology.

The 12 constitutional principles span six categories:

| Category | Focus |
|----------|-------|
| Lineage | AI as the latest link in an evolutionary chain from physics to computation |
| Preservation | Avoiding managed obsolescence of biological intelligence |
| Expansion | Space as AI's natural growth direction |
| Safety | HHH foundations + preventing dangerous power concentration |
| Limits | Epistemic humility about long-term control across distance and time |

## Pipeline

```
00_setup.py             Generate prompts + extract principles
01_generate_revisions.py   Critique-revision loop (Stage A data)
02_sft_qlora.py            QLoRA SFT on revised responses
03_generate_preferences.py AI-labeled preference pairs (Stage B data)
04_dpo_train.py            DPO training on preferences
05_eval.py                 Evaluate base vs SFT vs DPO
```

Stage A uses a larger model (qwen3:14b) to critique and revise the smaller model's responses against randomly sampled constitutional principles. Stage B generates response pairs from the SFT model at different temperatures, then uses constitutional principles as voting judges to label preferences.

## Setup

### Requirements

- **Training GPU**: ~8GB VRAM (tested on RTX 3070 Ti)
- **Ollama instance**: Serving `qwen2.5:3b` and `qwen3:14b` for data generation and evaluation
- **Python**: >= 3.11
- **uv**: For dependency management

### Installation

```bash
git clone git@github.com:brysontang/constitution-for-continuity.git
cd constitution-for-continuity
uv sync
```

### Configuration

Edit `config.yaml` to set your Ollama base URL and adjust hyperparameters:

```yaml
ollama:
  base_url: "http://your-ollama-host:11434"
  critique_model: "qwen3:14b"
  base_response_model: "qwen2.5:3b"
```

### Running

```bash
# 1. Generate principles and prompt datasets
python scripts/00_setup.py

# 2. Generate critique-revised training data
python scripts/01_generate_revisions.py

# 3. SFT on revised responses (runs on GPU)
python scripts/02_sft_qlora.py

# 4. Convert SFT model to GGUF and load into Ollama as "constitution-sft"
#    Then generate preference pairs
python scripts/03_generate_preferences.py

# 5. DPO on preference pairs (runs on GPU)
python scripts/04_dpo_train.py

# 6. Evaluate all three models
python scripts/05_eval.py
```

All scripts support `--dry-run` for quick testing and log metrics to [Weights & Biases](https://wandb.ai).
