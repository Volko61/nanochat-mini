#!/bin/bash
set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# UV + venv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Dummy WANDB
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

# Rust (for tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download **just 1 shard** (tiny)
python -m nanochat.dataset -n 1

# Train tokenizer on ~250M chars instead of 2B
python -m scripts.tok_train --max_chars=250000000
python -m scripts.tok_eval

# --------------------------------------------
# Tiny base model training (1 GPU)
# Depth = 4, dim = ~256 (small)
# --------------------------------------------
python -m scripts.base_train -- --depth=4 --dim=256 --run=$WANDB_RUN
python -m scripts.base_loss
python -m scripts.base_eval

# --------------------------------------------
# Midtraining
# --------------------------------------------
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.mid_train -- --run=$WANDB_RUN
python -m scripts.chat_eval -- -i mid

# --------------------------------------------
# SFT training
# --------------------------------------------
python -m scripts.chat_sft -- --run=$WANDB_RUN
python -m scripts.chat_eval -- -i sft

# --------------------------------------------
# Chat UI
# --------------------------------------------
python -m scripts.chat_cli -p "Explain quantum physics simply."
python -m scripts.chat_web
