import os

# data
DATASET = "OmniSVG/MMSVG-Icon"
SPLIT = "train"
CAPTION_FIELD = "description"

SAMPLE_SIZE = 50_000
VAL_SIZE = 1_000
SEED = 42

PROMPT_BUDGET = 150
MAX_LENGTH = 1000

PROMPT_TEMPLATE = "Generate an SVG icon for:\n{description}\n\n"

# model
MODEL_NAME = "Qwen/Qwen2.5-7B"
LOAD_IN_4BIT = True

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

# training
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR_LORA = 2e-4
LR_EMBEDDING = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.03
MAX_GRAD_NORM = 1.0

# io
ROOT = os.environ.get("SVG_ROOT", "PATH" if os.name == "nt" else "/workspace")

HF_HOME = os.environ.get("HF_HOME") or f"{ROOT}/hf_cache"
os.environ.setdefault("HF_HOME", HF_HOME)

CKPT_DIR = f"{ROOT}/checkpoints"
SAVE_EVERY = 200
EVAL_EVERY = 200
EVAL_BATCHES = 50

def describe() -> str:
    return "\n".join([
        f"  model      {MODEL_NAME}",
        f"  data       {SAMPLE_SIZE:,} icons, val {VAL_SIZE:,}",
        f"  max length {MAX_LENGTH}",
        f"  batch      {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}",
        f"  epochs     {EPOCHS}",
        f"  root       {ROOT}",
    ])

if __name__ == "__main__":
    print(describe())