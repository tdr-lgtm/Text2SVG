from transformers import AutoTokenizer

from svgicon import config
from svgicon.vocab import VocabBridge
from svgicon.data import load_all

hf = AutoTokenizer.from_pretrained(config.MODEL_NAME)
bridge = VocabBridge(hf, config.MODEL_NAME)

print(config.describe())
print()

(train_loader, val_loader), train, val = load_all(hf, bridge)

lengths = sorted(len(e["input_ids"]) for e in train)
def p(q): return lengths[int(q * len(lengths))]

print(f"\nsequence length   p50 {p(.5)}  p90 {p(.9)}  "
        f"p99 {p(.99)}  max {lengths[-1]}")
print(f"batches           train {len(train_loader):,}  "
        f"val {len(val_loader):,}")

batch = next(iter(train_loader))
print(f"\nbatch shape       {tuple(batch['input_ids'].shape)}")

# the checks that catch silent corruption
ids = batch["input_ids"][0].tolist()
labels = batch["labels"][0].tolist()
svg_positions = [i for i, v in enumerate(ids) if bridge.is_svg_id(v)]

first = bridge.to_svg.get(ids[svg_positions[0]])
last = bridge.to_svg.get(ids[svg_positions[-1]])

print(f"prompt tokens     {svg_positions[0]}")
print(f"svg tokens        {len(svg_positions)}")
print(f"starts            {first}")
print(f"ends              {last}")

assert first == "<BOS>", f"expected <BOS>, got {first}"
assert last == "<EOS>", f"expected <EOS>, got {last} - truncation!"
assert all(v == -100 for v in labels[:svg_positions[0]]), "prompt not masked"
print("\nall checks passed")

"""
  model      Qwen/Qwen2.5-7B
  data       50,000 icons, val 1,000
  max length 1000
  batch      8 x 2 = 16
  epochs     2
  root       PATH

train  kept  48,953   too long     0   unusable   47   (99.9% retained)
val    kept   1,000   too long     0   unusable    0   (100.0% retained)

sequence length   p50 264  p90 492  p99 645  max 815
batches           train 6,120  val 125

batch shape       (8, 344)
prompt tokens     24
svg tokens        320
starts            <BOS>
ends              <EOS>

all checks passed
"""