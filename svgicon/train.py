import math
import os
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from . import checkpoint as ckpt
from . import config
from . import data as datamod
from . import model as modelmod
from .vocab import VocabBridge


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cosine_schedule(optimizer, total_steps, warmup_steps):
    """Create a scheduler with linear warmup and cosine decay.

    Parameters:
        optimizer: Optimizer to update.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.

    Returns:
        Learning-rate scheduler.
    """
    def scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)

def sanity_check(model, loader, bridge):
    """Check the batch structure and verify the SVG tokens.

    Parameters:
        model: Model to check.
        loader: Training data loader.
        bridge: SVG vocabulary bridge.

    Returns:
        Initial loss for the checked batch.
    """
    batch = next(iter(loader))
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    positions = [i for i, v in enumerate(ids) if bridge.is_svg_id(v)]
    first = bridge.to_svg.get(ids[positions[0]])
    last = bridge.to_svg.get(ids[positions[-1]])

    assert first == "<BOS>", f"expected <BOS>, got {first}"
    assert last == "<EOS>", f"expected <EOS>, got {last} - truncation!"
    assert all(v == -100 for v in labels[:positions[0]]), "prompt not masked"

    with torch.no_grad():
        loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss.item()

    print(f"  shape {tuple(batch['input_ids'].shape)}   "
          f"prompt {positions[0]}   svg {len(positions)}")
    print(f"  starts {first}  ends {last}  prompt masked   OK")
    print(f"  initial loss {loss:.3f}")
    return loss

@torch.no_grad()
def evaluate(model, loader, max_batches):
    """Calculate the average loss on a validation set.

    Parameters:
        model: Model to evaluate.
        loader: Validation data loader.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        Average validation loss.
    """
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        count += 1
    model.train()
    return total / max(1, count)

def main():
    # Set up the model, train it, and save checkpoints
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    set_seed(config.SEED)

    assert torch.cuda.is_available(), "no GPU"
    print(torch.cuda.get_device_name(0))
    print(config.describe())

    # setup
    hf = modelmod.load_tokenizer()
    bridge = VocabBridge(hf, config.MODEL_NAME)
    print(f"\n{bridge.describe()}")

    (train_loader, val_loader), train_ex, val_ex = datamod.load_all(hf, bridge)

    model, optimizer, info = modelmod.build(bridge)

    print(f"\nresized to {info['total_vocab']:,}   "
        f"tied {info['tied']}   {info['optimizer']}")

    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_loader) / config.GRAD_ACCUM)
    total_steps = steps_per_epoch * config.EPOCHS
    warmup_steps = max(20, int(config.WARMUP_FRAC * total_steps))
    scheduler = cosine_schedule(optimizer, total_steps, warmup_steps)

    print(f"\n{len(train_loader):,} batches/epoch -> {steps_per_epoch:,} steps")
    print(f"{total_steps:,} total steps, {warmup_steps} warmup\n")

    print("sanity check")
    sanity_check(model, train_loader, bridge)

    # train
    def save(tag, step, history, with_optimizer=False):
            return ckpt.save(
                os.path.join(config.CKPT_DIR, tag), model, hf, bridge,
                info["total_vocab"], step, history,
                optimizer if with_optimizer else None,
                scheduler if with_optimizer else None,
            )
    
    history, step, best_val = [], 0, float("inf")
    model.train()
    started = time.time()

    for epoch in range(config.EPOCHS):
        running, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.EPOCHS}")

        for i, batch in enumerate(bar):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / config.GRAD_ACCUM).backward()

            running += loss.item()
            seen += 1

            is_last = (i + 1) == len(train_loader)
            if (i + 1) % config.GRAD_ACCUM == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % config.EVAL_EVERY == 0:
                    val = evaluate(model, val_loader, config.EVAL_BATCHES)
                    train_avg = running / max(1, seen)
                    history.append({"step": step, "train": train_avg, "val": val})
                    running, seen = 0.0, 0

                    tqdm.write(
                        f"  step {step:>5}  train {train_avg:6.3f}  "
                        f"val {val:6.3f}  "
                        f"lr {scheduler.get_last_lr()[0]:.2e}  "
                        f"{(time.time() - started) / 60:.0f}m")

                    if val < best_val:
                        best_val = val
                        size = save("best", step, history)
                        tqdm.write(f"  new best ({size:.0f} MB)")

                if step % config.SAVE_EVERY == 0:
                    save("last", step, history, with_optimizer=True)

            bar.set_postfix(loss=f"{running / max(1, seen):.3f}", step=step)

        save(f"epoch{epoch + 1}", step, history)

    final = evaluate(model, val_loader, 10 ** 9)
    print(f"\nfinal val loss (full set): {final:.4f}")
    save("final", step, history)

if __name__ == "__main__":
    main()