import json
import os

import torch

from peft import PeftModel
from transformers import AutoTokenizer

from . import config
from .model import load_model
from .tokenizer import VOCAB
from .vocab import VocabBridge

def save(path, model, hf_tokenizer, bridge, total_vocab, step, history, optimizer=None, scheduler=None):
    """
    Save the trained model components and checkpoint metadata.

    Saves the LoRA adapter, Hugging Face tokenizer, trained SVG embedding
    rows, and metadata. If an optimizer is provided, its state and the
    optional scheduler state are saved for training resumption.

    Only the SVG embedding rows are saved separately because the
    pretrained rows are intended to be restored from the base checkpoint.

    Parameters:
        path: Directory where the checkpoint files will be saved.
        model: Trained language model with LoRA adapters attached.
        hf_tokenizer: Hugging Face tokenizer used during training.
        bridge: Vocabulary bridge containing the SVG token ID range.
        total_vocab: Total number of rows in the resized vocabulary.
        step: Current training step.
        history: Training history to store in the metadata.
        optimizer: Optional optimizer whose state should be saved.
        scheduler: Optional learning-rate scheduler to save.

    Returns:
        Total size of the files in the checkpoint directory, in MiB.
    """

    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path, save_embedding_layers=False)
    hf_tokenizer.save_pretrained(path)

    embed_in = model.get_input_embeddings()
    embed_out = model.get_output_embeddings()

    torch.save(embed_in.weight[bridge.lo:bridge.hi].detach().clone().cpu(), os.path.join(path, "svg_embeddings.pt"))

    if embed_out.weight.data_ptr() != embed_in.weight.data_ptr():
        torch.save(embed_out.weight[bridge.lo:bridge.hi].detach().clone().cpu(), os.path.join(path, "svg_lm_head.pt"))

    meta = {
        "model_name": config.MODEL_NAME,
        "base_vocab": bridge.base_vocab,
        "svg_id_lo": bridge.lo,
        "svg_id_hi": bridge.hi,
        "total_vocab": total_vocab,
        "step": step,
        "history": history,
    }

    with open(os.path.join(path, "meta.json"), "w") as handle:
        json.dump(meta, handle)

    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "step": step},
                   os.path.join(path, "optim.pt"))

    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1024 ** 2


def load(path, device_map="auto"):
    """
    Load a trained model for SVG generation.

    Loads the saved tokenizer, restores the SVG embeddings, and attaches
    the LoRA adapters to the base model.

    Parameters:
        path: Checkpoint directory.
        device_map: Device placement setting (default: "auto").

    Returns:
        model: Trained model ready for generation.
        hf_tokenizer: Hugging Face tokenizer.
        bridge: SVG vocabulary bridge.
        meta: Checkpoint metadata.
    """

    with open(os.path.join(path, "meta.json")) as handle:
        meta = json.load(handle)

    hf_tokenizer = AutoTokenizer.from_pretrained(path)

    base = load_model()
    base.resize_token_embeddings(meta["total_vocab"])
    base.config.vocab_size = meta["total_vocab"]

    lo, hi = meta["svg_id_lo"], meta["svg_id_hi"]

    with torch.no_grad():
        rows = torch.load(os.path.join(path, "svg_embeddings.pt"))
        weight = base.get_input_embeddings().weight
        weight[lo:hi] = rows.to(weight.device, weight.dtype)

        head_path = os.path.join(path, "svg_lm_head.pt")
        if os.path.exists(head_path):
            rows = torch.load(head_path)
            weight = base.get_output_embeddings().weight
            weight[lo:hi] = rows.to(weight.device, weight.dtype)
        elif base.get_output_embeddings().weight.data_ptr() != \
                base.get_input_embeddings().weight.data_ptr():
            raise RuntimeError(
                "checkpoint has no svg_lm_head.pt but this model has "
                "untied embeddings - the output layer would be random")

    model = PeftModel.from_pretrained(base, path)
    model.eval()
    model.config.use_cache = True # re-enable for generation

    bridge = VocabBridge(hf_tokenizer, meta["model_name"])

    # a changed tokenizer would silently shift every token ID
    if bridge.svg_vocab != len(VOCAB):
        raise RuntimeError("tokenizer vocabulary differs from the checkpoint")

    return model, hf_tokenizer, bridge, meta