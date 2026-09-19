import random

import torch
from torch.utils.data import DataLoader, Sampler

from . import config
from .tokenizer import tokenize

from datasets import load_dataset

class LengthBucketSampler(Sampler):
    """
    Group sequences of similar lengths into batches and optionally shuffle the batches to reduce padding.

    Parameters:
        examples: A list of examples, each containing input_ids.
        batch_size: The maximum number of examples in each batch.
        shuffle: Whether to shuffle the order of the batches.

    Methods:
        __iter__: Return an iterator over the batches.
        __len__: Return the total number of batches.
    """

    def __init__(self, examples, batch_size, shuffle=True):
        order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
        self.batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        self.shuffle = shuffle

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.batches)

def make_collate_fn(pad_id):
    """
    Create a collate function that pads sequences in a batch to the same length.

    Parameters:
        pad_id: The token ID used for padding.

    Returns:
        A collate function that returns input_ids, labels, and attention_mask
        as PyTorch tensors.
    """
    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
    return collate

def make_loaders(train_examples, val_examples, pad_id):
    """
    Create DataLoaders for training and validation.

    Parameters:
        train_examples: The training examples.
        val_examples: The validation examples.
        pad_id: The token ID used for padding.

    Returns:
        A tuple containing the training and validation DataLoaders.
    """
    collate = make_collate_fn(pad_id)

    train_loader = DataLoader(
        train_examples,
        batch_sampler=LengthBucketSampler(train_examples, config.BATCH_SIZE, shuffle=True),
        collate_fn=collate, 
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_examples,
        batch_sampler=LengthBucketSampler(val_examples, config.BATCH_SIZE, shuffle=False),
        collate_fn=collate, 
        pin_memory=True,
    )
    return train_loader, val_loader

def build_examples(dataset, hf_tokenizer, bridge, name="", verbose=True):
    """
    Convert dataset rows into examples containing input_ids and labels.

    The input contains the caption followed by the SVG tokens.
    The caption labels are masked with -100 so the loss is calculated
    only on the SVG tokens.

    Parameters:
        dataset: The dataset containing captions and SVG strings.
        hf_tokenizer: The Hugging Face tokenizer used to encode captions.
        bridge: The vocabulary bridge used to encode SVG tokens.
        name: A name displayed in the processing summary.
        verbose: Whether to print the processing summary.

    Returns:
        A list of examples, each containing input_ids and labels.
        Examples with unusable SVGs or sequences exceeding MAX_LENGTH
        are skipped.
    """
    kept = 0
    too_long = 0
    unusable = 0
    examples = []

    for row in dataset:
        svg_tokens = tokenize(row["svg"])

        # corrupt file
        if svg_tokens is None:
            unusable += 1
            continue

        prompt_ids = hf_tokenizer(
            config.PROMPT_TEMPLATE.format(
                description=row[config.CAPTION_FIELD]),
                add_special_tokens=True,
                truncation=True,
                max_length=config.PROMPT_BUDGET,
            )["input_ids"]

        svg_ids = bridge.encode(svg_tokens)

        # Skip examples that exceed the maximum sequence length.
        if len(prompt_ids) + len(svg_ids) > config.MAX_LENGTH:
            too_long += 1
            continue

        examples.append({
            "input_ids": prompt_ids + svg_ids,
            "labels": [-100] * len(prompt_ids) + svg_ids,
        })
        kept += 1

    if verbose:
        total = max(1, kept + too_long + unusable)
        print(f"{name:<6} kept {kept:>7,}   too long {too_long:>5,}   "
              f"unusable {unusable:>4,}   ({100 * kept / total:.1f}% retained)")

    return examples

def load_all(hf_tokenizer, bridge):
    """
    Load the dataset, split it into training and validation sets,
    convert the data into model-ready examples, and create DataLoaders.

    Parameters:
        hf_tokenizer: The Hugging Face tokenizer used to encode prompts.
        bridge: The vocabulary bridge used to encode SVG tokens.

    Returns:
        A tuple containing:
            - The training and validation DataLoaders.
            - The processed training examples.
            - The processed validation examples.
    """

    ds = load_dataset(config.DATASET, split=config.SPLIT)
    ds = ds.shuffle(seed=config.SEED).select(range(config.SAMPLE_SIZE))

    split = ds.train_test_split(test_size=config.VAL_SIZE, seed=config.SEED)

    train = build_examples(split["train"], hf_tokenizer, bridge, "train")
    val = build_examples(split["test"], hf_tokenizer, bridge, "val")

    if len(train) < 1000:
        raise RuntimeError(
            f"only {len(train)} usable examples - raise MAX_LENGTH "
            f"or check the tokenizer")

    return make_loaders(train, val, bridge.pad_id), train, val


if __name__ == "__main__":
    examples = [
    {"input_ids": [10, 11, 12], "labels": [11, 12, 2]},
    {"input_ids": [20, 21, 22, 23, 2], "labels": [21, 22, 23, 2, -100]},
    {"input_ids": [30, 31], "labels": [31, 2]},
    {"input_ids": [40, 41, 42, 2], "labels": [41, 42, 2, -100]},
]

    batch_size = 2
    pad_id = 0

    sampler = LengthBucketSampler(
        examples,
        batch_size=batch_size,
        shuffle=False
    )

    print("Batches of indices:", list(sampler))

    collate = make_collate_fn(pad_id)
    batch = collate([examples[2], examples[0]])

    print("Input IDs:")
    print(batch["input_ids"])

    print("Labels:")
    print(batch["labels"])

    print("Attention mask:")
    print(batch["attention_mask"])