import argparse
import os

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from . import config
from .grammer import SVGGrammar
from .tokenizer import decode, VOCAB
from . import checkpoint as ckpt

DEFAULT_PROMPTS = [
    "A black coffee cup icon on a white background.",
    "A simple blue arrow pointing to the right.",
    "A yellow star icon with five points.",
    "A green checkmark inside a circle.",
]

class SVGConstraint(LogitsProcessor):
    """
    Allow the model to generate only tokens permitted by the SVG grammar.

    Parameters:
        bridge: Maps SVG tokens to model token IDs and back.
        prompt_length (int): Number of tokens in the input prompt.
        batch_size (int): Number of sequences being generated.
        max_new_tokens (int, optional): Maximum number of tokens to generate.
    """

    def __init__(self, bridge, prompt_length, batch_size, max_new_tokens=None):
        self.bridge = bridge
        self.machines = [SVGGrammar() for _ in range(batch_size)]
        for machine in self.machines:
            machine.reset()
        self.seen = [prompt_length] * batch_size
        self.prompt_length = prompt_length
        self.max_new_tokens = max_new_tokens

    def __call__(self, input_ids, scores):
        generated = input_ids.shape[1] - self.prompt_length
        remaining = (None if self.max_new_tokens is None else self.max_new_tokens - generated)

        for row in range(input_ids.shape[0]):
            machine = self.machines[row]

            # consume anything generated since the last call
            for position in range(self.seen[row], input_ids.shape[1]):
                token = self.bridge.to_svg.get(input_ids[row, position].item())
                if token in (None, "<BOS>", "<PAD>"):
                    continue
                try:
                    machine.step(token)
                except ValueError:
                    pass
            
            self.seen[row] = input_ids.shape[1]

            legal = (machine.allowed() if remaining is None else machine.allowed_with_budget(remaining))

            ids = [self.bridge.to_model[t] for t in legal if t in self.bridge.to_model]

            if not ids:
                continue

            mask = torch.full_like(scores[row], float("-inf"))
            index = torch.tensor(ids, device=scores.device)
            mask[index] = scores[row][index]
            scores[row] = mask

        return scores

@torch.no_grad()
def generate(model, hf_tokenizer, bridge, description,
            max_new_tokens=900, temperature=0.5, top_p=0.95,
            constrained=True, n=1, seed=None):
    """
    Generate SVGs for a given description.

    Parameters:
        model: The trained model used to generate tokens.
        hf_tokenizer: The Hugging Face tokenizer used to encode the prompt.
        bridge: Maps between SVG tokens and model token IDs.
        description (str): Description of the SVG to generate.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Controls randomness during sampling.
        top_p (float): Nucleus sampling threshold.
        constrained (bool): Whether to apply the SVG grammar constraints.
        n (int): Number of SVG sequences to generate.
        seed (int, optional): Random seed for reproducible generation.

    Returns:
        list[str]: The generated SVG strings. If decoding fails for a
        sequence, its result contains an error comment.
    """
    if seed is not None:
        torch.manual_seed(seed)

    prompt = config.PROMPT_TEMPLATE.format(description=description)
    inputs = hf_tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    processors = LogitsProcessorList()
    if constrained:
        processors.append(SVGConstraint(bridge, prompt_length, n, max_new_tokens))

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.15,
        no_repeat_ngram_size=12,
        num_return_sequences=n,
        eos_token_id=bridge.eos_id,
        pad_token_id=hf_tokenizer.pad_token_id,
        logits_processor=processors,
    )

    results = []
    for row in output:
        ids = [i for i in row[prompt_length:].tolist() if bridge.is_svg_id(i)]
        tokens = bridge.decode(ids)
        try:
            results.append(decode(tokens))
        except Exception as error:
            results.append(f"<!-- decode failed: {error} -->")
    return results

def render(svg_text, path, width=400):
    import cairosvg

    """Convert an SVG string into a PNG image.

    Parameters:
        svg_text (str): SVG markup to render.
        path (str): Output path for the PNG file.
        width (int): Width of the output image in pixels.

    Returns:
        str: Path to the generated PNG file.
    """
    cairosvg.svg2png(bytestring=svg_text.encode(), write_to=path, output_width=width, background_color="white")
    return path

def embedding_report(model, bridge, top=8):
    """
    Check whether coordinate bins are ordered in embedding space.

    Compares the cosine similarity of nearby and distant coordinate
    bins to see whether the model learned their ordering.

    Parameters:
        model: The trained model.
        bridge: Maps SVG tokens to model token IDs.
        top (int): Number of nearest tokens to display.

    Returns:
        None: Prints similarity scores and the nearest tokens to X_128.
    """
    rows = model.get_input_embeddings().weight[bridge.lo:bridge.hi]
    rows = torch.nn.functional.normalize(rows.detach().float(), dim=1)

    index = {t: i for i, t in enumerate(VOCAB)}

    def similarity(a, b):
        return torch.dot(rows[index[a]], rows[index[b]]).item()

    print("cosine similarity between coordinate bins")
    print(f"  X_128 vs X_129 (adjacent) {similarity('X_128','X_129'):+.3f}")
    print(f"  X_128 vs X_140 (near)     {similarity('X_128','X_140'):+.3f}")
    print(f"  X_128 vs X_250 (far)      {similarity('X_128','X_250'):+.3f}")
    print(f"  X_128 vs PATH_C           {similarity('X_128','PATH_C'):+.3f}")
    print("  adjacent should exceed near, which should exceed far")

    scores = rows @ rows[index["X_128"]]
    best = torch.topk(scores, top + 1).indices.tolist()[1:]
    print(f"\n  nearest to X_128: {[VOCAB[i] for i in best]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(config.CKPT_DIR))
    parser.add_argument("--prompt", action="append")
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--out", default="samples")
    parser.add_argument("--unconstrained", action="store_true")
    parser.add_argument("--from-train", action="store_true",
                        help="use real captions from the dataset")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    print(f"loading {args.ckpt}")
    model, hf, bridge, meta = ckpt.load(args.ckpt)
    print(f"step {meta['step']}")

    print()
    embedding_report(model, bridge)
    print()

    prompts = args.prompt
    if args.from_train:
        from datasets import load_dataset
        ds = load_dataset(config.DATASET, split="train[:200]")
        prompts = [ds[i][config.CAPTION_FIELD] for i in range(4)]
        print("captions from the training set:")
        for p in prompts:
            print(f"  {p[:90]}")
        print()
    if not prompts:
        prompts = DEFAULT_PROMPTS

    os.makedirs(args.out, exist_ok=True)

    for index, prompt in enumerate(prompts):
        print(f"[{index}] {prompt[:70]}")
        samples = generate(
            model, hf, bridge, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            constrained=not args.unconstrained,
            n=args.n,
        )

        for k, svg in enumerate(samples):
            stem = os.path.join(args.out, f"{index:02d}_{k}")
            with open(stem + ".svg", "w") as handle:
                handle.write(svg)

            paths = svg.count("<path")
            status = "ok"
            if not args.no_render:
                try:
                    render(svg, stem + ".png")
                except ImportError:
                    status = "cairosvg not installed"
                except Exception as error:
                    status = f"render failed: {type(error).__name__}"

            print(f"    {stem}.svg   {len(svg):>6} chars   "
                  f"{paths:>3} paths   {status}")

    print(f"\nwrote to {args.out}/")

if __name__ == "__main__":
    main()