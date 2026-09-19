from transformers import AutoConfig, AutoTokenizer

from .tokenizer import VOCAB, tokenize

class VocabBridge:
    """
    Connect the SVG vocabulary to the model's vocabulary by assigning each SVG token its own token ID.

    Parameters:
        hf_tokenizer: The Hugging Face tokenizer.
        model_name: The name of the pretrained model.

    Methods:
        encode: Convert SVG tokens into model token IDs.
        decode: Convert SVG token IDs back into SVG tokens.
        is_svg_id: Check whether a token ID belongs to the SVG vocabulary.
        describe: Return a summary of the vocabulary sizes and ID range.
    """
    def __init__(self, hf_tokenizer, model_name):
        config = AutoConfig.from_pretrained(model_name)
        vocab_size = getattr(config, "vocab_size", None)

        self.base_vocab = max(len(hf_tokenizer), vocab_size)
        self.svg_vocab = len(VOCAB)

        self.lo = self.base_vocab
        self.hi = self.base_vocab + self.svg_vocab

        self.to_model = {t: self.lo + i for i, t in enumerate(VOCAB)}
        self.to_svg = {v: k for k, v in self.to_model.items()}

        self.pad_id = self.to_model["<PAD>"]
        self.bos_id = self.to_model["<BOS>"]
        self.eos_id = self.to_model["<EOS>"]
        self.unk_id = self.to_model["<UNK>"]

    def encode(self, tokens: list) -> list:
        return [self.to_model.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: list) -> list:
        return [self.to_svg[i] for i in ids if i in self.to_svg]

    def is_svg_id(self, token_id: int) -> bool:
        return self.lo <= token_id < self.hi
 
    def describe(self) -> str:
        return (f"base vocab {self.base_vocab:,}   "
                f"svg vocab {self.svg_vocab:,}   "
                f"range [{self.lo:,}, {self.hi:,})")