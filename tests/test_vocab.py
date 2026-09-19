from transformers import AutoTokenizer

from svgicon.tokenizer import tokenize
from svgicon.vocab import VocabBridge

MODEL = "Qwen/Qwen2.5-7B"

hf = AutoTokenizer.from_pretrained(MODEL)
bridge = VocabBridge(hf, MODEL)

print(bridge.describe())
print(f"len(tokenizer) = {len(hf):,}")

svg = ('<svg viewBox="0 0 200 200">'
        '<path fill="#3EA050" d="M50 20 L150 20 Z"/></svg>')

tokens = tokenize(svg)
ids = bridge.encode(tokens)

print(f"\ntokens {tokens[:5]}")
print(f"ids    {ids[:5]}")
print(f"round trip:   {bridge.decode(ids) == tokens}")
print(f"all in range: {all(bridge.is_svg_id(i) for i in ids)}")

"""
base vocab 152,064   svg vocab 4,621   range [152,064, 156,685)
len(tokenizer) = 151,665

tokens ['<BOS>', '<PATH>', 'FILL', 'COLOR_933', 'D']
ids    [152065, 152068, 152070, 153521, 152071]
round trip:   True
all in range: True
"""