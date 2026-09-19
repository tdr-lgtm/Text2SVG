from svgicon.tokenizer import (
    color_to_token,
    token_to_color,
    coord_to_bin,
    bin_to_coord,
    parse_path,
    flatten_arcs,
    tokenize,
    decode,
    roundtrip_ok,
    encode,
    decode_ids,
    VOCAB,
    ARGS,
    TOKEN_TO_ID,
)

SAMPLE = ('<svg viewBox="0.0 0.0 200.0 200.0">'
            '<path fill="#3EA050" fill-opacity="1.0" filling="0" '
            'd="M50 20 L150 20 A30 30 0 0 1 180 50 Z"/>'
            '<path fill="currentColor" d="M10 10 L20 20 Z"/>'
            '<path fill="" d="M5 5 L8 8 Z"/></svg>')

print("color")
for c in ["#3EA050", "#FF0000", "#FFF", "#000000"]:
    t = color_to_token(c)
    print(f"  {c:<10} -> {t:<5} -> {token_to_color(t)}")

print("\ncoordinates")
for v in [0, 50, 102.47, 200, -5, 250]:
    b = coord_to_bin(v)
    print(f"  {v:>8} -> {b:>3} -> {bin_to_coord(b):.2f}")

print("\nparse, with argument-count check")
for cmd, nums in parse_path("M 100 50 L 160 90 C 20 30 40 50 60 70 Z"):
    ok = "ok" if len(nums) == ARGS.get(cmd.upper(), -1) else "WRONG"
    print(f"  {cmd}  {nums}  {ok}")

print("\nexponent numbers")
for cmd, nums in parse_path("M 1.5e-5 2E3 L -1e2 0"):
    print(f"  {cmd}  {nums}")

print("\narc flattening")
commands = parse_path("M 50 20 L 150 20 A 30 30 0 0 1 180 50 Z")
print("  before:", [(c, len(n)) for c, n in commands])
print("  after: ", [(c, len(n)) for c, n in flatten_arcs(commands)])

print("\ntokenize")
for t in tokenize(SAMPLE):
    print("  ", t)

print("\ndecode")
print("  ", decode(tokenize(SAMPLE)))

print("\nround trip")
ok, _, _ = roundtrip_ok(SAMPLE)
print("  stable:", ok)

print("\nvocabulary")
print(f"  size {len(VOCAB):,}")
print(f"  first 12 {VOCAB[:12]}")

toks = tokenize(SAMPLE)
ids = encode(toks)
print(f"  ids round trip: {decode_ids(ids) == toks}")
print(f"  missing from vocab: {[t for t in toks if t not in TOKEN_TO_ID]}")