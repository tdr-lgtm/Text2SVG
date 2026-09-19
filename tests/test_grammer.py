from svgicon.tokenizer import  tokenize
from svgicon.grammer import SVGGrammar

SAMPLES = {
    "simple": ('<svg viewBox="0 0 200 200">'
                '<path fill="#3EA050" d="M50 20 L150 20 Z"/></svg>'),
    "arc": ('<svg viewBox="0 0 200 200">'
            '<path fill="#FF0000" d="M50 20 L150 20 '
            'A30 30 0 0 1 180 50 Z"/></svg>'),
    "no fill": ('<svg viewBox="0 0 200 200">'
                '<path fill="" d="M5 5 L8 8 Z"/></svg>'),
    "currentColor": ('<svg viewBox="0 0 200 200">'
                        '<path fill="currentColor" d="M1 1 L9 9 Z"/></svg>'),
    "multi path": ('<svg viewBox="0 0 200 200">'
                    '<path fill="#111" d="M1 1 L9 9 Z"/>'
                    '<path fill="#222" d="M2 2 C3 3 4 4 5 5 Z"/></svg>'),
}

grammar = SVGGrammar()
ok = True

for name, svg in SAMPLES.items():
    good, at, err = grammar.validate(tokenize(svg))
    print(f"{'PASS' if good else 'FAIL'}  {name:<14}"
            + ("" if good else f" at {at}: {err}"))
    ok &= good

# the caps must fire
capped = SVGGrammar(max_path_segments=3)
capped.reset()
for t in ["<PATH>", "D", "PATH_M", "X_10", "Y_10"]:
    capped.step(t)
for _ in range(2):
    capped.step("PATH_L"); capped.step("X_10"); capped.step("Y_10")
assert capped.allowed() == {"</PATH>"}, capped.allowed()
print("\nsegment cap fires -> runaway paths unreachable")

budget = SVGGrammar()
for t in ["<BOS>", "<PATH>", "FILL", "COLOR_10", "D", "PATH_M",
            "X_10", "Y_10", "PATH_C", "X_11", "Y_11", "X_12"]:
    budget.step(t)

assert budget.allowed_with_budget(500) == budget.allowed()
steps, left = [], 8
while not budget.done:
    legal = budget.allowed_with_budget(left)
    assert legal.issubset(budget.allowed()), legal
    token = sorted(legal)[0]
    budget.step(token)
    steps.append(token)
    left -= 1
assert left >= 0, f"needed more than the reserve: {steps}"
print(f"budget cap closes the path in {len(steps)} tokens -> {steps}")

print("\nall pass" if ok else "\nFAILURES")