import re
from .arc import arc_to_cubics

COLOR_LEVELS = 16
COLOR_COUNT = COLOR_LEVELS ** 3

# path data
# Arguments each command takes, used to validate the parser.
ARGS = {'M': 2, 'L': 2, 'C': 6, 'A': 7, 'Z': 0}

# After flattening only these remain.
ARG_COUNT = {'M': 2, 'L': 2, 'C': 6, 'Z': 0}

# The canvas is 200 × 200 units.
# Values above 10,000 are treated as suspicious data.
# One SVG out of 20,000 had an extremely large arc radius (1.5e+16).
MAX_VALUE = 1e4

COMMAND_RE = re.compile(
    r'([MmZzLlHhVvCcSsQqTtAa])([^MmZzLlHhVvCcSsQqTtAa]*)')

# Without the exponent part, 1.5e-5 could be read as 1.5 and -5.
# This would make the SVG commands use the wrong values.
NUMBER_RE = re.compile(
    r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'
)

# tokenize / decode
PATH_RE = re.compile(r'<path\b[^>]*/?>', re.I)
ATTR_RE = re.compile(r'(\w[\w-]*)\s*=\s*"([^"]*)"')
HEX_RE = re.compile(r'#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?$')

def color_to_token(hex_color: str) -> int:
    """
    Convert a hexadecimal RGB color into a single integer token.

    Parameters:
        hex_color (str): A hexadecimal RGB color string.
            
    Returns:
        int: A unique integer from 0 to 4095 representing the reduced RGB color.
    """

    color = hex_color.lstrip("#")

    # FFF -> #FFFFFF
    if len(color) == 3:
        color = "".join(c * 2 for c in color)

    # Convert each hexadecimal RGB channel to decimal,
    # then reduce it from 8 bits to 4 bits (0–15).
    r, g, b = (
        int(color[i:i + 2], 16) // COLOR_LEVELS
        for i in (0, 2, 4)
    )

    # Pack the three 4-bit channel values into one unique integer using base-16 place values.
    return r * COLOR_LEVELS ** 2 + g * COLOR_LEVELS + b

def token_to_color(token: int) -> str:
    """
    Convert integer token back into a hexadecimal RGB color.

    Parameters:
        token (int): An integer is a reduced RGB color.

    Returns:
        str: A 6-digit hexadecimal RGB color string.
    """

    r = token // COLOR_LEVELS ** 2
    g = (token // COLOR_LEVELS) % COLOR_LEVELS
    b = token % COLOR_LEVELS

    r, g, b = r * 17, g * 17, b * 17
    
    return f"#{r:02x}{g:02x}{b:02x}"

# coordinates
CANVAS = 200.0
BINS = 256

def coord_to_bin(value: float) -> int:
    """
    Convert a canvas coordinate into a discrete bin index.

    Parameters:
        value (float): A canvas coordinate.

    Returns:
        int: A bin index in the range 0–255.
    """
    # 102.47 -> 131
    index = round(value / CANVAS * (BINS - 1))
    # clamp to 0..255
    return max(0, min(BINS - 1, index))

def bin_to_coord(index: int) -> float:
    """
        Convert a discrete bin index into a canvas coordinate.
    
        Parameters:
            value (int): A bin index in the range 0–255.
    
        Returns:
            float: A canvas coordinate.
    """
    # 131 -> 102.75
    return index / (BINS - 1) * CANVAS

def parse_path(d: str) -> list:
    """
    M 100 50 L 160 90 Z' -> [('M', [100, 50]), ('L', [160, 90]), ('Z', [])]

    Parameters:
        d: SVG path data as a string.

    Returns:
        A list of tuples.
    """
    result = []

    for command, values in COMMAND_RE.findall(d):
        numbers = [float(x) for x in NUMBER_RE.findall(values)]
        result.append((command, numbers))

    return result

def flatten_arcs(commands: list) -> list:
    """
    Replace every A with the cubics that draw the same curve.

    Parameters:
        commands: A list of SVG commands and their values.

    Returns:
        A new list with A commands replaced by C commands.
        Other commands are kept, while the current point is tracked to find where each arc starts.
    """
    result = []

    x = y = 0.0
    sx = sy = 0.0

    for command, numbers in commands:

        if command == 'M':
            x, y = numbers[0], numbers[1]
            sx, sy = x, y
            result.append(('M', numbers))

        elif command == 'L':
            x, y = numbers[0], numbers[1]
            result.append(('L', numbers))

        elif command == 'C':
            x, y = numbers[4], numbers[5]
            result.append(('C', numbers))

        elif command == 'Z':
            x, y = sx, sy
            result.append(('Z', []))

        elif command == 'A':
            rx, ry, rotation, large_arc, sweep, ex, ey = numbers

            for cubic in arc_to_cubics(x, y, rx, ry, rotation,
                                       large_arc, sweep, ex, ey):
                result.append(('C', list(cubic)))

            x, y = ex, ey

    return result

def tokenize(svg: str) -> list | None:
    """
    Parameters:
        svg: An SVG file as a string.

    Returns:
        A list of tokens representing the SVG.
        Returns None if the file is unusable.
    """

    tokens = ["<BOS>"]

    for tag in PATH_RE.findall(svg):
        attrs = dict(ATTR_RE.findall(tag))

        d = attrs.get("d", "")
        if not d:
            continue

        commands = parse_path(d)

        # corrupt file
        if any(abs(n) > MAX_VALUE for _, nums in commands for n in nums):
            return None

        commands = flatten_arcs(commands)
        if not commands:
            continue

        tokens.append("<PATH>")

        # If fill is empty or missing, do not add a color token.
        fill = attrs.get("fill", "")
        if fill == "currentColor":
            tokens.extend(["FILL", "COLOR_CURRENT"])
        elif HEX_RE.match(fill):
            tokens.extend(["FILL", f"COLOR_{color_to_token(fill)}"])

        tokens.append("D")
        for command, numbers in commands:
            tokens.append(f"PATH_{command}")
            # after flattening every number is a coordinate, so they pair
            for i in range(0, len(numbers), 2):
                tokens.append(f"X_{coord_to_bin(numbers[i])}")
                tokens.append(f"Y_{coord_to_bin(numbers[i + 1])}")

        tokens.append("</PATH>")

    tokens.append("<EOS>")
    return tokens if len(tokens) > 2 else None

def decode(tokens: list) -> str:
    """
    Parameters:
        tokens: A list of tokens representing an SVG.

    Returns:
        An SVG string reconstructed from the tokens.
        Converts coordinate bins back into values and restores
        path commands and fill colors.
    """
    paths = []
    fill = None
    parts = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "<PATH>":
            fill, parts = None, []

        elif token == "FILL":
            i += 1
            if i < len(tokens):
                value = tokens[i]
                if value == "COLOR_CURRENT":
                    fill = "currentColor"
                elif value.startswith("COLOR_"):
                    fill = token_to_color(int(value[6:]))

        elif token.startswith("PATH_"):
            letter = token[5:]
            n = ARG_COUNT.get(letter, 0)

            numbers = []
            for k in range(n):
                i += 1
                prefix = "X_" if k % 2 == 0 else "Y_"
                if i < len(tokens) and tokens[i].startswith(prefix):
                    numbers.append(bin_to_coord(int(tokens[i][2:])))
                # malformed, stop here
                else:
                    i -= 1
                    break

            if numbers:
                parts.append(letter + " "
                             + " ".join(f"{v:.2f}" for v in numbers))
            elif n == 0:
                parts.append(letter)

        elif token == "</PATH>":
            if parts:
                attr = f' fill="{fill}"' if fill is not None else ""
                paths.append(f'<path{attr} d="{" ".join(parts)}"/>')

        i += 1

    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {CANVAS:.0f} {CANVAS:.0f}">{"".join(paths)}</svg>')

def roundtrip_ok(svg: str):
    """
    Parameters:
        svg: An SVG file as a string.

    Returns:
        A tuple containing:
        - True if the original and final tokens match, otherwise False.
        - The tokens from the original SVG.
        - The tokens after decoding and tokenizing again.

        Returns (None, None, None) if the SVG cannot be tokenized.
    """
    first = tokenize(svg)
    if first is None:
        return None, None, None
    return (first == tokenize(decode(first))), first, tokenize(decode(first))

def build_vocab() -> list:
    """
    Returns:
        A list of all possible tokens in a fixed order.
        Each token's position in the list is its ID.
        The order must stay the same so trained models
        continue to use the correct token IDs.
    """
    vocab = []

    vocab += ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]           # specials
    vocab += ["<PATH>", "</PATH>", "FILL", "D"]             # structural
    vocab += ["PATH_M", "PATH_L", "PATH_C", "PATH_Z"]       # commands

    vocab += [f"X_{i}" for i in range(BINS)]
    vocab += [f"Y_{i}" for i in range(BINS)]

    vocab += [f"COLOR_{i}" for i in range(COLOR_COUNT)]
    vocab += ["COLOR_CURRENT"]

    return vocab

VOCAB = build_vocab()
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for i, t in enumerate(VOCAB)}

def encode(tokens: list) -> list:
    """
    Parameters:
        tokens: A list of tokens.

    Returns:
        A list of token IDs. Unknown tokens are replaced with
        the ID of <UNK>.
    """
    unknown = TOKEN_TO_ID["<UNK>"]
    return [TOKEN_TO_ID.get(t, unknown) for t in tokens]


def decode_ids(ids: list) -> list:
    """
    Parameters:
        ids: A list of token IDs.

    Returns:
        A list of tokens. Unknown IDs are replaced with <UNK>.
    """
    return [ID_TO_TOKEN.get(i, "<UNK>") for i in ids]
