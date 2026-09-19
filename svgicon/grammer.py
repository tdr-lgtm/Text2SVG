from .tokenizer import BINS, COLOR_COUNT

# Measurements from 20k files: p50 16 segments, p99 95, max 163.
# A limit of 200 allows room for real paths while preventing runaway paths.
MAX_PATH_SEGMENTS = 200
MAX_PATHS = 40

# Reserve tokens to finish a sequence legally.
# In the worst case, a path needs 6 coordinates(C Bizier), </PATH>, and <EOS>.
# Two extra tokens provide a small safety margin.
RESERVE = 10

class SVGGrammar:
    def __init__(self, max_path_segments=MAX_PATH_SEGMENTS, max_paths=MAX_PATHS, reserve=RESERVE):
        self.max_path_segments = max_path_segments
        self.max_paths = max_paths
        self.reserve = reserve

        self.X = {f"X_{i}" for i in range(BINS)}
        self.Y = {f"Y_{i}" for i in range(BINS)}
        self.COLORS = {f"COLOR_{i}" for i in range(COLOR_COUNT)}
        self.COLORS.add("COLOR_CURRENT")

        self.PATH_CMDS = {"PATH_M", "PATH_L", "PATH_C", "PATH_Z"}

        self.reset()

    # State
    def reset(self):
        # Reset the grammar to its initial state.
        self.pending = []
        # Start
        self.mode = "start"
        self.path_open = False
        self.has_fill = False
        self.in_data = False
        self.path_started = False
        self.segments = 0
        self.paths = 0
        self.done = False

    def allowed(self) -> set:
        """Return the tokens allowed in the current grammar state.

        Returns:
            set: The set of tokens that are legal at the current position.
        """

        if self.done:
            return {"<EOS>"}

        if self.pending:
            return self.pending[0]

        if self.mode == "start":
            return {"<BOS>", "<PATH>"} if self.paths < self.max_paths else {"<EOS>"}

        if self.in_data:
            if not self.path_started:
                return {"PATH_M"}
            if self.segments >= self.max_path_segments:
                return {"</PATH>"}
            return self.PATH_CMDS | {"</PATH>"}

        if self.path_open:
            if not self.has_fill:
                # fill is optional
                return {"FILL", "D"}
            return {"D"}

        # between paths
        if self.paths >= self.max_paths:
            return {"<EOS>"}
        
        # force at least one path before allowing <EOS>
        if self.paths == 0:
            return {"<PATH>"}
        return {"<PATH>", "<EOS>"}
    
    def step(self, token: str):
        """Consume a token and update the grammar state.

        Parameters:
            token: The token to consume.

        Returns:
            self: The updated grammar object.
        """

        if token not in self.allowed():
            raise ValueError(f"illegal token {token!r} in mode {self.mode}")

        # Remove the next required token after it has been consumed.
        if self.pending:
            self.pending.pop(0)

        # marks the start, no state change
        if token == "<BOS>":
            pass

        elif token == "<PATH>":
            self.mode = "path"
            self.path_open = True
            self.has_fill = False
            self.in_data = False
            self.paths += 1

        elif token == "FILL":
            self.has_fill = True
            self.pending = [self.COLORS]

        elif token == "D":
            self.in_data = True
            self.path_started = False
            self.segments = 0

        elif token in ("PATH_M", "PATH_L"):
            self.pending = [self.X, self.Y]
            self.path_started = True
            self.segments += 1

        elif token == "PATH_C":
            self.pending = [self.X, self.Y] * 3
            self.segments += 1

        elif token == "PATH_Z":
            self.segments += 1

        elif token == "</PATH>":
            self.path_open = False
            self.in_data = False
            self.mode = "between"

        elif token == "<EOS>":
            self.done = True

        return self

    def validate(self, tokens: list):
        """Check whether a token sequence follows the grammar rules.

        Parameters:
            tokens: List of tokens to validate.

        Returns:
            A tuple containing:
                valid: Whether the sequence is valid.
                position: Index of the invalid token, or the sequence length if all tokens are valid.
                error: Error message if a token is invalid, otherwise None.
        """
        self.reset()
        for i, token in enumerate(tokens):
            if token in ("<BOS>", "<PAD>"):
                continue
            try:
                self.step(token)
            except ValueError as error:
                return False, i, str(error)
        return True, len(tokens), None
    
    def allowed_with_budget(self, remaining: int) -> set:
        """Return allowed tokens while respecting the remaining token budget.

        Parameters:
            remaining (int): Number of tokens left to generate.

        Returns:
            set: The legal tokens for the current state. When the remaining
                budget reaches the reserve, the grammar forces the shortest
                legal route toward EOS.
        """

        # Use the normal grammar rules if the sequence is finished or
        # the remaining budget is above the reserve.
        if self.done or remaining > self.reserve:
            return self.allowed()
        
        if self.pending:
            return self.pending[0]

        if self.mode == "start":
            return self.allowed()

        if self.in_data:
            if not self.path_started:
                return {"PATH_M"}
            return {"</PATH>"}

        if self.path_open:
            return {"D"}

        if self.paths == 0:
            return {"<PATH>"}

        return {"<EOS>"}
