"""Number formatting and the macro writer for make_numbers.py."""
from __future__ import annotations


def f4(x: float) -> str:
    return f"{x:.4f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def sgn4(x: float) -> str:
    """Signed, with a true minus sign in math mode."""
    s = f"{x:+.4f}"
    return ("$-$" + s[1:]) if s[0] == "-" else s


def intc(x: int) -> str:
    """Integer with a thin-space-free LaTeX thousands comma."""
    return f"{int(x):,}".replace(",", "{,}")


class Out:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.names: set[str] = set()
        self.count = 0

    def c(self, text: str) -> None:
        self.lines.append(f"% {text}")

    def m(self, name: str, value: str) -> None:
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        self.names.add(name)
        self.count += 1
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"
