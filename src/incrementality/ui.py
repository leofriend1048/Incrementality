"""LIFT — Branded UI system for the incrementality CLI.

Provides the visual identity, theming, and reusable components that give
LIFT its distinctive, app-like terminal experience.

Brand:
    Name:     LIFT
    Mark:     ▲
    Tagline:  Geo Incrementality Platform
    Palette:  Teal/cyan primary, green for positive, red for negative
"""

from __future__ import annotations

import importlib.metadata
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# ── Brand Identity ────────────────────────────────────────────────────────────

BRAND = "LIFT"
TAGLINE = "Geo Incrementality Platform"

try:
    VERSION = importlib.metadata.version("incrementality")
except importlib.metadata.PackageNotFoundError:
    VERSION = "0.3.0"

# ── Theme ─────────────────────────────────────────────────────────────────────

_THEME = Theme({
    # Brand colors
    "lift": "bold bright_cyan",
    "lift.dim": "dim cyan",
    "lift.mark": "bright_cyan",
    # Typography
    "heading": "bold white",
    "label": "bold",
    "value": "white",
    "muted": "dim",
    # Semantic
    "ok": "green",
    "warn": "yellow",
    "bad": "bold red",
    "accent": "bright_cyan",
})

console = Console(theme=_THEME)

# ── Glyphs ────────────────────────────────────────────────────────────────────

MARK = "▲"
CHECK = "✓"
CROSS = "✗"
DOT = "●"
WARN_ICON = "⚠"
BAR_FILL = "█"
BAR_EMPTY = "░"


# ── Banner ────────────────────────────────────────────────────────────────────

def banner() -> None:
    """Print the LIFT startup banner."""
    console.print()
    console.print(
        f"  [lift]{MARK}[/lift] [heading]{BRAND}[/heading]  "
        f"[muted]v{VERSION}[/muted]"
    )
    console.print(f"  [muted]{TAGLINE}[/muted]")
    console.print()


# ── Step Progress ─────────────────────────────────────────────────────────────

@contextmanager
def step(message: str):
    """Show a spinner while a step is running, then print a checkmark."""
    with console.status(
        f"  [lift.mark]{DOT}[/lift.mark] {message}",
        spinner="dots",
        spinner_style="bright_cyan",
    ):
        yield
    console.print(f"  [ok]{CHECK}[/ok] {message}")


def done(message: str) -> None:
    """Print a completed step (no spinner)."""
    console.print(f"  [ok]{CHECK}[/ok] {message}")


def fail(message: str) -> None:
    """Print a failed step."""
    console.print(f"  [bad]{CROSS}[/bad] {message}")


def warning(message: str) -> None:
    """Print a warning."""
    console.print(f"  [warn]{WARN_ICON}[/warn]  {message}")


def info(message: str) -> None:
    """Print a muted info line."""
    console.print(f"  [muted]{DOT} {message}[/muted]")


# ── Section Headers ───────────────────────────────────────────────────────────

def section(title: str, width: int = 52) -> None:
    """Print a branded section divider."""
    rule_len = max(width - len(title) - 1, 10)
    console.print()
    console.print(
        f"  [heading]{title}[/heading] [muted]{'─' * rule_len}[/muted]"
    )
    console.print()


# ── Score Bars ────────────────────────────────────────────────────────────────

def score_bar(value: float, max_value: float = 100, width: int = 20) -> str:
    """Return a visual bar like  ████████████░░░░░░░░."""
    ratio = min(max(value / max_value, 0.0), 1.0)
    filled = int(ratio * width)
    empty = width - filled
    if ratio >= 0.80:
        color = "ok"
    elif ratio >= 0.60:
        color = "warn"
    else:
        color = "bad"
    return (
        f"[{color}]{BAR_FILL * filled}[/{color}]"
        f"[muted]{BAR_EMPTY * empty}[/muted]"
    )


def trust_meter(score: float) -> None:
    """Print a visual trust score meter with verdict."""
    bar = score_bar(score)
    if score >= 80:
        verdict = "[ok]TRUSTWORTHY[/ok]"
    elif score >= 60:
        verdict = "[warn]MARGINAL[/warn]"
    else:
        verdict = "[bad]NOT TRUSTWORTHY[/bad]"
    console.print(
        f"    {bar}  [heading]{score:.0f}[/heading][muted]/100[/muted]"
        f"  {verdict}"
    )


# ── Badges ────────────────────────────────────────────────────────────────────

def pass_fail(passed: bool) -> str:
    """Return a styled PASS/FAIL string."""
    if passed:
        return f"[ok]{CHECK} PASS[/ok]"
    return f"[bad]{CROSS} FAIL[/bad]"


def quality_badge(
    value: float, good: float, ok_thresh: float, *, reverse: bool = False,
) -> str:
    """Return Good/OK/Poor badge.  reverse=True means lower is better."""
    if reverse:
        if value < good:
            return "[ok]Good[/ok]"
        if value < ok_thresh:
            return "[warn]OK[/warn]"
        return "[bad]Poor[/bad]"
    else:
        if value > good:
            return "[ok]Good[/ok]"
        if value > ok_thresh:
            return "[warn]OK[/warn]"
        return "[bad]Poor[/bad]"


def sig_badge(is_significant: bool, p_value: float) -> str:
    """Return a significance indicator."""
    if is_significant:
        return (
            f"[ok]{CHECK} SIGNIFICANT[/ok]  "
            f"[muted](p = {p_value:.4f})[/muted]"
        )
    return (
        f"[bad]{CROSS} NOT SIGNIFICANT[/bad]  "
        f"[muted](p = {p_value:.4f})[/muted]"
    )


def lift_value(
    lift: float, lower: float, upper: float, is_significant: bool,
) -> str:
    """Return a styled lift with confidence interval."""
    style = "ok" if (is_significant and lift > 0) else (
        "bad" if lift < 0 else "value"
    )
    return (
        f"[{style}]{lift:+.1%}[/{style}]  "
        f"[muted][{lower:+.1%}, {upper:+.1%}][/muted]"
    )


def iroas_value(iroas: float) -> str:
    """Return a styled iROAS with quality label."""
    if iroas >= 3.0:
        return f"[ok]{iroas:.2f}x[/ok]  [muted]Excellent[/muted]"
    if iroas >= 1.0:
        return f"[ok]{iroas:.2f}x[/ok]  [muted]Profitable[/muted]"
    if iroas >= 0:
        return f"[warn]{iroas:.2f}x[/warn]  [muted]Below breakeven[/muted]"
    return f"[bad]{iroas:.2f}x[/bad]  [muted]Negative[/muted]"


def money(amount: float) -> str:
    """Format a dollar amount."""
    return f"[value]${amount:,.2f}[/value]"


# ── Structured Output ────────────────────────────────────────────────────────

def kv(key: str, val: str, indent: int = 4) -> None:
    """Print a key-value pair with consistent alignment."""
    pad = " " * indent
    console.print(f"{pad}[label]{key:<24}[/label] {val}")


def card(title: str, content: str, border: str = "dim cyan") -> None:
    """Print a bordered card."""
    console.print(Panel(
        content,
        title=f"[heading] {title} [/heading]",
        border_style=border,
        padding=(1, 3),
    ))


def branded_table(title: str, **kwargs) -> Table:
    """Create a Rich Table with LIFT styling."""
    return Table(
        title=f"[heading]{title}[/heading]",
        border_style="dim cyan",
        header_style="bold bright_cyan",
        title_style="",
        show_edge=True,
        pad_edge=True,
        **kwargs,
    )


def result_card(title: str, rows: list[tuple[str, str]], border: str = "dim cyan") -> None:
    """Print a card with key-value rows inside."""
    lines = []
    for key, val in rows:
        lines.append(f"[label]{key:<24}[/label] {val}")
    card(title, "\n".join(lines), border=border)


def spacer() -> None:
    """Print a blank line."""
    console.print()
