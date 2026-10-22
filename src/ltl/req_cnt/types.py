from __future__ import annotations

from dataclasses import dataclass
from typing import Set


@dataclass
class StrRA:
    """Reach-avoid subgoal with string-encoded formulas."""
    reach: str
    avoid: str = ""
    mode: str = "once"


@dataclass
class SetRA:
    """Internal reach-avoid pair with set-encoded propositions (cnt_spec only)."""
    reach: Set[str]
    avoid: Set[str]
