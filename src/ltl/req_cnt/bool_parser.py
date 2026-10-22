from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
from ..parser import (
    LTLFormula,
    LTLFormulaBinaryOp,
    LTLFormulaUnaryOp,
    Token,
    F_AND,
    F_NOT,
    F_NEXT,
    F_GLOBALLY,
    F_AP,
)


# =========================================================
# AST
# =========================================================

class BoolFormula:
    def to_ltl(self) -> LTLFormula:
        raise NotImplementedError

    def to_str(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class BoolAtom(BoolFormula):
    name: str

    def to_ltl(self) -> LTLFormula:
        return F_AP(self.name)

    def to_str(self) -> str:
        return self.name


@dataclass(frozen=True)
class BoolNot(BoolFormula):
    child: "BoolFormula"

    def to_ltl(self) -> LTLFormula:
        return F_NOT(self.child.to_ltl())

    def to_str(self) -> str:
        return f"!{self.child.to_str()}"


@dataclass(frozen=True)
class BoolAnd(BoolFormula):
    left: "BoolFormula"
    right: "BoolFormula"

    def to_ltl(self) -> LTLFormula:
        return F_AND(self.left.to_ltl(), self.right.to_ltl())

    def to_str(self) -> str:
        return f"({self.left.to_str()} & {self.right.to_str()})"


@dataclass(frozen=True)
class BoolOr(BoolFormula):
    left: "BoolFormula"
    right: "BoolFormula"

    def to_ltl(self) -> LTLFormula:
        return LTLFormulaBinaryOp(Token.OR, self.left.to_ltl(), self.right.to_ltl())

    def to_str(self) -> str:
        return f"({self.left.to_str()} | {self.right.to_str()})"

# DNF representation types
ClauseLiteral = Tuple[str, bool]
Clause = Tuple[ClauseLiteral, ...]


# =========================================================
# Tokenizer
# =========================================================

def tokenize_bool(expr: str) -> List[str]:
    tokens: List[str] = []
    idx = 0
    while idx < len(expr):
        ch = expr[idx]
        if ch.isspace():
            idx += 1
            continue
        if ch in "()&|!":
            tokens.append(ch)
            idx += 1
            continue
        start = idx
        while idx < len(expr) and (expr[idx].isalnum() or expr[idx] == "_"):
            idx += 1
        if start == idx:
            raise ValueError(f"Unexpected character in boolean formula: {expr[idx]!r}")
        tokens.append(expr[start:idx])
    return tokens


# =========================================================
# Parser
# =========================================================

class BoolParser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> BoolFormula:
        node = self.parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected trailing tokens: {self.tokens[self.pos:]}")
        return node

    def parse_or(self) -> BoolFormula:
        node = self.parse_and()
        while self.peek() == "|":
            self.consume("|")
            node = BoolOr(node, self.parse_and())
        return node

    def parse_and(self) -> BoolFormula:
        node = self.parse_not()
        while self.peek() == "&":
            self.consume("&")
            node = BoolAnd(node, self.parse_not())
        return node

    def parse_not(self) -> BoolFormula:
        if self.peek() == "!":
            self.consume("!")
            return BoolNot(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> BoolFormula:
        token = self.peek()
        if token == "(":
            self.consume("(")
            node = self.parse_or()
            self.consume(")")
            return node
        if token is None:
            raise ValueError("Unexpected end of boolean formula")
        self.pos += 1
        return BoolAtom(token)

    def peek(self) -> Optional[str]:
        return None if self.pos >= len(self.tokens) else self.tokens[self.pos]

    def consume(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Expected token {expected!r}, got {actual!r}")
        self.pos += 1


def parse_bool(expr: str) -> BoolFormula:
    return BoolParser(tokenize_bool(expr)).parse()


# =========================================================
# Formula evaluation (used by req_spec reach/avoid checks)
# =========================================================

@lru_cache(maxsize=None)
def parse_formula_string(expr: str) -> BoolFormula:
    stripped = expr.strip()
    if not stripped:
        raise ValueError("Empty formula string is not supported")
    return BoolParser(tokenize_bool(stripped)).parse()


def formula_holds(expr: str, active_aps: Set[str]) -> bool:
    stripped = expr.strip()
    if not stripped:
        return False

    def eval_node(node: BoolFormula) -> bool:
        if isinstance(node, BoolAtom):
            return node.name in active_aps
        if isinstance(node, BoolNot):
            return not eval_node(node.child)
        if isinstance(node, BoolAnd):
            return eval_node(node.left) and eval_node(node.right)
        if isinstance(node, BoolOr):
            return eval_node(node.left) or eval_node(node.right)
        raise TypeError(f"Unsupported formula node: {type(node)}")

    return eval_node(parse_formula_string(stripped))


# =========================================================
# NNF and DNF conversion (used by cnt_spec)
# =========================================================

def bool_to_nnf(node: BoolFormula) -> BoolFormula:
    if isinstance(node, BoolAtom):
        return node
    if isinstance(node, BoolNot):
        child = node.child
        if isinstance(child, BoolAtom):
            return node
        if isinstance(child, BoolNot):
            return bool_to_nnf(child.child)
        if isinstance(child, BoolAnd):
            return BoolOr(bool_to_nnf(BoolNot(child.left)), bool_to_nnf(BoolNot(child.right)))
        if isinstance(child, BoolOr):
            return BoolAnd(bool_to_nnf(BoolNot(child.left)), bool_to_nnf(BoolNot(child.right)))
    if isinstance(node, BoolAnd):
        return BoolAnd(bool_to_nnf(node.left), bool_to_nnf(node.right))
    if isinstance(node, BoolOr):
        return BoolOr(bool_to_nnf(node.left), bool_to_nnf(node.right))
    raise TypeError(f"Unsupported boolean node: {type(node)}")


def _normalize_clause(clause: Iterable[ClauseLiteral]) -> Optional[Clause]:
    lit_map: Dict[str, bool] = {}
    for atom, positive in clause:
        if atom in lit_map and lit_map[atom] != positive:
            return None
        lit_map[atom] = positive
    return tuple(sorted(lit_map.items(), key=lambda x: (x[0], 0 if x[1] else 1)))


def bool_to_dnf(node: BoolFormula) -> List[Clause]:
    node = bool_to_nnf(node)

    def dnf(inner: BoolFormula) -> List[Clause]:
        if isinstance(inner, BoolAtom):
            return [((inner.name, True),)]
        if isinstance(inner, BoolNot):
            child = inner.child
            if not isinstance(child, BoolAtom):
                raise ValueError("NNF negation must apply to atoms only")
            return [((child.name, False),)]
        if isinstance(inner, BoolOr):
            out = dnf(inner.left) + dnf(inner.right)
            dedup: Dict[Clause, None] = {}
            for clause in out:
                normalized = _normalize_clause(clause)
                if normalized is not None:
                    dedup[normalized] = None
            return list(dedup.keys())
        if isinstance(inner, BoolAnd):
            out: List[Clause] = []
            for left_clause in dnf(inner.left):
                for right_clause in dnf(inner.right):
                    normalized = _normalize_clause(left_clause + right_clause)
                    if normalized is not None:
                        out.append(normalized)
            dedup: Dict[Clause, None] = {}
            for clause in out:
                dedup[clause] = None
            return list(dedup.keys())
        raise TypeError(f"Unsupported boolean node: {type(inner)}")

    return dnf(node)
