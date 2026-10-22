from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union
from .bool_parser import parse_bool, bool_to_dnf, Clause
from .subtask_lists import AndSubtasks, OrSubtasks
from .types import SetRA, StrRA
from ..parser import (
    LTLFormula,
    LTLFormulaBinaryOp,
    LTLFormulaUnaryOp,
    Token,
    F_AND,
    F_NOT,
    F_AP,
)

# =========================================================
# AST
# =========================================================


AgentSet = FrozenSet[int]


def _ltl_or(formulas: List[LTLFormula]) -> LTLFormula:
    result = formulas[0]
    for f in formulas[1:]:
        result = LTLFormulaBinaryOp(Token.OR, result, f)
    return result


def _ltl_and(formulas: List[LTLFormula]) -> LTLFormula:
    result = formulas[0]
    for f in formulas[1:]:
        result = F_AND(result, f)
    return result


class Formula:
    def to_ltl(self) -> LTLFormula:
        raise NotImplementedError


@dataclass(frozen=True)
class Sync(Formula):
    I: AgentSet
    alpha: str
    m: int

    def to_ltl(self) -> LTLFormula:
        # Get all subsets of size m
        subsets = list(combinations(sorted(self.I), self.m))
        # Convert each subset to formula
        subset_formulas = [
            # Combine each element with AND
            _ltl_and([F_AP(self.alpha + "_" + str(i)) for i in subset])
            for subset in subsets
        ]
        # Combine subsets with OR
        combined = _ltl_or(subset_formulas)
        return LTLFormulaUnaryOp(Token.FINALLY, combined)


@dataclass(frozen=True)
class Async(Formula):
    I: AgentSet
    alpha: str
    m: int

    def to_ltl(self) -> LTLFormula:
        # Get all subsets of size m
        subsets = list(combinations(sorted(self.I), self.m))
        # Convert each subset to formula
        subset_formulas = [
            # Combine each element with AND
            _ltl_and([LTLFormulaUnaryOp(Token.FINALLY, F_AP(self.alpha + "_" + str(i))) for i in subset])
            for subset in subsets
        ]
        # Combine subsets with OR
        return _ltl_or(subset_formulas)


@dataclass(frozen=True)
class Not(Formula):
    child: "Formula"

    def to_ltl(self) -> LTLFormula:
        return F_NOT(self.child.to_ltl())


@dataclass(frozen=True)
class And(Formula):
    left: "Formula"
    right: "Formula"

    def to_ltl(self) -> LTLFormula:
        return F_AND(self.left.to_ltl(), self.right.to_ltl())


@dataclass(frozen=True)
class Or(Formula):
    left: "Formula"
    right: "Formula"

    def to_ltl(self) -> LTLFormula:
        return LTLFormulaBinaryOp(Token.OR, self.left.to_ltl(), self.right.to_ltl())


Res = List[List[SetRA]]
ChoiceItem = Tuple[int, str, SetRA]
ChoiceBranch = List[ChoiceItem]
CntSpecInput = Union[str, Formula]


@dataclass(frozen=True)
class Lit:
    op: str
    positive: bool
    I: AgentSet
    alpha: str
    m: int


NNFFormula = Union[Lit, And, Or]



# =========================================================
# Cardinality spec string parser
# =========================================================

class SpecParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self) -> Formula:
        node = self.parse_or()
        self.skip_ws()
        if self.pos != len(self.text):
            raise ValueError(f"Unexpected trailing input: {self.text[self.pos:]}")
        return node

    def parse_or(self) -> Formula:
        node = self.parse_and()
        while True:
            self.skip_ws()
            if not self.try_consume("|"):
                break
            node = Or(node, self.parse_and())
        return node

    def parse_and(self) -> Formula:
        node = self.parse_not()
        while True:
            self.skip_ws()
            if not self.try_consume("&"):
                break
            node = And(node, self.parse_not())
        return node

    def parse_not(self) -> Formula:
        self.skip_ws()
        if self.try_consume("!"):
            return Not(self.parse_not())
        return self.parse_primary()

    def parse_primary(self) -> Formula:
        self.skip_ws()
        if self.try_consume("("):
            node = self.parse_or()
            self.consume(")")
            return node
        return self.parse_literal()

    def parse_literal(self) -> Formula:
        self.skip_ws()
        name = self.parse_identifier()
        self.skip_ws()
        self.consume("[")
        agents = self.parse_agent_set()
        self.skip_ws()
        self.consume(",")
        alpha = self.parse_alpha_field()
        self.consume(",")
        threshold = self.parse_int()
        self.skip_ws()
        self.consume("]")

        lower = name.lower()
        if lower == "sync":
            return Sync(agents, alpha, threshold)
        if lower == "async":
            return Async(agents, alpha, threshold)
        raise ValueError(f"Unknown cardinality operator: {name}")

    def parse_agent_set(self) -> AgentSet:
        self.skip_ws()
        self.consume("{")
        items: List[int] = []
        self.skip_ws()
        if self.peek() == "}":
            self.consume("}")
            return frozenset(items)

        while True:
            items.append(self.parse_int())
            self.skip_ws()
            if self.try_consume("}"):
                break
            self.consume(",")
        return frozenset(items)

    def parse_alpha_field(self) -> str:
        self.skip_ws()
        start = self.pos
        paren_depth = 0
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                if paren_depth == 0:
                    break
                paren_depth -= 1
            elif ch == "," and paren_depth == 0:
                break
            self.pos += 1
        alpha = self.text[start:self.pos].strip()
        if not alpha:
            raise ValueError("Empty alpha formula in cardinality literal")
        return alpha

    def parse_identifier(self) -> str:
        self.skip_ws()
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isalpha():
            self.pos += 1
        if start == self.pos:
            raise ValueError(f"Expected identifier at: {self.text[self.pos:]}")
        return self.text[start:self.pos]

    def parse_int(self) -> int:
        self.skip_ws()
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if start == self.pos:
            raise ValueError(f"Expected integer at: {self.text[self.pos:]}")
        return int(self.text[start:self.pos])

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def peek(self) -> Optional[str]:
        self.skip_ws()
        return self.text[self.pos] if self.pos < len(self.text) else None

    def consume(self, token: str) -> None:
        self.skip_ws()
        if not self.text.startswith(token, self.pos):
            raise ValueError(f"Expected '{token}' at: {self.text[self.pos:]}")
        self.pos += len(token)

    def try_consume(self, token: str) -> bool:
        self.skip_ws()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False


def parse_cnt_spec(spec: str) -> Formula:
    return SpecParser(spec).parse()


def grounded(atom: str, idx: int) -> str:
    return f"{atom}_{idx}"


def clause_to_string(clause: Clause, idx: int) -> str:
    terms = []
    for atom, positive in clause:
        term = grounded(atom, idx)
        terms.append(term if positive else f"!{term}")
    return " & ".join(terms)


def clause_holds(clause_str: str, active_propositions: Set[str]) -> bool:
    for term in clause_str.split(" & "):
        term = term.strip()
        if term.startswith("!"):
            if term[1:] in active_propositions:
                return False
        elif term not in active_propositions:
            return False
    return True


def extract_agent_idx(clause_str: str) -> Optional[int]:
    first = clause_str.split(" & ")[0].strip()
    if first.startswith("!"):
        first = first[1:]
    if "_" not in first:
        return None
    try:
        return int(first.rsplit("_", 1)[1])
    except ValueError:
        return None


def format_reach(reach: Set[str]) -> str:
    return " & ".join(sorted(reach))


def positive_terms(clause_str: str) -> Set[str]:
    out: Set[str] = set()
    for term in clause_str.split(" & "):
        term = term.strip()
        if term and not term.startswith("!"):
            out.add(term)
    return out


INTERNAL_AND = " && "


def split_internal_terms(expr: str) -> List[str]:
    return [term.strip() for term in expr.split(INTERNAL_AND) if term.strip()]


def trim_not_sync_avoid(avoid: Set[str], active_propositions: Set[str]) -> Set[str]:
    if not active_propositions:
        return set(avoid)
    trimmed = {
        clause for clause in avoid
        if any(positive_terms(term) & active_propositions for term in split_internal_terms(clause))
    }
    return trimmed if trimmed else set(avoid)


def format_avoid(avoid: Set[str]) -> str:
    return " | ".join(sorted(entry.replace(INTERNAL_AND, " & ") for entry in avoid))


def public_ra_variants(ra: SetRA) -> List[StrRA]:
    reach_str = format_reach(ra.reach)
    avoid_entries = sorted(ra.avoid)
    if len(avoid_entries) <= 1 or all(INTERNAL_AND not in entry for entry in avoid_entries):
        return [StrRA(reach=reach_str, avoid=format_avoid(ra.avoid))]
    return [StrRA(reach=reach_str, avoid=entry.replace(INTERNAL_AND, " & ")) for entry in avoid_entries]


def to_or_subtasks(res: Res):
    branches = []
    for branch in res:
        variants_per_item = [public_ra_variants(item) for item in branch]
        for choice in product(*variants_per_item):
            branches.append(AndSubtasks(*choice))
    return OrSubtasks(*branches)


def team_subsets(items: Iterable[int], m: int) -> Iterable[Tuple[int, ...]]:
    yield from combinations(tuple(sorted(items)), m)


def powerset_upto(items: Iterable[int], k: int) -> Iterable[Tuple[int, ...]]:
    items = tuple(sorted(items))
    for r in range(0, k + 1):
        yield from combinations(items, r)


# =========================================================
# NNF / DNF for cardinality formula
# =========================================================

def to_nnf(node: Formula) -> NNFFormula:
    if isinstance(node, Sync):
        return Lit("sync", True, node.I, node.alpha, node.m)
    if isinstance(node, Async):
        return Lit("async", True, node.I, node.alpha, node.m)
    if isinstance(node, Not):
        child = node.child
        if isinstance(child, Sync):
            return Lit("sync", False, child.I, child.alpha, child.m)
        if isinstance(child, Async):
            return Lit("async", False, child.I, child.alpha, child.m)
        if isinstance(child, Not):
            return to_nnf(child.child)
        if isinstance(child, And):
            return Or(to_nnf(Not(child.left)), to_nnf(Not(child.right)))
        if isinstance(child, Or):
            return And(to_nnf(Not(child.left)), to_nnf(Not(child.right)))
    if isinstance(node, And):
        return And(to_nnf(node.left), to_nnf(node.right))
    if isinstance(node, Or):
        return Or(to_nnf(node.left), to_nnf(node.right))
    raise TypeError(f"Unsupported node type: {type(node)}")


def to_dnf(node: NNFFormula) -> List[List[Lit]]:
    if isinstance(node, Lit):
        return [[node]]
    if isinstance(node, And):
        left_terms = to_dnf(node.left)
        right_terms = to_dnf(node.right)
        return [left + right for left in left_terms for right in right_terms]
    if isinstance(node, Or):
        return to_dnf(node.left) + to_dnf(node.right)
    raise TypeError(f"Unsupported NNF node: {type(node)}")


# =========================================================
# Extracted representation
# =========================================================

@dataclass(frozen=True)
class ExtractedSubgoal:
    subgoal_id: int
    literal_id: int
    branch_id: int
    source_kind: str
    finite_horizon: bool
    agents: AgentSet
    alpha: str
    m: int
    ra: SetRA


@dataclass(frozen=True)
class ExtractedLiteral:
    literal_id: int
    branch_id: int
    source_kind: str
    finite_horizon: bool
    agents: AgentSet
    alpha: str
    m: int
    subgoal_ids: Tuple[int, ...]


@dataclass(frozen=True)
class ExtractedBranch:
    branch_id: int
    literal_ids: Tuple[int, ...]


@dataclass(frozen=True)
class ExtractedSpec:
    formula: Formula
    root: "ExtractedNode"
    branches: Tuple[ExtractedBranch, ...]
    literals: Dict[int, ExtractedLiteral]
    subgoals: Dict[int, ExtractedSubgoal]


@dataclass(frozen=True)
class ExtractedLitNode:
    literal_id: int


@dataclass(frozen=True)
class ExtractedAndNode:
    left: "ExtractedNode"
    right: "ExtractedNode"


@dataclass(frozen=True)
class ExtractedOrNode:
    left: "ExtractedNode"
    right: "ExtractedNode"


ExtractedNode = Union[ExtractedLitNode, ExtractedAndNode, ExtractedOrNode]


class CntRAExtractor:
    def __init__(self) -> None:
        self._alpha_clause_cache: Dict[str, List[str]] = {}
        self._next_literal_id = 0
        self._next_subgoal_id = 0

    def extract(self, spec: CntSpecInput) -> ExtractedSpec:
        formula = parse_cnt_spec(spec) if isinstance(spec, str) else spec
        literals: Dict[int, ExtractedLiteral] = {}
        subgoals: Dict[int, ExtractedSubgoal] = {}
        root = self._extract_node(to_nnf(formula), literals, subgoals, branch_id=0)

        return ExtractedSpec(
            formula=formula,
            root=root,
            branches=tuple(),
            literals=literals,
            subgoals=subgoals,
        )

    def _extract_node(
        self,
        node: NNFFormula,
        literals: Dict[int, ExtractedLiteral],
        subgoals: Dict[int, ExtractedSubgoal],
        branch_id: int,
    ) -> ExtractedNode:
        if isinstance(node, Lit):
            literal_id = self._next_literal_id
            self._next_literal_id += 1
            extracted_subgoals = self._extract_literal_subgoals(node, literal_id, branch_id)
            for sg in extracted_subgoals:
                subgoals[sg.subgoal_id] = sg
            source_kind = node.op if node.positive else f"not_{node.op}"
            literals[literal_id] = ExtractedLiteral(
                literal_id=literal_id,
                branch_id=branch_id,
                source_kind=source_kind,
                finite_horizon=node.positive,
                agents=node.I,
                alpha=node.alpha,
                m=node.m,
                subgoal_ids=tuple(sg.subgoal_id for sg in extracted_subgoals),
            )
            return ExtractedLitNode(literal_id)
        if isinstance(node, And):
            return ExtractedAndNode(
                self._extract_node(node.left, literals, subgoals, branch_id),
                self._extract_node(node.right, literals, subgoals, branch_id),
            )
        if isinstance(node, Or):
            return ExtractedOrNode(
                self._extract_node(node.left, literals, subgoals, branch_id),
                self._extract_node(node.right, literals, subgoals, branch_id),
            )
        raise TypeError(f"Unsupported extracted node: {type(node)}")

    def _grounded_clause_strings(self, alpha: str, agent_idx: int) -> List[str]:
        cache_key = f"{alpha}@@{agent_idx}"
        if cache_key not in self._alpha_clause_cache:
            clauses = bool_to_dnf(parse_bool(alpha))
            self._alpha_clause_cache[cache_key] = [clause_to_string(clause, agent_idx) for clause in clauses]
        return self._alpha_clause_cache[cache_key]

    def _extract_literal_subgoals(self, lit: Lit, literal_id: int, branch_id: int) -> List[ExtractedSubgoal]:
        source_kind = lit.op if lit.positive else f"not_{lit.op}"
        finite_horizon = lit.positive
        out: List[ExtractedSubgoal] = []

        if lit.op == "sync" and lit.positive:
            subset_sizes = [lit.m]
            for subset_size in subset_sizes:
                for S in team_subsets(lit.I, subset_size):
                    per_agent_options = [self._grounded_clause_strings(lit.alpha, i) for i in S]
                    for choice in product(*per_agent_options):
                        subgoal_id = self._next_subgoal_id
                        self._next_subgoal_id += 1
                        out.append(ExtractedSubgoal(
                            subgoal_id=subgoal_id,
                            literal_id=literal_id,
                            branch_id=branch_id,
                            source_kind=source_kind,
                            finite_horizon=finite_horizon,
                            agents=lit.I,
                            alpha=lit.alpha,
                            m=lit.m,
                            ra=SetRA(reach=set(choice), avoid=set()),
                        ))
            return out

        if lit.op == "async" and lit.positive:
            for subset_size in range(lit.m, len(lit.I) + 1):
                for S in team_subsets(lit.I, subset_size):
                    per_agent_options = [self._grounded_clause_strings(lit.alpha, i) for i in S]
                    for choice in product(*per_agent_options):
                        subgoal_id = self._next_subgoal_id
                        self._next_subgoal_id += 1
                        out.append(ExtractedSubgoal(
                            subgoal_id=subgoal_id,
                            literal_id=literal_id,
                            branch_id=branch_id,
                            source_kind=source_kind,
                            finite_horizon=finite_horizon,
                            agents=lit.I,
                            alpha=lit.alpha,
                            m=lit.m,
                            ra=SetRA(reach=set(choice), avoid=set()),
                        ))
            return out

        if lit.op == "sync" and not lit.positive:
            avoid: Set[str] = set()
            for S in team_subsets(lit.I, lit.m):
                per_agent_options = [self._grounded_clause_strings(lit.alpha, i) for i in S]
                for choice in product(*per_agent_options):
                    avoid.add(INTERNAL_AND.join(choice))
            subgoal_id = self._next_subgoal_id
            self._next_subgoal_id += 1
            out.append(ExtractedSubgoal(
                subgoal_id=subgoal_id,
                literal_id=literal_id,
                branch_id=branch_id,
                source_kind=source_kind,
                finite_horizon=finite_horizon,
                agents=lit.I,
                alpha=lit.alpha,
                m=lit.m,
                ra=SetRA(reach=set(), avoid=avoid),
            ))
            return out

        if lit.op == "async" and not lit.positive:
            for S in powerset_upto(lit.I, lit.m - 1):
                disallowed = sorted(set(lit.I) - set(S))
                avoid: Set[str] = set()
                for j in disallowed:
                    avoid.update(self._grounded_clause_strings(lit.alpha, j))
                subgoal_id = self._next_subgoal_id
                self._next_subgoal_id += 1
                out.append(ExtractedSubgoal(
                    subgoal_id=subgoal_id,
                    literal_id=literal_id,
                    branch_id=branch_id,
                    source_kind=source_kind,
                    finite_horizon=finite_horizon,
                    agents=lit.I,
                    alpha=lit.alpha,
                    m=lit.m,
                    ra=SetRA(reach=set(), avoid=avoid),
                ))
            return out

        raise ValueError(f"Unsupported literal: {lit}")


# =========================================================
# Monitor
# =========================================================

LiteralState = str  # "OPEN" | "SAT" | "VIOL"


@dataclass(frozen=True)
class StepResult:
    is_status_changed: bool
    violated: bool
    satisfied: bool


class CntRAMonitor:
    def __init__(self, extracted: ExtractedSpec):
        self.extracted = extracted
        self.literal_states: Dict[int, LiteralState] = {
            literal_id: "OPEN" for literal_id in extracted.literals
        }
        self.subgoal_achieved: Set[int] = set()
        self.sync_seen: Dict[int, bool] = {
            literal_id: False
            for literal_id, literal in extracted.literals.items()
            if literal.source_kind == "sync"
        }
        self.async_seen_agents: Dict[int, Set[int]] = {
            literal_id: set()
            for literal_id, literal in extracted.literals.items()
            if literal.source_kind == "async"
        }
        self.last_active: Set[str] = set()
        self.history_active: List[Set[str]] = []
        self._current_subgoals_signature = self._compute_subgoals_signature()

    def reset(self) -> None:
        self.literal_states = {literal_id: "OPEN" for literal_id in self.extracted.literals}
        self.subgoal_achieved = set()
        self.sync_seen = {
            literal_id: False
            for literal_id, literal in self.extracted.literals.items()
            if literal.source_kind == "sync"
        }
        self.async_seen_agents = {
            literal_id: set()
            for literal_id, literal in self.extracted.literals.items()
            if literal.source_kind == "async"
        }
        self.last_active = set()
        self.history_active = []
        self._current_subgoals_signature = self._compute_subgoals_signature()

    def step(self, active_propositions: Set[str]) -> StepResult:
        active = set(active_propositions)
        self.last_active = active
        self.history_active.append(active)
        self._update_positive_progress()
        self._update_literal_states()

        new_signature = self._compute_subgoals_signature()
        is_status_changed = new_signature != self._current_subgoals_signature
        self._current_subgoals_signature = new_signature

        status = self.formula_status()
        return StepResult(
            is_status_changed=is_status_changed,
            violated=status["violated"],
            satisfied=status["satisfied"],
        )

    def _literal_current_agents(self, literal: ExtractedLiteral) -> Set[int]:
        current: Set[int] = set()
        for idx in literal.agents:
            if any(
                clause_holds(clause, self.last_active)
                for clause in self._grounded_clause_strings(literal.alpha, idx)
            ):
                current.add(idx)
        return current

    def _grounded_clause_strings(self, alpha: str, agent_idx: int) -> List[str]:
        clauses = bool_to_dnf(parse_bool(alpha))
        return [clause_to_string(clause, agent_idx) for clause in clauses]

    def _update_positive_progress(self) -> None:
        for literal_id, literal in self.extracted.literals.items():
            if literal.source_kind == "sync":
                if len(self._literal_current_agents(literal)) >= literal.m:
                    self.sync_seen[literal_id] = True
            elif literal.source_kind == "async":
                self.async_seen_agents[literal_id].update(self._literal_current_agents(literal))

    def _subgoal_currently_feasible(self, subgoal: ExtractedSubgoal) -> bool:
        if subgoal.source_kind == "not_sync":
            return all(not clause_holds(clause, self.last_active) for clause in subgoal.ra.avoid)
        if subgoal.source_kind == "not_async":
            return all(
                not any(clause_holds(clause, step_active) for step_active in self.history_active)
                for clause in subgoal.ra.avoid
            )
        if subgoal.source_kind == "sync":
            return True
        if subgoal.source_kind == "async":
            return True
        return True

    def _subgoal_achieved_now(self, subgoal: ExtractedSubgoal) -> bool:
        if subgoal.source_kind == "sync":
            return all(clause_holds(clause, self.last_active) for clause in subgoal.ra.reach)
        if subgoal.source_kind == "async":
            return all(
                any(clause_holds(clause, step_active) for step_active in self.history_active)
                for clause in subgoal.ra.reach
            )
        return False

    def _update_literal_states(self) -> None:
        for literal_id, literal in self.extracted.literals.items():
            subgoals = [self.extracted.subgoals[subgoal_id] for subgoal_id in literal.subgoal_ids]

            for subgoal in subgoals:
                if self._subgoal_achieved_now(subgoal):
                    self.subgoal_achieved.add(subgoal.subgoal_id)

            if literal.source_kind == "sync":
                if self.sync_seen[literal_id]:
                    self.literal_states[literal_id] = "SAT"
                    continue
                remaining = [
                    subgoal for subgoal in subgoals
                    if self._subgoal_currently_feasible(subgoal)
                    and subgoal.subgoal_id not in self.subgoal_achieved
                ]
                self.literal_states[literal_id] = "OPEN" if remaining else "VIOL"
                continue

            if literal.source_kind == "async":
                if len(self.async_seen_agents[literal_id]) >= literal.m:
                    self.literal_states[literal_id] = "SAT"
                    continue
                remaining = [
                    subgoal for subgoal in subgoals
                    if self._subgoal_currently_feasible(subgoal)
                    and subgoal.subgoal_id not in self.subgoal_achieved
                ]
                self.literal_states[literal_id] = "OPEN" if remaining else "VIOL"
                continue

            if literal.source_kind in {"not_sync", "not_async"}:
                remaining = [
                    subgoal for subgoal in subgoals
                    if self._subgoal_currently_feasible(subgoal)
                ]
                self.literal_states[literal_id] = "OPEN" if remaining else "VIOL"
                continue

    def formula_status(self) -> Dict[str, bool]:
        state, _ = self._node_state_and_choices(self.extracted.root)
        violated = state == "VIOL"
        satisfied = state == "SAT"
        return {
            "violated": violated,
            "satisfied": satisfied,
            "open": not violated and not satisfied,
        }

    def _literal_choices(self, literal_id: int) -> Tuple[LiteralState, List[ChoiceBranch]]:
        literal = self.extracted.literals[literal_id]
        state = self.literal_states[literal_id]

        if literal.source_kind in {"sync", "async"} and state == "SAT":
            return state, [[(literal_id, literal.source_kind, SetRA(reach=set(), avoid=set()))]]

        subgoals = [
            self.extracted.subgoals[subgoal_id]
            for subgoal_id in literal.subgoal_ids
            if self._subgoal_currently_feasible(self.extracted.subgoals[subgoal_id])
            and (
                literal.source_kind not in {"sync", "async"}
                or subgoal_id not in self.subgoal_achieved
            )
        ]
        return state, [
            [(literal_id, literal.source_kind, SetRA(reach=set(sg.ra.reach), avoid=set(sg.ra.avoid)))]
            for sg in subgoals
        ]

    def _node_state_and_choices(self, node: ExtractedNode) -> Tuple[LiteralState, List[ChoiceBranch]]:
        if isinstance(node, ExtractedLitNode):
            return self._literal_choices(node.literal_id)

        if isinstance(node, ExtractedAndNode):
            left_state, left_choices = self._node_state_and_choices(node.left)
            right_state, right_choices = self._node_state_and_choices(node.right)
            if left_state == "VIOL" or right_state == "VIOL":
                return "VIOL", []
            choices = [left + right for left in left_choices for right in right_choices]
            if left_state == "SAT" and right_state == "SAT":
                return "SAT", choices
            return "OPEN", choices

        if isinstance(node, ExtractedOrNode):
            left_state, left_choices = self._node_state_and_choices(node.left)
            right_state, right_choices = self._node_state_and_choices(node.right)
            if left_state == "SAT" and right_state == "SAT":
                return "SAT", left_choices + right_choices
            if left_state == "SAT":
                return "SAT", left_choices
            if right_state == "SAT":
                return "SAT", right_choices
            if left_state == "VIOL" and right_state == "VIOL":
                return "VIOL", []
            if left_state == "VIOL":
                return "OPEN", right_choices
            if right_state == "VIOL":
                return "OPEN", left_choices
            return "OPEN", left_choices + right_choices

        raise TypeError(f"Unsupported extracted node: {type(node)}")

    def _compute_subgoals_signature(self) -> Tuple[str, Tuple[Tuple[Tuple[str, str], ...], ...]]:
        status = self.formula_status()
        if status["violated"]:
            return ("violated", ())
        if status["satisfied"]:
            return ("satisfied", ())

        _, node_choices = self._node_state_and_choices(self.extracted.root)
        res: Res = []
        for choice_branch in node_choices:
            merged_avoid: Set[str] = set()
            for _, _, ra in choice_branch:
                merged_avoid |= ra.avoid

            merged_reach: Set[str] = set()
            valid = True
            for literal_id, source_kind, ra in choice_branch:
                reach = set(ra.reach)
                if source_kind == "async":
                    satisfied_agents = self.async_seen_agents.get(literal_id, set())
                    reach = {
                        clause for clause in reach
                        if extract_agent_idx(clause) not in satisfied_agents
                    }
                    reach -= merged_avoid
                elif source_kind == "not_sync":
                    pass
                elif reach & merged_avoid:
                    valid = False
                    break
                merged_reach |= reach
            if any(source_kind == "not_sync" for _, source_kind, _ in choice_branch):
                merged_avoid = trim_not_sync_avoid(merged_avoid, self.last_active)
            if valid and format_reach(merged_reach) and format_reach(merged_reach) == format_avoid(merged_avoid):
                valid = False
            if valid and any(clause_holds(clause, self.last_active) for clause in merged_avoid):
                valid = False
            if valid:
                res.append([SetRA(reach=merged_reach, avoid=merged_avoid)])

        res = self._deduplicate_res(res)
        public = to_or_subtasks(res)
        branches = []
        for branch in public:
            items = []
            for item in branch:
                items.append((item.reach, item.avoid))
            branches.append(tuple(items))
        return ("open", tuple(branches))

    def current_subtasks(self) -> Dict[str, object]:
        status = self.formula_status()
        if status["violated"]:
            return {
                "success": False,
                "violated": True,
                "satisfied": False,
                "message": "spec violated",
            }
        if status["satisfied"]:
            return {
                "success": True,
                "violated": False,
                "satisfied": True,
                "message": "spec satisfied",
                "res": OrSubtasks(),
            }

        _, node_choices = self._node_state_and_choices(self.extracted.root)
        res: Res = []
        for choice_branch in node_choices:
            merged_avoid: Set[str] = set()
            for _, _, ra in choice_branch:
                merged_avoid |= ra.avoid

            merged_reach: Set[str] = set()
            valid = True
            choice_has_not_sync = False
            for literal_id, source_kind, ra in choice_branch:
                reach = set(ra.reach)
                if source_kind == "async":
                    satisfied_agents = self.async_seen_agents.get(literal_id, set())
                    reach = {
                        clause for clause in reach
                        if extract_agent_idx(clause) not in satisfied_agents
                    }
                    reach -= merged_avoid
                elif source_kind == "not_sync":
                    choice_has_not_sync = True
                elif reach & merged_avoid:
                    valid = False
                    break
                merged_reach |= reach
            if choice_has_not_sync:
                merged_avoid = trim_not_sync_avoid(merged_avoid, self.last_active)
            if valid and format_reach(merged_reach) and format_reach(merged_reach) == format_avoid(merged_avoid):
                valid = False
            if valid and any(clause_holds(clause, self.last_active) for clause in merged_avoid):
                valid = False
            if valid:
                res.append([SetRA(reach=merged_reach, avoid=merged_avoid)])

        res = self._deduplicate_res(res)

        return {
            "success": True,
            "violated": False,
            "satisfied": status["satisfied"],
            "res": to_or_subtasks(res),
        }

    @staticmethod
    def _deduplicate_res(res: Res) -> Res:
        seen = set()
        out: Res = []
        for branch in res:
            key = tuple(
                sorted(
                    (
                        tuple(sorted(item.reach)),
                        tuple(sorted(item.avoid)),
                    )
                    for item in branch
                )
            )
            if key not in seen:
                seen.add(key)
                out.append(branch)
        return out


def extract_cnt_subtasks(formula: Formula) -> ExtractedSpec:
    return CntRAExtractor().extract(formula)


def build_cnt_monitor(formula: Formula) -> CntRAMonitor:
    return CntRAMonitor(extract_cnt_subtasks(formula))


def print_result(result: Dict[str, object], limit: Optional[int] = None) -> None:
    if not result["success"]:
        print(result["message"])
        return

    if result.get("satisfied"):
        print(result.get("message", "spec satisfied"))
        return

    branches = result["res"]
    if limit is not None:
        print(f"number of subtasks: {len(branches)}")
        branches = OrSubtasks(*list(branches)[:limit])

    print(f"{branches:pretty}")


if __name__ == "__main__":
    extractor = CntRAExtractor()

    formula = parse_cnt_spec("!Async[{0, 1, 2, 4}, (blue | yellow), 2] & (Sync[{0, 2, 3}, green, 2] | Async[{0, 1, 2, 3, 4}, yellow, 4])")
    extracted = extractor.extract(formula)
    # print(f"Extracted = {extracted}")
    monitor = CntRAMonitor(extracted)
    # monitor.step(set(["blue_0"]))
    monitor.step(set(["blue_0"]))
    # monitor.step(set(["yellow_1"]))
    # monitor.step(set(["blue_1"]))
    # monitor.step(set(["green_0"]))
    # monitor.step(set(["green_0", "green_2"]))
    print_result(monitor.current_subtasks(), limit=None)


    print("=" * 50)
    formula = parse_cnt_spec("Async[{0, 1}, c, 2] & !Sync[{0, 1}, c, 2]")
    extracted = extractor.extract(formula)
    monitor = CntRAMonitor(extracted)
    monitor.step(set())
    print_result(monitor.current_subtasks(), limit=None)
    monitor.step(set(["c_0"]))
    print_result(monitor.current_subtasks(), limit=None)
    monitor.step(set())
    print_result(monitor.current_subtasks(), limit=None)
    monitor.step(set(["c_1"]))
    print_result(monitor.current_subtasks(), limit=None)


    print("=" * 50)
    formula = parse_cnt_spec("async[{0, 1, 2}, green, 2] | !sync[{0, 1, 2}, blue, 3]")
    extracted = extractor.extract(formula)
    monitor = CntRAMonitor(extracted)
    monitor.step(set())
    print_result(monitor.current_subtasks(), limit=None)

    print("=" * 50)
    formula = Not(Sync(frozenset({0, 1, 2}), "(a & b) | c", 2))
    extracted = extractor.extract(formula)
    monitor = CntRAMonitor(extracted)
    monitor.step(set())
    print_result(monitor.current_subtasks(), limit=10)

    print("=" * 50)
    # !Async[{0,1,2,4}, blue, 2]
    formula = parse_cnt_spec("!Async[{0, 1, 2, 4}, blue, 2]")
    # formula = Not(Async(frozenset({0, 1, 2, 4}), "blue", 2))
    extracted = extractor.extract(formula)
    monitor = CntRAMonitor(extracted)
    monitor.step(set())
    print_result(monitor.current_subtasks(), limit=10)
    monitor.step({"blue_0"})
    print_result(monitor.current_subtasks(), limit=10)
    monitor.step({"blue_1"})
    print_result(monitor.current_subtasks(), limit=10)


    print("=" * 50)
    # !Sync[{0,1,2,4}, blue, 2]
    formula = parse_cnt_spec("!Sync[{0, 1, 2, 4}, blue, 2]")
    # formula = Not(Async(frozenset({0, 1, 2, 4}), "blue", 2))
    extracted = extractor.extract(formula)
    monitor = CntRAMonitor(extracted)
    monitor.step(set())
    print_result(monitor.current_subtasks(), limit=10)
    monitor.step({"blue_0"})
    print_result(monitor.current_subtasks(), limit=10)
    monitor.step({"blue_1"})
    print_result(monitor.current_subtasks(), limit=10)


    print("=" * 50)
    # !Async[{0,1}, yellow, 2]
    formula = parse_cnt_spec("!Async[{0, 1}, yellow, 2]")
    # formula = Not(Async(frozenset({0, 1, 2, 4}), "blue", 2))
    extracted = extractor.extract(formula)
    monitor = CntRAMonitor(extracted)
    monitor.step(set())
    print_result(monitor.current_subtasks(), limit=10)
    monitor.step({"yellow_0"})
    print_result(monitor.current_subtasks(), limit=10)
    monitor.step({"yellow_1"})
    print_result(monitor.current_subtasks(), limit=10)


    print("=" * 50)
    # !Sync[{0,1}, yellow, 2]
    formula = parse_cnt_spec("!Sync[{0, 1}, yellow, 2]")
    # formula = Not(Async(frozenset({0, 1, 2, 4}), "blue", 2))
    extracted = extractor.extract(formula)
    monitor = CntRAMonitor(extracted)
    info = monitor.step(set())
    print(f"is status changed? {info.is_status_changed}")
    print_result(monitor.current_subtasks(), limit=10)
    info = monitor.step({"yellow_0"})
    print(f"is status changed? {info.is_status_changed}")
    print_result(monitor.current_subtasks(), limit=10)
    info = monitor.step({"yellow_1"})
    print(f"is status changed? {info.is_status_changed}")
    print_result(monitor.current_subtasks(), limit=10)


    print("=" * 50)
    # !Async[{0,1,2,4}, blue, 2] & Sync[{0,2,3}, green, 2]
    formula = parse_cnt_spec("!Async[{0,1,2,4}, blue, 2] & Sync[{0,2,3}, green, 2]")
    extracted = extractor.extract(formula)
    monitor = CntRAMonitor(extracted)
    # monitor.step({"green_0", "green_2"})
    info = monitor.step({})
    print(f"is status changed? {info.is_status_changed}")
    info = monitor.step({"green_0", "green_2"})
    print(f"is status changed? {info.is_status_changed}")
    print_result(monitor.current_subtasks(), limit=10)
