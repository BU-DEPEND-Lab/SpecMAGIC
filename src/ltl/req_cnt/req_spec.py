from __future__ import annotations

import re
from collections import deque
from itertools import product
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

from .bool_parser import BoolAtom, BoolAnd, BoolOr, formula_holds
from .subtask_lists import AndSubtasks, OrSubtasks, SeqSubtasks
from .types import StrRA
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

# BoolAtom, BoolAnd, BoolOr imported from bool_parser
BoolExpr = Union[BoolAtom, BoolAnd, BoolOr]


# =========================================================
# Req formula AST: alpha Req beta
# =========================================================

class Formula:
    def to_ltl(self) -> LTLFormula:
        raise NotImplementedError


@dataclass(frozen=True)
class Req(Formula):
    a: BoolExpr
    beta: BoolExpr

    def to_ltl(self) -> LTLFormula:
        return LTLFormulaBinaryOp(Token.UNTIL, F_NOT(self.a.to_ltl()), self.beta.to_ltl())


@dataclass(frozen=True)
class EachReq(Formula):
    a: BoolExpr
    beta: BoolExpr

    def to_ltl(self) -> LTLFormula:
        # (!a U beta) & G((a & X !a) -> X(!a W beta)), with weak until encoded as
        # phi W psi == (phi U psi) | G phi since the LTL AST has no W token.
        not_a_u_beta = LTLFormulaBinaryOp(Token.UNTIL, F_NOT(self.a.to_ltl()), self.beta.to_ltl())
        a_and_next_not_a = LTLFormulaBinaryOp(Token.AND, self.a.to_ltl(), F_NEXT(F_NOT(self.a.to_ltl())))
        not_a_w_beta = LTLFormulaBinaryOp(
            Token.OR,
            LTLFormulaBinaryOp(Token.UNTIL, F_NOT(self.a.to_ltl()), self.beta.to_ltl()),
            F_GLOBALLY(F_NOT(self.a.to_ltl())),
        )
        repeat = F_GLOBALLY(LTLFormulaBinaryOp(Token.IMPLIES, a_and_next_not_a, F_NEXT(not_a_w_beta)))
        return F_AND(not_a_u_beta, repeat)


@dataclass(frozen=True)
class ReqAnd(Formula):
    left: "Formula"
    right: "Formula"

    def to_ltl(self) -> LTLFormula:
        return F_AND(self.left.to_ltl(), self.right.to_ltl())


@dataclass(frozen=True)
class ReqOr(Formula):
    left: "Formula"
    right: "Formula"

    def to_ltl(self) -> LTLFormula:
        return LTLFormulaBinaryOp(Token.OR, self.left.to_ltl(), self.right.to_ltl())

# Backward-compatible aliases for older code. Prefer BoolAtom/BoolAnd/BoolOr
# and ReqAnd/ReqOr in new code because they are consistent with cnt_spec.py
# and avoid ambiguous generic names.
AP = BoolAtom
BAnd = BoolAnd
BOr = BoolOr
And = ReqAnd
Or = ReqOr


# =========================================================
# Types
# =========================================================

ConjTerm = FrozenSet[str]
StepSubtask = Union[StrRA, AndSubtasks[StrRA]]
SeqOrRA = Union[StrRA, SeqSubtasks[StepSubtask], AndSubtasks[StrRA]]
SubgoalId = Tuple[int, ...]
ReqSpecInput = Union[str, Formula]


@dataclass(frozen=True)
class ChosenReq:
    trigger_terms: Tuple[ConjTerm, ...]  # DNF conjunction terms of alpha
    trigger_formula: str  # one Boolean formula string for alpha
    pre: ConjTerm         # one DNF conjunction term of beta
    mode: str             # "once" for Req, "repeat" for EachReq


# =========================================================
# Helpers
# =========================================================

def extract_agent_id(ap: str) -> int:
    _, idx = ap.rsplit("_", 1)
    return int(idx)


def same_agent_overlap(term: FrozenSet[str]) -> bool:
    """
    Reject conjunction terms containing different grounded APs
    from the same agent.
    """
    seen: Dict[int, str] = {}
    for ap in term:
        agent_id = extract_agent_id(ap)
        if agent_id in seen and seen[agent_id] != ap:
            return True
        seen[agent_id] = ap
    return False


def conjunction_str(term: ConjTerm) -> str:
    ordered = sorted(term)
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    return " & ".join(ordered)


def disjunction_str(terms: List[ConjTerm]) -> str:
    rendered: List[str] = []
    seen: Set[ConjTerm] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        text = conjunction_str(term)
        if len(term) > 1:
            text = f"({text})"
        rendered.append(text)
    if not rendered:
        return ""
    if len(rendered) == 1:
        return rendered[0]
    return " | ".join(rendered)


def build_step_subtask(reach: str, avoid_terms: List[ConjTerm], mode: str) -> StepSubtask:
    unique_terms: List[ConjTerm] = []
    seen: Set[ConjTerm] = set()
    for term in avoid_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    if len(unique_terms) == 1:
        return StrRA(reach=reach, avoid=conjunction_str(unique_terms[0]), mode=mode)

    if all(len(term) == 1 for term in unique_terms):
        return StrRA(reach=reach, avoid=disjunction_str(unique_terms), mode=mode)

    return AndSubtasks(*[
        StrRA(reach=reach, avoid=conjunction_str(term), mode=mode)
        for term in unique_terms
    ])


def canonical_ra(item: StrRA) -> Tuple[str, str, str]:
    return (item.reach, item.avoid, item.mode)

def normalize_and_step(item: AndSubtasks[StrRA]) -> AndSubtasks[StrRA]:
    seen = set()
    out: AndSubtasks[StrRA] = AndSubtasks()
    for subitem in item:
        key = canonical_ra(subitem)
        if key not in seen:
            seen.add(key)
            out.append(StrRA(reach=subitem.reach, avoid=subitem.avoid, mode=subitem.mode))
    return out


def canonical_and_step(item: AndSubtasks[RA]):
    return tuple(sorted(canonical_ra(subitem) for subitem in normalize_and_step(item)))


def canonical_step(item: StepSubtask):
    if isinstance(item, AndSubtasks):
        return ("and", canonical_and_step(item))
    return ("ra", canonical_ra(item))


def deduplicate_ra_list(items: List[StepSubtask]) -> List[StepSubtask]:
    seen = set()
    out: List[StepSubtask] = type(items)()  # Create new list with the same type
    for item in items:
        key = canonical_step(item)
        if key not in seen:
            seen.add(key)
            if isinstance(item, AndSubtasks):
                out.append(normalize_and_step(item))
            else:
                out.append(StrRA(reach=item.reach, avoid=item.avoid, mode=item.mode))
    return out


def normalize_sequence(seq: List[StepSubtask]) -> List[StepSubtask]:
    # A sequence represents an enforced temporal order.  Two ordered units may
    # legitimately have the same reach set at different dependency depths, so
    # only exact duplicates are removed.
    return deduplicate_ra_list(seq)


def canonical_sequence(seq: List[StepSubtask]):
    return tuple(canonical_step(x) for x in normalize_sequence(seq))


def canonical_conj_item(item: SeqOrRA):
    if isinstance(item, SeqSubtasks):
        return ("seq", canonical_sequence(item))
    if isinstance(item, AndSubtasks):
        return ("and", canonical_and_step(item))
    return ("ra", canonical_ra(item))


def normalize_conj_branch(branch: AndSubtasks[SeqOrRA]) -> AndSubtasks[SeqOrRA]:
    seen = set()
    out: AndSubtasks[SeqOrRA] = AndSubtasks()

    for item in branch:
        if isinstance(item, SeqSubtasks):
            norm_item: SeqOrRA = normalize_sequence(item)
        elif isinstance(item, AndSubtasks):
            norm_item = normalize_and_step(item)
        else:
            norm_item = StrRA(reach=item.reach, avoid=item.avoid, mode=item.mode)

        key = canonical_conj_item(norm_item)
        if key not in seen:
            seen.add(key)
            if isinstance(norm_item, AndSubtasks):
                out.extend(norm_item)
            else:
                out.append(norm_item)

    return out


def deduplicate_res(res: OrSubtasks[AndSubtasks[SeqOrRA]]) -> OrSubtasks[AndSubtasks[SeqOrRA]]:
    seen = set()
    out: OrSubtasks[AndSubtasks[SeqOrRA]] = OrSubtasks()

    for branch in res:
        norm_branch = normalize_conj_branch(branch)
        key = tuple(sorted(canonical_conj_item(x) for x in norm_branch))
        if key not in seen:
            seen.add(key)
            out.append(norm_branch)

    return out


def and_contains_or(item) -> bool:
    if isinstance(item, OrSubtasks):
        return True
    if isinstance(item, SeqSubtasks):
        return any(and_contains_or(x) for x in item)
    if isinstance(item, AndSubtasks):
        return any(and_contains_or(x) for x in item)
    return False


def validate_no_or_inside_and(res: OrSubtasks[AndSubtasks[SeqOrRA]]) -> None:
    for branch in res:
        for item in branch:
            if and_contains_or(item):
                raise ValueError("Invalid extracted structure: OrSubtasks found inside AndSubtasks")


# =========================================================
# Boolean formula -> DNF
# Returns list of conjunction terms, each term is a frozenset of grounded APs
# =========================================================

def bool_to_dnf(expr: BoolExpr) -> List[ConjTerm]:
    if isinstance(expr, BoolAtom):
        return [frozenset([expr.name])]

    if isinstance(expr, BoolAnd):
        left_terms = bool_to_dnf(expr.left)
        right_terms = bool_to_dnf(expr.right)
        return [l | r for l in left_terms for r in right_terms]

    if isinstance(expr, BoolOr):
        return bool_to_dnf(expr.left) + bool_to_dnf(expr.right)

    raise TypeError(f"Unsupported BoolExpr: {type(expr)}")


# =========================================================
# String parser for Req / EachReq specifications
# =========================================================

TOKEN_RE = re.compile(r"\s*(EachReq|Req|\(|\)|&|\||[A-Za-z0-9_]+)")


def tokenize_req_spec(text: str) -> List[str]:
    tokens: List[str] = []
    pos = 0
    while pos < len(text):
        match = TOKEN_RE.match(text, pos)
        if not match:
            raise ValueError(f"Unexpected token near: {text[pos:pos+20]!r}")
        token = match.group(1)
        tokens.append(token)
        pos = match.end()
    return tokens


class ReqSpecParser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def consume(self, expected: Optional[str] = None) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of input")
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def parse_formula(self) -> Formula:
        left = self.parse_formula_and()
        while self.peek() == "|":
            self.consume("|")
            right = self.parse_formula_and()
            left = ReqOr(left, right)
        return left

    def parse_formula_and(self) -> Formula:
        left = self.parse_formula_atom()
        while self.peek() == "&":
            self.consume("&")
            right = self.parse_formula_atom()
            left = ReqAnd(left, right)
        return left

    def parse_formula_atom(self) -> Formula:
        start = self.pos
        try:
            return self.parse_req_clause()
        except ValueError:
            self.pos = start

        if self.peek() == "(":
            self.consume("(")
            inner = self.parse_formula()
            self.consume(")")
            return inner
        return self.parse_req_clause()

    def parse_req_clause(self) -> Union[Req, EachReq]:
        alpha = self.parse_bool_expr()
        op = self.peek()
        if op not in {"Req", "EachReq"}:
            raise ValueError(f"Expected 'Req' or 'EachReq', got {op!r}")
        self.consume()
        beta = self.parse_bool_expr()
        if op == "Req":
            return Req(alpha, beta)
        return EachReq(alpha, beta)

    def parse_bool_expr(self) -> BoolExpr:
        return self.parse_bool_or()

    def parse_bool_or(self) -> BoolExpr:
        left = self.parse_bool_and()
        while self.peek() == "|":
            self.consume("|")
            right = self.parse_bool_and()
            left = BoolOr(left, right)
        return left

    def parse_bool_and(self) -> BoolExpr:
        left = self.parse_bool_atom()
        while self.peek() == "&":
            self.consume("&")
            right = self.parse_bool_atom()
            left = BoolAnd(left, right)
        return left

    def parse_bool_atom(self) -> BoolExpr:
        token = self.peek()
        if token == "(":
            self.consume("(")
            expr = self.parse_bool_expr()
            self.consume(")")
            return expr
        if token is None:
            raise ValueError("Unexpected end of input while parsing boolean expression")
        if token in {"Req", "EachReq", "&", "|", ")"}:
            raise ValueError(f"Unexpected token in boolean expression: {token!r}")
        self.consume()
        return BoolAtom(token)


def parse_req_spec(text: str) -> Formula:
    parser = ReqSpecParser(tokenize_req_spec(text))
    formula = parser.parse_formula()
    if parser.peek() is not None:
        raise ValueError(f"Unexpected trailing token: {parser.peek()!r}")
    return formula


# =========================================================
# Whole formula -> DNF over Req clauses
# =========================================================

def formula_to_dnf(node: Formula) -> List[List[Union[Req, EachReq]]]:
    if isinstance(node, (Req, EachReq)):
        return [[node]]

    if isinstance(node, ReqAnd):
        left_terms = formula_to_dnf(node.left)
        right_terms = formula_to_dnf(node.right)
        return [l + r for l in left_terms for r in right_terms]

    if isinstance(node, ReqOr):
        return formula_to_dnf(node.left) + formula_to_dnf(node.right)

    raise TypeError(f"Unsupported Formula node: {type(node)}")


# =========================================================
# Clause expansion
# A disjunctive trigger alpha = \/r A_r means every trigger term A_r must be
# handled. Therefore one clause expands to one or more clause-realizations,
# where each realization contains one chosen prerequisite term for every
# trigger term.
# =========================================================

def expand_req_clause(clause: Union[Req, EachReq]) -> List[List[ChosenReq]]:
    alpha_terms = [t for t in bool_to_dnf(clause.a) if not same_agent_overlap(t)]
    beta_terms = [t for t in bool_to_dnf(clause.beta) if not same_agent_overlap(t)]
    mode = "repeat" if isinstance(clause, EachReq) else "once"
    trigger_formula = clause.a.to_str()

    if not alpha_terms or not beta_terms:
        return []

    out: List[List[ChosenReq]] = []
    for pre_term in beta_terms:
        out.append([
            ChosenReq(
                trigger_terms=tuple(alpha_terms),
                trigger_formula=trigger_formula,
                pre=pre_term,
                mode=mode,
            )
        ])

    # deduplicate clause-realizations
    seen = set()
    uniq: List[List[ChosenReq]] = []
    for realization in out:
        key = tuple(
            sorted(
                (
                    tuple(sorted(tuple(sorted(term)) for term in x.trigger_terms)),
                    x.trigger_formula,
                    tuple(sorted(x.pre)),
                    x.mode,
                )
                for x in realization
            )
        )
        if key not in seen:
            seen.add(key)
            uniq.append(realization)
    return uniq


# =========================================================
# Dependency graph over chosen clause-units
#
# unit i -> unit j iff trigger_i is required by pre_j
# i.e. trigger_i ∈ pre_j
#
# This correctly catches:
#   (a Req (b & c)) & (b Req a)
# because:
#   unit2.trigger = b ∈ unit1.pre
#   unit1.trigger = a ∈ unit2.pre
# =========================================================

def build_unit_graph(chosen: List[ChosenReq]) -> Dict[int, Set[int]]:
    n = len(chosen)
    graph: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if any(term & chosen[j].pre for term in chosen[i].trigger_terms):
                graph[i].add(j)

    return graph


def has_cycle_unit(graph: Dict[int, Set[int]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}

    def dfs(v: int) -> bool:
        color[v] = GRAY
        for u in graph[v]:
            if color[u] == GRAY:
                return True
            if color[u] == WHITE and dfs(u):
                return True
        color[v] = BLACK
        return False

    for v in graph:
        if color[v] == WHITE and dfs(v):
            return True
    return False


def compute_descendants_unit(graph: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    desc: Dict[int, Set[int]] = {}

    def dfs(v: int, visited: Set[int]) -> None:
        for u in graph[v]:
            if u not in visited:
                visited.add(u)
                dfs(u, visited)

    for v in graph:
        visited: Set[int] = set()
        dfs(v, visited)
        desc[v] = visited

    return desc


def topological_sort_unit(graph: Dict[int, Set[int]], nodes: Set[int]) -> List[int]:
    indegree = {v: 0 for v in nodes}
    for v in nodes:
        for u in graph[v]:
            if u in nodes:
                indegree[u] += 1

    q = deque(sorted(v for v in nodes if indegree[v] == 0))
    order: List[int] = []

    while q:
        v = q.popleft()
        order.append(v)
        for u in sorted(graph[v]):
            if u not in nodes:
                continue
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)

    if len(order) != len(nodes):
        raise ValueError("cycle detected")
    return order


def weakly_connected_components_unit(graph: Dict[int, Set[int]], nodes: Set[int]) -> List[Set[int]]:
    undirected: Dict[int, Set[int]] = {v: set() for v in nodes}
    for v in nodes:
        for u in graph[v]:
            undirected[v].add(u)
            undirected[u].add(v)

    visited: Set[int] = set()
    comps: List[Set[int]] = []

    for v in sorted(nodes):
        if v in visited:
            continue
        stack = [v]
        comp: Set[int] = set()
        visited.add(v)

        while stack:
            x = stack.pop()
            comp.add(x)
            for y in undirected[x]:
                if y not in visited:
                    visited.add(y)
                    stack.append(y)

        comps.append(comp)

    return comps


# =========================================================
# Extraction from one connected component
#
# For unit i:
#   reach = pre_i
#   avoid = trigger_i OR descendants of i
# =========================================================

def extract_component_item(chosen: List[ChosenReq], graph: Dict[int, Set[int]], comp: Set[int]) -> Optional[SeqOrRA]:
    desc = compute_descendants_unit(graph)
    topo = topological_sort_unit(graph, comp)

    seq: SeqSubtasks[StepSubtask] = SeqSubtasks()
    for i in topo:
        reach = conjunction_str(chosen[i].pre)
        avoid_terms: List[ConjTerm] = list(chosen[i].trigger_terms)
        for j in sorted(desc[i] & comp):
            avoid_terms.extend(chosen[j].trigger_terms)

        seq.append(build_step_subtask(reach, avoid_terms, chosen[i].mode))

    seq = normalize_sequence(seq)

    if not seq:
        return None
    if len(seq) == 1:
        return seq[0]
    return seq


# =========================================================
# Main pipeline
#
# res = [subtask_1, subtask_2, ...]
# outer list: OR
# each subtask: AND (list)
# each item in subtask:
#   - single RA dict, or
#   - sequence (list of RA dicts)
# =========================================================

def req_formula_to_res(formula: Formula) -> Dict[str, object]:
    dnf_branches = formula_to_dnf(formula)
    final_res: OrSubtasks[AndSubtasks[SeqOrRA]] = OrSubtasks()

    for req_branch in dnf_branches:
        per_clause_choices: List[List[List[ChosenReq]]] = []
        valid_branch = True

        for clause in req_branch:
            choices = expand_req_clause(clause)
            if not choices:
                valid_branch = False
                break
            per_clause_choices.append(choices)

        if not valid_branch:
            continue

        for chosen_tuple in product(*per_clause_choices):
            chosen = [unit for clause_realization in chosen_tuple for unit in clause_realization]
            graph = build_unit_graph(chosen)

            if has_cycle_unit(graph):
                continue

            nodes = set(range(len(chosen)))
            comps = weakly_connected_components_unit(graph, nodes)

            conj_branch: AndSubtasks[SeqOrRA] = AndSubtasks()
            for comp in comps:
                item = extract_component_item(chosen, graph, comp)
                if item is not None:
                    conj_branch.append(item)

            conj_branch = normalize_conj_branch(conj_branch)
            if conj_branch:
                final_res.append(conj_branch)

    final_res = deduplicate_res(final_res)
    validate_no_or_inside_and(final_res)

    if not final_res:
        return {
            "success": False,
            "message": "all branches are invalid or cyclic under the current Req compilation"
        }

    return {
        "success": True,
        "res": final_res
    }


# # =========================================================
# # Pretty print
# # =========================================================

# def print_result(result: Dict[str, object]) -> None:
#     if not result["success"]:
#         print("UNSUCCESSFUL:")
#         print(result["message"])
#     elif "res" in result:
#         res = result['res']
#         print(f"{res:pretty}")
#     else:
#         print("ERROR: 'res' field not found")
#         print(result)



# formula_holds imported from bool_parser


# =========================================================
# Extracted representation
# =========================================================


@dataclass(frozen=True)
class ExtractedReqSubgoal:
    subgoal_id: SubgoalId
    branch_id: int
    reach: str
    avoid: str
    mode: str


@dataclass(frozen=True)
class ExtractedReqBranch:
    branch_id: int
    structure: AndSubtasks[SeqOrRA]
    subgoal_ids: Tuple[SubgoalId, ...]


@dataclass(frozen=True)
class ExtractedReqSpec:
    source: ReqSpecInput
    formula: Formula
    branches: Tuple[ExtractedReqBranch, ...]
    subgoals: Dict[SubgoalId, ExtractedReqSubgoal]


class ReqRAExtractor:
    def extract(self, spec: ReqSpecInput) -> ExtractedReqSpec:
        formula = parse_req_spec(spec) if isinstance(spec, str) else spec
        result = req_formula_to_res(formula)
        if not result.get("success"):
            raise ValueError(result.get("message", "Req extraction failed"))

        res: OrSubtasks[AndSubtasks[SeqOrRA]] = result["res"]
        subgoals: Dict[SubgoalId, ExtractedReqSubgoal] = {}
        branches: List[ExtractedReqBranch] = []

        for branch_id, branch in enumerate(res):
            branch_subgoal_ids = list(self._collect_subgoals(branch_id, branch, subgoals))
            branches.append(
                ExtractedReqBranch(
                    branch_id=branch_id,
                    structure=branch,
                    subgoal_ids=tuple(branch_subgoal_ids),
                )
            )

        return ExtractedReqSpec(
            source=spec,
            formula=formula,
            branches=tuple(branches),
            subgoals=subgoals,
        )

    def _collect_subgoals(
        self,
        branch_id: int,
        item: Union[AndSubtasks[SeqOrRA], SeqSubtasks[StepSubtask], AndSubtasks[StrRA], StrRA],
        subgoals: Dict[SubgoalId, ExtractedReqSubgoal],
        path: Tuple[int, ...] = (),
    ) -> Iterable[SubgoalId]:
        if isinstance(item, StrRA):
            subgoal_id = (branch_id,) + path
            subgoals[subgoal_id] = ExtractedReqSubgoal(
                subgoal_id=subgoal_id,
                branch_id=branch_id,
                reach=item.reach,
                avoid=item.avoid,
                mode=item.mode,
            )
            yield subgoal_id
            return

        for idx, child in enumerate(item):
            yield from self._collect_subgoals(branch_id, child, subgoals, path + (idx,))


# =========================================================
# Monitor
# =========================================================


BranchState = str  # "OPEN" | "SAT" | "VIOL"


@dataclass(frozen=True)
class StepResult:
    is_status_changed: bool
    violated: bool
    satisfied: bool


class ReqRAMonitor:
    def __init__(self, extracted: ExtractedReqSpec):
        self.extracted = extracted
        self.last_active: Set[str] = set()
        self.branch_states: Dict[int, BranchState] = {
            branch.branch_id: "OPEN" for branch in extracted.branches
        }
        self.once_satisfied: Set[SubgoalId] = set()
        self.repeat_armed: Set[SubgoalId] = set()

    def reset(self) -> None:
        self.last_active = set()
        self.branch_states = {
            branch.branch_id: "OPEN" for branch in self.extracted.branches
        }
        self.once_satisfied = set()
        self.repeat_armed = set()

    def step(self, active_aps: Set[str]) -> StepResult:
        prev_status = self.formula_status()
        prev_subtasks_sig = self._subtasks_signature(self.current_subtasks())
        self.last_active = set(active_aps)
        consumed_repeat_families = self._prepare_repeat_step()

        for branch in self.extracted.branches:
            branch_id = branch.branch_id
            if self.branch_states[branch_id] in {"SAT", "VIOL"}:
                continue

            violated = self._advance_item(branch.structure, branch_id, (), consumed_repeat_families)
            if violated:
                self.branch_states[branch_id] = "VIOL"
                continue

            filtered = self._filter_item(branch.structure, branch_id, ())
            if filtered is None or len(filtered) == 0:
                self.branch_states[branch_id] = "SAT"

        status = self.formula_status()
        curr_subtasks_sig = self._subtasks_signature(self.current_subtasks())
        is_status_changed = (
            prev_status["violated"] != status["violated"]
            or prev_status["satisfied"] != status["satisfied"]
            or prev_subtasks_sig != curr_subtasks_sig
        )
        return StepResult(
            is_status_changed=is_status_changed,
            violated=status["violated"],
            satisfied=status["satisfied"],
        )

    def _subtasks_signature(self, result: Dict[str, object]):
        if not result.get("success"):
            return (False, result.get("violated"), result.get("satisfied"), result.get("message"))
        if result.get("satisfied"):
            return (True, True, result.get("message", "spec satisfied"))
        if "res" in result:
            return (True, False, repr(result["res"]))
        return (True, False, repr(result))

    def formula_status(self) -> Dict[str, bool]:
        any_sat = any(state == "SAT" for state in self.branch_states.values())
        all_viol = all(state == "VIOL" for state in self.branch_states.values())
        return {
            "satisfied": any_sat,
            "violated": all_viol,
            "open": not any_sat and not all_viol,
        }

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

        open_branches = OrSubtasks()
        for branch in self.extracted.branches:
            if self.branch_states[branch.branch_id] != "OPEN":
                continue
            filtered = self._filter_item(branch.structure, branch.branch_id, ())
            if filtered is not None and len(filtered) > 0:
                open_branches.append(filtered)

        if len(open_branches) == 0:
            return {
                "success": False,
                "violated": True,
                "satisfied": False,
                "message": "spec violated",
            }

        return {
            "success": True,
            "violated": False,
            "satisfied": False,
            "res": open_branches,
        }

    def current_subgoals(self) -> Dict[str, object]:
        """Backward-compatible alias for current_subtasks()."""
        return self.current_subtasks()

    def _subgoal_id(self, branch_id: int, path: Tuple[int, ...]) -> SubgoalId:
        return (branch_id,) + path

    def _subgoal_violated(self, subgoal: ExtractedReqSubgoal) -> bool:
        return formula_holds(subgoal.avoid, self.last_active)

    def _subgoal_satisfied(self, subgoal: ExtractedReqSubgoal) -> bool:
        return formula_holds(subgoal.reach, self.last_active)

    def _repeat_family_key(self, subgoal: ExtractedReqSubgoal) -> Tuple[Tuple[int, ...], str, str]:
        return (subgoal.subgoal_id[1:], subgoal.avoid, subgoal.mode)

    def _iter_subgoals(
        self,
        item: Union[AndSubtasks[SeqOrRA], SeqSubtasks[StepSubtask], AndSubtasks[StrRA], StrRA],
        branch_id: int,
        path: Tuple[int, ...],
    ):
        if isinstance(item, StrRA):
            subgoal = self.extracted.subgoals[self._subgoal_id(branch_id, path)]
            yield subgoal
            return

        for idx, child in enumerate(item):
            yield from self._iter_subgoals(child, branch_id, path + (idx,))

    def _prepare_repeat_step(self) -> Set[Tuple[Tuple[int, ...], str, str]]:
        for branch in self.extracted.branches:
            if self.branch_states[branch.branch_id] != "OPEN":
                continue
            for subgoal in self._iter_subgoals(branch.structure, branch.branch_id, ()): 
                if subgoal.mode != "repeat":
                    continue
                if not self._subgoal_violated(subgoal) and self._subgoal_satisfied(subgoal):
                    self.repeat_armed.add(subgoal.subgoal_id)

        consumed_families: Set[Tuple[Tuple[int, ...], str, str]] = set()
        to_disarm: Set[SubgoalId] = set()
        for branch in self.extracted.branches:
            if self.branch_states[branch.branch_id] != "OPEN":
                continue
            for subgoal in self._iter_subgoals(branch.structure, branch.branch_id, ()): 
                if subgoal.mode != "repeat":
                    continue
                if self._subgoal_violated(subgoal) and subgoal.subgoal_id in self.repeat_armed:
                    consumed_families.add(self._repeat_family_key(subgoal))
                    to_disarm.add(subgoal.subgoal_id)

        self.repeat_armed.difference_update(to_disarm)
        return consumed_families

    def _advance_item(
        self,
        item: Union[AndSubtasks[SeqOrRA], SeqSubtasks[StepSubtask], AndSubtasks[StrRA], StrRA],
        branch_id: int,
        path: Tuple[int, ...],
        consumed_repeat_families: Set[Tuple[Tuple[int, ...], str, str]],
    ) -> bool:
        if isinstance(item, StrRA):
            subgoal = self.extracted.subgoals[self._subgoal_id(branch_id, path)]
            violated_now = self._subgoal_violated(subgoal)
            satisfied_now = self._subgoal_satisfied(subgoal)

            if subgoal.mode == "once":
                if subgoal.subgoal_id in self.once_satisfied:
                    return False
                if violated_now:
                    return True
                if satisfied_now:
                    self.once_satisfied.add(subgoal.subgoal_id)
                return False

            if violated_now:
                return self._repeat_family_key(subgoal) not in consumed_repeat_families

            if satisfied_now:
                self.repeat_armed.add(subgoal.subgoal_id)
            return False

        for idx, child in enumerate(item):
            if self._advance_item(child, branch_id, path + (idx,), consumed_repeat_families):
                return True
        return False

    def _filter_item(
        self,
        item: Union[AndSubtasks[SeqOrRA], SeqSubtasks[StepSubtask], AndSubtasks[StrRA], StrRA],
        branch_id: int,
        path: Tuple[int, ...],
    ) -> Optional[Union[AndSubtasks[SeqOrRA], SeqSubtasks[StepSubtask], AndSubtasks[StrRA], StrRA]]:
        if isinstance(item, StrRA):
            subgoal = self.extracted.subgoals[self._subgoal_id(branch_id, path)]
            if subgoal.mode == "once" and subgoal.subgoal_id in self.once_satisfied:
                return None
            return StrRA(reach=item.reach, avoid=item.avoid, mode=item.mode)

        if isinstance(item, SeqSubtasks):
            kept: List[StepSubtask] = []
            for idx, child in enumerate(item):
                filtered = self._filter_item(child, branch_id, path + (idx,))
                if filtered is not None:
                    kept.append(filtered)
            if not kept:
                return None
            return SeqSubtasks(*kept)

        if isinstance(item, AndSubtasks):
            kept: List[Union[SeqOrRA, RA]] = []
            for idx, child in enumerate(item):
                filtered = self._filter_item(child, branch_id, path + (idx,))
                if filtered is not None:
                    kept.append(filtered)
            if not kept:
                return None
            return AndSubtasks(*kept)

        raise TypeError(f"Unsupported item type: {type(item)}")


# =========================================================
# Public helpers
# =========================================================


def extract_req_subtasks(spec: ReqSpecInput) -> ExtractedReqSpec:
    return ReqRAExtractor().extract(spec)


def build_req_monitor(spec: ReqSpecInput) -> ReqRAMonitor:
    return ReqRAMonitor(extract_req_subtasks(spec))


def print_result(result: Dict[str, object]) -> None:
    if not result.get("success"):
        print(result.get("message", "unsuccessful"))
        return
    if result.get("satisfied"):
        print(result.get("message", "spec satisfied"))
        return
    if "res" in result:
        print(f"{result['res']:pretty}")
        return
    print(result)


if __name__ == "__main__":
    print('='*50)
    monitor = build_req_monitor("((blue_0 Req (green_1 | yellow_3)) & (green_1 Req magenta_2)) | (magenta_2 Req blue_0)")
    print_result(monitor.current_subtasks())
    monitor.step({"magenta_2"})
    print_result(monitor.current_subtasks())
    monitor.step({"blue_0"})
    print_result(monitor.current_subtasks())

    monitor.reset()
    print_result(monitor.current_subtasks())
    monitor.step({"green_1"})
    print_result(monitor.current_subtasks())
    print('='*50)
    monitor = build_req_monitor("(((blue_0 & yellow_4) Req (green_1 | yellow_3)) & (green_1 Req magenta_2)) | (magenta_2 Req blue_0)")
    print_result(monitor.current_subtasks())
    monitor.step({"magenta_2"})
    print_result(monitor.current_subtasks())
    monitor.step({"blue_0"})
    print_result(monitor.current_subtasks())

    # monitor.reset()
    # print_result(monitor.current_subtasks())
    # monitor.step({"green_1"})
    # print_result(monitor.current_subtasks())


    print('='*50)
    spec = "(blue_0 & green_1) Req ((magenta_2 | yellow_3) & green_0)"
    monitor = build_req_monitor(spec)
    print_result(monitor.current_subtasks())


    print('='*50)
    spec = "((blue_0 | (green_1 & green_3)) Req magenta_2) & (blue_0 EachReq (yellow_3 | blue_1))"
    monitor = build_req_monitor(spec)
    print_result(monitor.current_subtasks())
    info = monitor.step({"green_1", "magenta_2"})
    print(f"is status changed? {info.is_status_changed}")
    print_result(monitor.current_subtasks())
    info = monitor.step({"yellow_3"})
    print(f"is status changed? {info.is_status_changed}")
    info = monitor.step({"blue_0"})
    print(f"is status changed? {info.is_status_changed}")
    print_result(monitor.current_subtasks())
    info = monitor.step({"blue_1"})
    print(f"is status changed? {info.is_status_changed}")
    info = monitor.step({"blue_0"})
    print(f"is status changed? {info.is_status_changed}")
    print_result(monitor.current_subtasks())


    print(f"{'='*50}")
    spec = "(magenta_1 Req yellow_0) & (yellow_0 EachReq blue_1)"
    monitor = build_req_monitor(spec)
    monitor.step({"green_0", "green_1"})
    print_result(monitor.current_subtasks())
    monitor.step({"blue_1"})
    print_result(monitor.current_subtasks())
    monitor.step({"yellow_0", "yellow_1"})
    print_result(monitor.current_subtasks())
    
