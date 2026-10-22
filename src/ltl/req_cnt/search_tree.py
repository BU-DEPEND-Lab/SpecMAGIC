import copy
import re
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable
from .subtask_lists import AndSubtasks, OrSubtasks, SeqSubtasks
from .types import StrRA


AP_RE = re.compile(r"[A-Za-z]+_\d+")


@dataclass(frozen=True)
class Subgoal:
    reach: str
    avoid: str = ""
    mode: str = "once"
    source: str = "agent"

    def __repr__(self) -> str:
        return f"({self.reach or 'empty'}, avoid={self.avoid or 'empty'}, {self.mode})"


@dataclass(frozen=True)
class Node:
    sigma: Subgoal
    active_aps: frozenset[str]

    def __repr__(self) -> str:
        return repr((self.sigma,))


NodeKey = tuple[str, str, str, frozenset[str]]


def node_key(node: Node) -> NodeKey:
    return (node.sigma.reach, node.sigma.avoid, node.sigma.mode, node.active_aps)


def canonical_node(node_pool: dict[NodeKey, Node] | None, sigma: Subgoal, active_aps: frozenset[str]) -> Node:
    node = Node(sigma=sigma, active_aps=active_aps)
    if node_pool is None:
        return node

    key = node_key(node)
    existing = node_pool.get(key)
    if existing is not None:
        return existing

    node_pool[key] = node
    return node


def atoms(expr: str) -> set[str]:
    return set(AP_RE.findall(expr or ""))


def agent_id(ap: str) -> str:
    return ap.rsplit("_", 1)[1]


def one_ap_per_agent(reach_clause: Iterable[str]) -> bool:
    seen: dict[str, str] = {}
    for ap in reach_clause:
        aid = agent_id(ap)
        if aid in seen and seen[aid] != ap:
            return False
        seen[aid] = ap
    return True


def reach_dnf(expr: str) -> list[frozenset[str]]:
    expr = (expr or "").strip()
    if not expr:
        return [frozenset()]
    # The monitor/subtask format emits positive AP formulas.  For planning, each
    # disjunct is a concrete symbolic reach option.
    return [frozenset(atoms(part)) for part in expr.split("|")]


def formula_satisfied(expr: str, active_aps: set[str]) -> bool:
    return any(clause.issubset(active_aps) for clause in reach_dnf(expr))


def avoid_terms(expr: str) -> set[frozenset[str]]:
    expr = (expr or "").strip()
    if not expr:
        return set()
    return {frozenset(atoms(part)) for part in expr.split("|") if atoms(part)}


def avoid_violated(expr: str, active_aps: set[str]) -> bool:
    return any(term.issubset(active_aps) for term in avoid_terms(expr))


def render_and(aps: Iterable[str]) -> str:
    return " & ".join(sorted(set(aps)))


def render_avoid_terms(terms: Iterable[frozenset[str]]) -> str:
    rendered = []
    for term in sorted(simplify_avoid_terms(set(terms)), key=lambda item: tuple(sorted(item))):
        rendered.append(render_and(term))
    return " | ".join(text for text in rendered if text)


def node_avoid_terms(node: Node) -> set[frozenset[str]]:
    return avoid_terms(node.sigma.avoid)


def simplify_avoid_terms(terms: set[frozenset[str]]) -> set[frozenset[str]]:
    simplified = set(terms)
    for left in terms:
        for right in terms:
            if left != right and left.issubset(right):
                simplified.discard(right)
    return simplified


def executable_avoid_expr(terms: set[frozenset[str]], reach_atoms: set[str]) -> str | None:
    """
    Simplify active avoid conditions relative to the concrete transition reach.

    A compound safety term is projected only when the current reach already
    contains part of it: avoiding yellow_0 & yellow_1 while reaching yellow_0
    means avoid yellow_1.  If the reach contains the whole term, infeasible.

    If a compound term does not overlap the reach, keep it; it still constrains
    the path to the reach.
    """
    simplified_terms: set[frozenset[str]] = set()
    for term in simplify_avoid_terms(terms):
        if term.issubset(reach_atoms):
            return None
        if term & reach_atoms:
            simplified_terms.add(frozenset(set(term) - reach_atoms))
        else:
            simplified_terms.add(term)

    simplified_terms = simplify_avoid_terms(simplified_terms)
    if not simplified_terms:
        return ""

    if all(len(term) == 1 for term in simplified_terms):
        return " | ".join(sorted(next(iter(term)) for term in simplified_terms))

    if len(simplified_terms) == 1:
        return render_and(next(iter(simplified_terms)))

    # The existing one-reach/one-avoid data shape cannot exactly express a
    # disjunction containing both singleton and compound terms.  Use a
    # conservative flat avoid to preserve safety without changing the shape.
    flat_atoms = sorted(set().union(*simplified_terms))
    return " | ".join(flat_atoms)


def executable_avoid_atoms(terms: set[frozenset[str]], reach_atoms: set[str]) -> set[str] | None:
    expr = executable_avoid_expr(terms, reach_atoms)
    if expr is None:
        return None
    return atoms(expr)


def combined_mode(subgoals: Iterable[Subgoal]) -> str:
    reach_subgoals = tuple(sg for sg in subgoals if atoms(sg.reach))
    return (
        "repeat"
        if reach_subgoals and all(sg.mode == "repeat" for sg in reach_subgoals)
        else "once"
    )


def leaf_to_subgoal(item, source: str) -> Subgoal:
    if isinstance(item, Subgoal):
        return Subgoal(item.reach, item.avoid, item.mode, source)
    if isinstance(item, StrRA):
        return Subgoal(reach=item.reach, avoid=item.avoid, mode=item.mode, source=source)
    raise TypeError(f"Unsupported subgoal leaf: {item!r}")


def normalize_subtasks(expr, source: str) -> list[list[list[Subgoal]]]:
    """
    Convert Or/And/Seq subtask syntax into alternatives of sequence pools.

    Return shape:
        alternatives -> sequence_pool -> sequence -> Subgoal

    OrSubtasks chooses one alternative, AndSubtasks creates parallel sequence
    obligations, and SeqSubtasks creates one ordered sequence.
    """
    if expr is None:
        return [[]]

    if isinstance(expr, OrSubtasks):
        alternatives: list[list[list[Subgoal]]] = []
        for child in expr:
            alternatives.extend(normalize_subtasks(child, source))
        return alternatives or [[]]

    if isinstance(expr, AndSubtasks):
        alternatives = [[]]
        for child in expr:
            child_alts = normalize_subtasks(child, source)
            alternatives = [left + right for left, right in product(alternatives, child_alts)]
        return alternatives

    if isinstance(expr, SeqSubtasks):
        seq_alternatives = [[[]]]
        for child in expr:
            child_alts = normalize_subtasks(child, source)
            next_alts: list[list[list[Subgoal]]] = []
            for prefix_pool, child_pool in product(seq_alternatives, child_alts):
                if len(prefix_pool) != 1 or len(child_pool) != 1:
                    raise ValueError("SeqSubtasks must contain ordered single-sequence children.")
                next_alts.append([prefix_pool[0] + child_pool[0]])
            seq_alternatives = next_alts
        return seq_alternatives

    if isinstance(expr, list):
        return [[[leaf_to_subgoal(child, source) for child in expr]]]

    return [[[leaf_to_subgoal(expr, source)]]]


def agent_sequence_choices(agent_subtasks: list) -> list[tuple[list[Subgoal], ...]]:
    per_agent = [normalize_subtasks(src, "agent") for src in agent_subtasks]
    choices = []
    for selected in product(*per_agent):
        seqs: list[list[Subgoal]] = []
        for pool in selected:
            seqs.extend(pool)
        choices.append(tuple(seqs))
    return choices


def first_repeat_index(seq: list[Subgoal]) -> int | None:
    for idx, sg in enumerate(seq):
        if sg.mode == "repeat":
            return idx
    return None


def advance_pointer(seq: list[Subgoal], ptr: int) -> int:
    ptr_next = ptr + 1
    if ptr_next < len(seq):
        return ptr_next
    repeat_start = first_repeat_index(seq)
    return repeat_start if repeat_start is not None else len(seq)


def in_repeat(seq: list[Subgoal], ptr: int) -> bool:
    repeat_start = first_repeat_index(seq)
    return repeat_start is not None and repeat_start <= ptr < len(seq)


def done(seq: list[Subgoal], ptr: int) -> bool:
    return ptr >= len(seq)


def front_satisfied_by_node(sg: Subgoal, node: Node) -> bool:
    if not formula_satisfied(sg.reach, set(node.active_aps)):
        return False
    return avoid_terms(sg.avoid).issubset(node_avoid_terms(node))


def advance_agent_pointers(seqs: tuple[list[Subgoal], ...], ptrs: tuple[int, ...], node: Node) -> tuple[int, ...]:
    new_ptrs = list(ptrs)
    for idx, seq in enumerate(seqs):
        # seen: set[int] = set()
        # while not done(seq, new_ptrs[idx]) and front_satisfied_by_node(seq[new_ptrs[idx]], node):
        #     if new_ptrs[idx] in seen:
        #         break
        #     seen.add(new_ptrs[idx])
        #     new_ptrs[idx] = advance_pointer(seq, new_ptrs[idx])
        if not done(seq, new_ptrs[idx]) and front_satisfied_by_node(seq[new_ptrs[idx]], node):
            new_ptrs[idx] = advance_pointer(seq, new_ptrs[idx])
    return tuple(new_ptrs)


def all_done_or_repeat(seqs: tuple[list[Subgoal], ...], ptrs: tuple[int, ...]) -> bool:
    for seq, ptr in zip(seqs, ptrs):
        if done(seq, ptr) or in_repeat(seq, ptr):
            continue
        return False
    return True


def has_repeat_obligation(seqs: tuple[list[Subgoal], ...]) -> bool:
    return any(first_repeat_index(seq) is not None for seq in seqs)


def all_done(seqs: tuple[list[Subgoal], ...], ptrs: tuple[int, ...]) -> bool:
    return all(done(seq, ptr) for seq, ptr in zip(seqs, ptrs))


def accepting_kind(
    seqs: tuple[list[Subgoal], ...],
    ptrs: tuple[int, ...],
    cycle_progress: frozenset[int],
    req_result: dict,
    cnt_result: dict,
) -> str | None:
    if not monitors_cycle_ready(req_result, cnt_result):
        return None
    if not has_repeat_obligation(seqs):
        return "ACCEPT_FINITE" if all_done(seqs, ptrs) else None
    if all_done_or_repeat(seqs, ptrs) and repeat_agent_indices(seqs, ptrs).issubset(cycle_progress):
        return "ACCEPT_CYCLE"
    return None


def any_agent_in_repeat(seqs: tuple[list[Subgoal], ...], ptrs: tuple[int, ...]) -> bool:
    return any(in_repeat(seq, ptr) for seq, ptr in zip(seqs, ptrs) if not done(seq, ptr))


def repeat_agent_indices(seqs: tuple[list[Subgoal], ...], ptrs: tuple[int, ...]) -> frozenset[int]:
    return frozenset(
        idx
        for idx, (seq, ptr) in enumerate(zip(seqs, ptrs))
        if not done(seq, ptr) and in_repeat(seq, ptr)
    )


def repeat_cycle_completed(seq: list[Subgoal], old_ptr: int, new_ptr: int, node: Node) -> bool:
    repeat_start = first_repeat_index(seq)
    if repeat_start is None or old_ptr < repeat_start or done(seq, old_ptr):
        return False
    if new_ptr == repeat_start and old_ptr != new_ptr:
        return True
    if old_ptr == new_ptr == repeat_start and old_ptr == len(seq) - 1:
        return front_satisfied_by_node(seq[old_ptr], node)
    return False


def pending_agent_once(seqs: tuple[list[Subgoal], ...], ptrs: tuple[int, ...]) -> bool:
    for seq, ptr in zip(seqs, ptrs):
        if not done(seq, ptr) and seq[ptr].mode != "repeat":
            return True
    return False


def monitor_has_once_front(result: dict, source: str) -> bool:
    for fronts in monitor_front_alternatives(result, source):
        for _, _, sg in fronts:
            if sg.mode != "repeat" and atoms(sg.reach):
                return True
    return False


def monitors_cycle_ready(req_result: dict, cnt_result: dict) -> bool:
    return (
        not monitor_has_once_front(req_result, "req")
        and not monitor_has_once_front(cnt_result, "cnt")
    )


def repeat_req_key(sg: Subgoal) -> tuple[str, str] | None:
    if sg.source == "req" and sg.mode == "repeat" and sg.avoid:
        return (sg.reach, sg.avoid)
    return None


def effective_req_subgoal(sg: Subgoal, req_credits: dict[tuple[str, str], int]) -> Subgoal:
    key = repeat_req_key(sg)
    if key is not None and req_credits.get(key, 0) > 0:
        return Subgoal(sg.reach, "", sg.mode, sg.source)
    return sg


def advance_req_credits(
    req_fronts: list[tuple[str, int, Subgoal]],
    req_credits: dict[tuple[str, str], int],
    node: Node,
    protected_avoid_atoms: set[str] | None = None,
) -> tuple[dict[tuple[str, str], int], frozenset[str]]:
    next_credits = dict(req_credits)
    req_active_aps = set(node.active_aps)
    protected_avoid_atoms = protected_avoid_atoms or set()

    for _, _, sg in req_fronts:
        key = repeat_req_key(sg)
        if key is None:
            continue
        triggered_terms = [
            term for term in avoid_terms(sg.avoid)
            if term.issubset(req_active_aps)
            and not (set(term) & protected_avoid_atoms)
        ]
        if next_credits.get(key, 0) > 0 and triggered_terms:
            next_credits[key] -= 1
            for term in triggered_terms:
                req_active_aps -= set(term)

    for _, _, sg in req_fronts:
        key = repeat_req_key(sg)
        if key is None:
            continue
        if formula_satisfied(sg.reach, set(node.active_aps)):
            next_credits[key] = min(1, next_credits.get(key, 0) + 1)

    next_credits = {key: value for key, value in next_credits.items() if value > 0}
    return next_credits, frozenset(req_active_aps)


def collect_once_avoid_atoms(item) -> set[str]:
    if isinstance(item, StrRA):
        if item.mode != "repeat":
            return atoms(item.avoid)
        return set()
    if isinstance(item, (OrSubtasks, AndSubtasks, SeqSubtasks, list)):
        out: set[str] = set()
        for child in item:
            out |= collect_once_avoid_atoms(child)
        return out
    return set()


def credits_key(req_credits: dict[tuple[str, str], int]) -> tuple:
    return tuple(sorted(req_credits.items()))


def current_agent_fronts(seqs: tuple[list[Subgoal], ...], ptrs: tuple[int, ...]) -> list[tuple[str, int, Subgoal]]:
    fronts = []
    for idx, (seq, ptr) in enumerate(zip(seqs, ptrs)):
        if not done(seq, ptr):
            fronts.append(("agent", idx, seq[ptr]))
    return fronts


def replay_monitor(monitor, history: tuple[Node, ...]):
    if monitor is None:
        return None
    current = copy.deepcopy(monitor)
    for node in history:
        current.step(set(node.active_aps))
    return current


def monitor_result(monitor, history: tuple[Node, ...]) -> dict:
    current = replay_monitor(monitor, history)
    if current is None:
        return {"success": True, "satisfied": True, "violated": False, "res": OrSubtasks()}
    if hasattr(current, "current_subtasks"):
        return current.current_subtasks()
    if hasattr(current, "current_subgoals"):
        return current.current_subgoals()
    raise TypeError(f"Monitor {current!r} has no current_subtasks/current_subgoals method.")


def monitor_front_alternatives(result: dict, source: str) -> list[list[tuple[str, int, Subgoal]]]:
    if not result.get("success", False) or result.get("violated", False):
        return []
    if result.get("satisfied", False):
        return [[]]

    alternatives = normalize_subtasks(result.get("res", OrSubtasks()), source)
    out: list[list[tuple[str, int, Subgoal]]] = []
    for pool in alternatives:
        fronts = []
        for idx, seq in enumerate(pool):
            if seq:
                fronts.append((source, idx, seq[0]))
        out.append(fronts)
    return out or [[]]


def filter_satisfied_req_item(item, satisfied_reaches: frozenset[str]):
    if isinstance(item, StrRA):
        if item.mode != "repeat" and item.reach in satisfied_reaches:
            return None
        return item

    if isinstance(item, SeqSubtasks):
        kept = [filter_satisfied_req_item(child, satisfied_reaches) for child in item]
        kept = [child for child in kept if child is not None]
        return SeqSubtasks(*kept) if kept else None

    if isinstance(item, AndSubtasks):
        kept = [filter_satisfied_req_item(child, satisfied_reaches) for child in item]
        kept = [child for child in kept if child is not None]
        return AndSubtasks(*kept) if kept else None

    if isinstance(item, OrSubtasks):
        kept = [filter_satisfied_req_item(child, satisfied_reaches) for child in item]
        kept = [child for child in kept if child is not None]
        return OrSubtasks(*kept)

    return item


def filter_satisfied_req_result(result: dict, satisfied_reaches: frozenset[str]) -> dict:
    if not satisfied_reaches or not result.get("success", False) or result.get("satisfied", False):
        return result
    filtered = dict(result)
    filtered["res"] = filter_satisfied_req_item(result.get("res", OrSubtasks()), satisfied_reaches) or OrSubtasks()
    return filtered


def collect_req_once_reaches(item, active_aps: set[str]) -> set[str]:
    if isinstance(item, StrRA):
        if item.mode != "repeat" and item.reach and formula_satisfied(item.reach, active_aps):
            return {item.reach}
        return set()
    if isinstance(item, (OrSubtasks, AndSubtasks, SeqSubtasks, list)):
        out: set[str] = set()
        for child in item:
            out |= collect_req_once_reaches(child, active_aps)
        return out
    return set()


def powerset(items):
    items = tuple(items)
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def front_agent_ids(front: tuple[str, int, Subgoal]) -> frozenset[str]:
    return frozenset(agent_id(ap) for ap in atoms(front[2].reach))


def selected_agent_ids(fronts: Iterable[tuple[str, int, Subgoal]]) -> frozenset[str]:
    out: set[str] = set()
    for front in fronts:
        out |= set(front_agent_ids(front))
    return frozenset(out)


def prioritized_subsets(
    optional: Iterable[tuple[str, int, Subgoal]],
    mandatory: Iterable[tuple[str, int, Subgoal]] = (),
    require_nonempty: bool = False,
) -> list[tuple[tuple[str, int, Subgoal], ...]]:
    mandatory_agents = selected_agent_ids(mandatory)
    subsets = [
        tuple(subset)
        for subset in powerset(optional)
        if not require_nonempty or subset
    ]
    return sorted(
        subsets,
        key=lambda subset: (
            -len(mandatory_agents | selected_agent_ids(subset)),
            -len(subset),
        ),
    )


def req_blocks_repeat_front(repeat_sg: Subgoal, fronts: list[tuple[str, int, Subgoal]]) -> bool:
    repeat_options = reach_dnf(repeat_sg.reach)
    for source, _, sg in fronts:
        if source != "req":
            continue
        req_avoid = avoid_terms(sg.avoid)
        if any(any(term.issubset(option) for term in req_avoid) for option in repeat_options):
            return True
    return False


def mandatory_fronts(fronts: list[tuple[str, int, Subgoal]], root: bool) -> list[tuple[str, int, Subgoal]]:
    required = []
    for item in fronts:
        source, _, sg = item
        if root and source == "agent":
            required.append(item)
        elif source == "agent" and sg.mode == "repeat" and not req_blocks_repeat_front(sg, fronts):
            required.append(item)
    return required


def build_nodes(
    selected: tuple[tuple[str, int, Subgoal], ...],
    active_pending: list[Subgoal],
    node_pool: dict[NodeKey, Node] | None = None,
) -> list[Node]:
    selected_subgoals = [item[2] for item in selected]
    choices = [reach_dnf(sg.reach) for sg in selected_subgoals]
    nodes: list[Node] = []
    seen: set[tuple[str, str, str]] = set()

    for clauses in product(*choices):
        reach_atoms = set().union(*clauses)
        if not one_ap_per_agent(reach_atoms):
            continue

        active_avoid_terms: set[frozenset[str]] = set()
        for sg in active_pending:
            active_avoid_terms |= avoid_terms(sg.avoid)

        avoid_expr = executable_avoid_expr(active_avoid_terms, reach_atoms)
        if avoid_expr is None:
            continue

        sigma = Subgoal(
            reach=render_and(reach_atoms),
            avoid=avoid_expr,
            mode=combined_mode(selected_subgoals),
            source="node",
        )
        key = (sigma.reach, sigma.avoid, sigma.mode)
        if key in seen:
            continue
        seen.add(key)
        nodes.append(canonical_node(node_pool, sigma, frozenset(reach_atoms)))

    return nodes


def transition_aps(previous: Node | None, current: Node) -> set[str]:
    active = set(current.active_aps)
    if previous is not None:
        active |= set(previous.active_aps)
    return active


def transition_feasible(previous: Node | None, current: Node) -> bool:
    if previous is None:
        return True
    if set(previous.active_aps) & atoms(current.sigma.avoid):
        return False
    active = transition_aps(previous, current)
    return not any(term.issubset(active) for term in node_avoid_terms(current))


def state_key(
    seqs: tuple[list[Subgoal], ...],
    ptrs: tuple[int, ...],
    req_credits: dict[tuple[str, str], int],
    req_result: dict,
    cnt_result: dict,
) -> tuple:
    seq_key = tuple(tuple((sg.reach, sg.avoid, sg.mode, sg.source) for sg in seq) for seq in seqs)
    return (
        seq_key,
        ptrs,
        credits_key(req_credits),
        repr(req_result.get("res", "")),
        bool(req_result.get("satisfied", False)),
        bool(req_result.get("violated", False)),
        repr(cnt_result.get("res", "")),
        bool(cnt_result.get("satisfied", False)),
        bool(cnt_result.get("violated", False)),
    )


def search_symbolic_tree(
    agent_subtasks: list,
    req_monitor=None,
    cnt_monitor=None,
    max_depth: int = 20,
    max_paths: int | None = None,
) -> list[list[object]]:
    """
    History-driven symbolic reach-avoid search.

    The planner does not inspect Req/Cnt internals.  At every tree node it
    replays the selected node history into the monitors, reads their current
    subtask pools, and expands combinations of all current front-most subgoals.

    Combinations are tried in an order that prioritizes involving more distinct
    agents, while all current agent avoids remain active while pending.
    """
    results: list[list[object]] = []
    result_keys: set[tuple] = set()
    node_pool: dict[NodeKey, Node] = {}

    def add_result(path: list[object]) -> None:
        key = tuple(repr(item) for item in path)
        if key in result_keys:
            return
        result_keys.add(key)
        results.append(path)

    def dfs(
        seqs: tuple[list[Subgoal], ...],
        ptrs: tuple[int, ...],
        history: tuple[Node, ...],
        req_history: tuple[Node, ...],
        req_credits: dict[tuple[str, str], int],
        cycle_progress: frozenset[int],
        path: list[object],
        seen: set[tuple],
        depth: int,
    ) -> None:
        if max_paths is not None and len(results) >= max_paths:
            return
        if depth > max_depth:
            return

        req_result = monitor_result(req_monitor, req_history)
        cnt_result = monitor_result(cnt_monitor, history)
        if not req_result.get("success", False) or not cnt_result.get("success", False):
            return
        if req_result.get("violated", False) or cnt_result.get("violated", False):
            return

        key = state_key(seqs, ptrs, req_credits, req_result, cnt_result)
        if key in seen:
            accept = accepting_kind(seqs, ptrs, cycle_progress, req_result, cnt_result)
            if accept is not None:
                add_result(path + [accept])
            return
        seen = seen | {key}

        accept = accepting_kind(seqs, ptrs, cycle_progress, req_result, cnt_result)
        if accept is not None:
            add_result(path + [accept])
            return

        agent_fronts = current_agent_fronts(seqs, ptrs)
        req_alts = monitor_front_alternatives(req_result, "req")
        cnt_alts = monitor_front_alternatives(cnt_result, "cnt")

        if not agent_fronts and req_alts == [[]] and cnt_alts == [[]]:
            add_result(path + ["ACCEPT_FINITE"])
            return

        for req_fronts, cnt_fronts in product(req_alts, cnt_alts):
            fronts = agent_fronts + req_fronts + cnt_fronts
            if not fronts:
                continue

            mandatory = mandatory_fronts(fronts, root=False)
            optional = [front for front in fronts if front not in mandatory]
            subset_iter = prioritized_subsets(
                optional,
                mandatory=mandatory,
                require_nonempty=not mandatory,
            )

            emitted: set[tuple] = set()
            for optional_subset in subset_iter:
                selected = tuple(mandatory) + tuple(optional_subset)

                # Req and agent avoids are active while their current subgoal is
                # pending.  Selected fronts determine what reach is executed.
                selected_set = set(selected)
                active_pending = []
                for item_source, _, sg in fronts:
                    if item_source == "req":
                        active_pending.append(effective_req_subgoal(sg, req_credits))
                    elif item_source == "agent":
                        active_pending.append(sg)
                    elif item_source == "cnt" and not atoms(sg.reach):
                        active_pending.append(sg)
                    elif (item_source, _, sg) in selected_set:
                        active_pending.append(sg)

                for node in build_nodes(selected, active_pending, node_pool=node_pool):
                    previous = history[-1] if history else None
                    if not transition_feasible(previous, node):
                        continue

                    next_ptrs = advance_agent_pointers(seqs, ptrs, node)
                    advanced_repeat = frozenset(
                        idx
                        for idx, (seq, old_ptr, new_ptr) in enumerate(zip(seqs, ptrs, next_ptrs))
                        if repeat_cycle_completed(seq, old_ptr, new_ptr, node)
                    )
                    next_cycle_progress = (
                        cycle_progress | advanced_repeat
                        if all_done_or_repeat(seqs, next_ptrs)
                        else frozenset()
                    )
                    next_history = history + (node,)
                    next_req_credits, req_active_aps = advance_req_credits(
                        req_fronts,
                        req_credits,
                        node,
                        protected_avoid_atoms=collect_once_avoid_atoms(req_result.get("res", OrSubtasks())),
                    )
                    req_node = Node(
                        sigma=Subgoal(node.sigma.reach, node.sigma.avoid, node.sigma.mode, node.sigma.source),
                        active_aps=req_active_aps,
                    )
                    next_req_history = req_history + (req_node,)
                    next_req_result = monitor_result(req_monitor, next_req_history)
                    next_cnt_result = monitor_result(cnt_monitor, next_history)
                    if not next_req_result.get("success", False) or not next_cnt_result.get("success", False):
                        continue
                    if next_req_result.get("violated", False) or next_cnt_result.get("violated", False):
                        continue
                    if any_agent_in_repeat(seqs, next_ptrs) and (
                        pending_agent_once(seqs, next_ptrs)
                    ):
                        continue

                    child_key = (repr(node), next_ptrs)
                    if child_key in emitted:
                        continue
                    emitted.add(child_key)

                    dfs(
                        seqs,
                        next_ptrs,
                        next_history,
                        next_req_history,
                        next_req_credits,
                        next_cycle_progress,
                        path + [node],
                        seen,
                        depth + 1,
                    )

    for seqs in agent_sequence_choices(agent_subtasks):
        dfs(
            seqs=seqs,
            ptrs=tuple(0 for _ in seqs),
            history=tuple(),
            req_history=tuple(),
            req_credits={},
            cycle_progress=frozenset(),
            path=[tuple()],
            seen=set(),
            depth=0,
        )

    return results


def first_executable_node(path: list[object]) -> Node | None:
    for item in path:
        if isinstance(item, Node):
            return item
    return None


def get_current_subgoals(paths) -> list[Node]:
    """
    Return non-duplicate first-step selected subgoal combinations.

    Accepts either one path, e.g. [(), Node(...), ..., "ACCEPT_*"], or the
    full list of paths returned by search_symbolic_tree(...).  The empty root
    and accept labels are ignored.
    """
    if not paths:
        return []

    first = paths[0]
    path_list = [paths] if isinstance(first, (Node, tuple, str)) else paths

    out: list[Node] = []
    seen: set[NodeKey] = set()
    for path in path_list:
        node = first_executable_node(path)
        if node is None:
            continue
        key = node_key(node)
        if key in seen:
            continue
        seen.add(key)
        out.append(node)
    return out


def print_paths(paths: list[list[object]], limit: int = 3) -> None:
    if not paths:
        print("No feasible global subgoal sequence.")
        return
    print(f"Found {len(paths)} feasible path(s). Showing first {min(limit, len(paths))}:")
    for path in paths[:limit]:
        for step, node in enumerate(path):
            print(f"  Step {step}: {node}")
        print()
    shortest = min(paths, key=len)
    print(f"Shortest branch has {len(shortest) - 1} step(s):")
    for step, node in enumerate(shortest):
        print(f"  Step {step}: {node}")


if __name__ == "__main__":
    from cnt_spec import build_cnt_monitor
    from req_spec import build_req_monitor

    G0 = OrSubtasks(
        SeqSubtasks(
            StrRA("yellow_0"),
            # StrRA("green_0 | blue_0", "magenta_0"),
            # StrRA("magenta_0"),
            # StrRA("green_0", "yellow_0", "repeat"),
            # StrRA("blue_0", "yellow_0", "repeat"),
        )
    )

    G1 = OrSubtasks(
        SeqSubtasks(
            StrRA("green_1", "yellow_1"),
            StrRA("yellow_1 | magenta_1"),
        ),
        SeqSubtasks(
            StrRA("magenta_1", "yellow_1"),
            StrRA("green_1", "yellow_1"),
        ),
    )

    # req_monitor = build_req_monitor("(magenta_1 Req yellow_0) & (yellow_1 EachReq green_0)")
    # req_monitor = build_req_monitor("green_0 EachReq green_1")
    req_monitor = build_req_monitor("blue_0 Req green_1")
    # req_monitor = build_req_monitor("(magenta_1 Req yellow_0) & (yellow_1 EachReq blue_1) & (blue_1 Req green_0)")
    # req_monitor = build_req_monitor("(magenta_1 Req yellow_0) & (yellow_0 Req blue_1)")
    cnt_monitor = build_cnt_monitor("!Sync[{0, 1}, yellow, 2]")
    # cnt_monitor = build_cnt_monitor("!Async[{0, 1}, yellow, 2]")

    paths = search_symbolic_tree(
        [G0, G1],
        req_monitor=req_monitor,
        cnt_monitor=cnt_monitor,
        max_depth=15,
    )
    print_paths(paths, limit=20)
