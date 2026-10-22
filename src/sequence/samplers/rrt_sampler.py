"""Sampling-based task allocation over reach-avoid subtasks (centralized, RRT-style).

Every specification component i in I (one per agent "agent_<n>", plus the
shared "req" and "cnt" components) has a set of candidate subtask sequences
G_i = {g_i^1, ..., g_i^{M_i}}, M_i = |G_i| (``build_component_sequences``).
Each g_i^j = [sigma_{i,0}^j, sigma_{i,1}^j, ...] is a finite sequence of
reach-avoid subtasks sigma_{i,k}^j = (A_{i,k}^{+,j}, A_{i,k}^{-,j}).  The
planner only needs these sequences, so it also applies when the
specifications are given as a collection of Büchi automata.

Search tree T over the subtask space:
  * a node x (``RRTNode``) holds one progress tuple (i, j, k) per component:
    j = selected sequence g_i^j, k = current subtask sigma_{i,k}^j,
    k = |g_i^j| once the end of the sequence is reached; the node also
    stores the reach clause sampled for every advanced subtask
    (``chosen_subgoals``), so the clauses are part of the node identity;
  * (i, 0, 0) = no sequence selected yet for component i, so the root is
    x_root = {(i, 0, 0) | i in I}  (``nominal_root``);
  * x is terminal when j != 0 and k = |g_i^j| for every i  (``complete``);
    the path x_root -> terminal node yields a subtask sequence satisfying
    all specifications.

One iteration (``ReachAvoidRRTSearch.search_tree``):
  1. sample x_parent uniformly from T  (``sample_node``);
  2. extend it to a successor x_new with compatible next subtasks
     (``sample_successor``):
       a. expandable components
          I_exp = {i in I | j = 0 or k < |g_i^j| for (i, j, k) in x_parent};
       b. sample a nonempty subset I_s of I_exp (``_sample_component_orders``),
          since enumerating all subsets is intractable;
       c. for each i in I_s (``_advance_component``): if its tuple is
          (i, 0, 0), sample a sequence g_i^j in G_i and select its first
          subtask, otherwise select the next subtask of g_i^j; one
          conjunctive clause ~A_i^+ is sampled from its reach, and the tuple
          is updated to record the progress.  Tuples of i not in I_s are
          unchanged (~A_i^+ = {});
       d. compatibility (``valid_combination``):
            A_joint^+ = U_{i in I} ~A_i^+,
            A_joint^- = U_{i in I} A_i^-   (every component's full avoid:
                        omitting an unselected component's avoid could let it
                        be satisfied during execution, after which that
                        specification can no longer be satisfied);
          x_new is invalid if some conjunctive clause B in A_joint^- has
          B subset of A_joint^+;
  3. if a valid x_new is found, add it to T with edge x_parent -> x_new;
  4. if x_new is terminal, return the path x_root -> x_new
     (``stop_after_first_path``), or keep sampling until ``max_iterations``
     and collect every complete path.

Output (``extract_reach_avoid_sequence``): every edge of the path that advances
at least one subtask becomes one joint reach-avoid stage with
reach = A_joint^+ and an avoid that blocks every clause of A_joint^- with
APs outside the reach.

``ReachAvoidRRTSearch(seed)(sequences, max_iterations)`` returns the resulting
sequence (``SeqSubtasks``) or None (a tuple of all sequences when
``stop_after_first_path=False``); ``last_result`` (``RRTResult``) keeps the
tree and the paths, which ``export_tree_graphviz`` renders.
"""

import argparse
import html
import random
import re
from itertools import combinations
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, "src")


Component = str
SequenceIndex = int
ProgressIndex = int


@dataclass(frozen=True)
class ChosenSubgoal:
    reach: str
    avoid: str
    mode: str


@dataclass(frozen=True)
class RRTNode:
    """A joint search state over one chosen sequence per component.

    sequence_indices[c] is -1 until a concrete BFS alternative has been chosen
    for component c. progress[c] is the number of subtasks already advanced in
    the selected sequence. satisfied[c] is true once progress[c] consumes the
    last subgoal in the selected sequence. chosen_subgoals[c] stores the sampled
    concrete reach branch for every advanced subgoal while preserving the full
    avoid formula.
    """

    sequence_indices: Tuple[SequenceIndex, ...]
    progress: Tuple[ProgressIndex, ...]
    satisfied: Tuple[bool, ...]
    chosen_subgoals: Tuple[Tuple[ChosenSubgoal, ...], ...]


@dataclass(frozen=True)
class RRTResult:
    path: Optional[List[RRTNode]]
    nodes: Tuple[RRTNode, ...]
    parent: Mapping[RRTNode, Optional[RRTNode]]
    merged_sequence: Optional[Any] = None
    paths: Tuple[List[RRTNode], ...] = ()
    merged_sequences: Tuple[Any, ...] = ()
    predecessors: Optional[Mapping[RRTNode, Tuple[RRTNode, ...]]] = None
    iterations: Optional[int] = None
    component_orders: Mapping[Tuple[RRTNode, RRTNode], Tuple[int, ...]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ProductBFSResult:
    automaton: Any
    paths: Tuple[Any, ...]
    sequences: Tuple[Any, ...]


def component_ltl(component: Component, specs: Mapping[Component, str]) -> str:
    from ltl.req_cnt import cnt_spec, req_spec

    if component == "req":
        return req_spec.parse_req_spec(specs[component]).to_ltl().to_str("spot")
    if component == "cnt":
        return cnt_spec.parse_cnt_spec(specs[component]).to_ltl().to_str("spot")
    return specs[component]


def build_component_sequences(
    specs: Mapping[Component, str],
    current_states: Optional[Mapping[Component, int]] = None,
    num_loops: int = 1,
) -> Dict[Component, List[Any]]:
    """Return all shortest SpotBFS sequence alternatives for each component."""

    from ltl.automata import ltl2spotldba
    from sequence.search import SpotBFS

    search = SpotBFS(None, None, None, num_loops=num_loops)
    sequences = {}

    for component in specs:
        spec = component_ltl(component, specs)
        automaton = ltl2spotldba(spec)
        current_state = (
            current_states[component]
            if current_states is not None and component in current_states
            else automaton.initial_state
        )
        sequences[component] = search.all_sequences(automaton, current_state)

    return sequences


def nominal_root(num_components: int) -> RRTNode:
    return RRTNode(
        sequence_indices=tuple(-1 for _ in range(num_components)),
        progress=tuple(0 for _ in range(num_components)),
        satisfied=tuple(False for _ in range(num_components)),
        chosen_subgoals=tuple(() for _ in range(num_components)),
    )


def sample_node(nodes: Sequence[RRTNode], rng: random.Random) -> RRTNode:
    return rng.choice(nodes)


def sample_successor(
    parent: RRTNode,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
    rng: random.Random,
    return_component_order: bool = False,
) -> Optional[Any]:
    expandable = [
        i
        for i, component in enumerate(components)
        if not parent.satisfied[i] and _remaining_options(parent, i, sequences[component])
    ]
    if not expandable:
        return None

    for component_order in _sample_component_orders(expandable, components, rng):
        child = _sample_joint_candidate(
            parent,
            component_order,
            components,
            sequences,
            rng,
            return_component_order=return_component_order,
        )
        if child is None:
            continue
        if return_component_order:
            child, applied_order = child
        else:
            applied_order = ()
        if child == parent:
            continue
        if valid_combination(parent, child, components, sequences):
            return (child, applied_order) if return_component_order else child
    return None


def valid_combination(
    parent: RRTNode,
    child: RRTNode,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
) -> bool:
    """Check the joint sampled reach against every component's active avoid.

    An edge becomes one stage whose reach is the union R of the sampled reach
    clauses (each proposition must hold at some step) and whose executable
    avoid is a set of individual blockers hitting every active avoid clause
    (see _minimum_blocking_propositions).  Such blockers can be chosen outside
    R iff no active avoid clause is a subset of R, so clauses are tested
    against R rather than against each component's reach separately; e.g. the
    clause magenta_0 & magenta_1 with reaches magenta_0 and magenta_1 from two
    components has no blocker outside R.  A component already committed to a
    sequence must satisfy that sequence's active avoid.  For an uncommitted
    component, the candidate is valid when at least one of its sequence
    alternatives has a compatible first avoid.
    """

    joint_reach = set()
    for i, _component in enumerate(components):
        for subgoal in child.chosen_subgoals[i][len(parent.chosen_subgoals[i]):]:
            joint_reach.update(_extract_props(subgoal.reach))
    joint_reaches = [joint_reach]

    for i, component in enumerate(components):
        if child.sequence_indices[i] < 0:
            # The component has not committed to an alternative yet.  It is
            # enough for one alternative to remain compatible; a later sample
            # will choose an alternative and validate it again.  Preserve the
            # original no-avoid behavior when no alternatives are available.
            if not sequences[component]:
                continue
            alternative_avoids = [
                (() if not sequence else tuple(
                    avoid for avoid in (_subgoal_reach_avoid_mode(sequence[0])[1],)
                    if avoid
                ))
                for sequence in sequences[component]
            ]
            if not any(
                _avoid_formulas_compatible(avoid_formulas, joint_reaches)
                for avoid_formulas in alternative_avoids
            ):
                return False
            continue

        avoid_formulas = _active_avoid_formulas(parent, child, i, components, sequences)
        if not _avoid_formulas_compatible(avoid_formulas, joint_reaches):
            return False
    if not _at_most_one_proposition_per_agent(joint_reach):
        return False
    return True


def _avoid_formulas_compatible(
    avoid_formulas: Sequence[str],
    sampled_reaches: Sequence[set[str]],
) -> bool:
    return not any(
        _formula_matches(formula, reach_props)
        for formula in avoid_formulas
        for reach_props in sampled_reaches
    )


def complete(
    node: RRTNode,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
) -> bool:
    for i, _component in enumerate(components):
        sequence_index = node.sequence_indices[i]
        if sequence_index < 0:
            return False
        if not node.satisfied[i]:
            return False
    return True


def extract_path(parent: Mapping[RRTNode, Optional[RRTNode]], node: RRTNode) -> List[RRTNode]:
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def selected_subgoal_sequences(
    node: RRTNode,
    components: Sequence[Component],
) -> Dict[Component, Tuple[ChosenSubgoal, ...]]:
    return {
        component: node.chosen_subgoals[i]
        for i, component in enumerate(components)
    }


def extract_reach_avoid_sequence(
    path: Sequence[RRTNode],
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
):
    from ltl.req_cnt import SeqSubtasks, StrRA

    subtasks = []
    protected_reaches = {
        proposition
        for component_sequences in sequences.values()
        for sequence in component_sequences
        for subgoal in sequence
        for proposition in _extract_props(_subgoal_reach_avoid_mode(subgoal)[0])
    }
    for parent, child in zip(path, path[1:]):
        current_subgoals = _new_subgoals(parent, child, components)
        if not current_subgoals:
            continue

        reach = _format_conjunction(
            proposition
            for subgoal in current_subgoals
            for proposition in _extract_props(subgoal.reach)
        )
        current_reaches = set(_extract_props(reach))
        active_avoid_formulas = list(
            formula
            for component_index in range(len(components))
            for formula in _active_avoid_formulas(
                parent,
                child,
                component_index,
                components,
                sequences,
            )
        )
        avoid = _format_disjunction(
            _minimum_blocking_propositions(
                active_avoid_formulas,
                protected_reaches,
                preferred=current_reaches,
            )
        )
        mode = "repeat" if any(subgoal.mode == "repeat" for subgoal in current_subgoals) else "once"
        subtasks.append(StrRA(reach=reach, avoid=avoid, mode=mode))

    return SeqSubtasks(*subtasks)


def post_process_reach_avoid_sequence(sequence):
    """Greedily merge adjacent compatible reach-avoid subgoals.

    Fully compatible stages collapse as before. Otherwise, independent safe
    reaches from the next stage can start in the current stage, without
    consuming any part of the next stage. Avoids here are the executable
    individual blockers produced by extract_reach_avoid_sequence.
    """

    from ltl.req_cnt import SeqSubtasks, StrRA

    if not sequence:
        return SeqSubtasks()

    merged = []
    current = sequence[0]
    for next_subgoal in sequence[1:]:
        merged_reach = _merge_reach_if_subgoals_compatible(current, next_subgoal)
        if merged_reach is not None:
            current = StrRA(
                reach=merged_reach,
                avoid=_format_disjunction(
                    list(_extract_props(current.avoid)) + list(_extract_props(next_subgoal.avoid))
                ),
                mode="repeat" if current.mode == "repeat" or next_subgoal.mode == "repeat" else "once",
            )
        else:
            partial_reach = _overlap_compatible_reach(current, next_subgoal)
            if partial_reach is not None:
                current = StrRA(
                    reach=partial_reach,
                    avoid=_format_disjunction(
                        list(_extract_props(current.avoid)) + list(_extract_props(next_subgoal.avoid))
                    ),
                    mode="repeat" if current.mode == "repeat" or next_subgoal.mode == "repeat" else "once",
                )
            merged.append(current)
            current = next_subgoal

    merged.append(current)
    return SeqSubtasks(*merged)


class ReachAvoidRRTSearch:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.last_result: Optional[RRTResult] = None
        self.last_components: Tuple[Component, ...] = ()
        self.last_sequences: Optional[Mapping[Component, Sequence[Any]]] = None

    def __call__(
        self,
        sequences: Mapping[Component, Sequence[Any]],
        max_iterations: int,
        stop_after_first_path: bool = True,
    ):
        self.last_result = self.search_tree(
            sequences,
            max_iterations,
            stop_after_first_path=stop_after_first_path,
        )
        if stop_after_first_path:
            return self.last_result.merged_sequence
        return self.last_result.merged_sequences

    def search_tree(
        self,
        sequences: Mapping[Component, Sequence[Any]],
        max_iterations: int,
        stop_after_first_path: bool = True,
    ) -> RRTResult:
        components = tuple(sequences.keys())
        self.last_components = components
        self.last_sequences = sequences

        root = nominal_root(len(components))
        nodes = [root]
        node_set = {root}
        parent = {root: None}
        predecessors = {root: []}
        component_orders = {}
        complete_nodes = []
        complete_node_set = set()

        for iteration in range(max_iterations):
            x_parent = sample_node(nodes, self.rng)
            successor = sample_successor(
                x_parent,
                components,
                sequences,
                self.rng,
                return_component_order=True,
            )

            if successor is None:
                continue
            x_new, component_order = successor
            component_orders[(x_parent, x_new)] = component_order

            if x_new not in node_set:
                nodes.append(x_new)
                node_set.add(x_new)
                parent[x_new] = x_parent
                predecessors[x_new] = [x_parent]
            elif x_parent not in predecessors[x_new]:
                predecessors[x_new].append(x_parent)

            if complete(x_new, components, sequences):
                if x_new not in complete_node_set:
                    complete_nodes.append(x_new)
                    complete_node_set.add(x_new)
                if stop_after_first_path:
                    path = extract_path(parent, x_new)
                    merged_sequence = extract_reach_avoid_sequence(path, components, sequences)
                    return RRTResult(
                        path=path,
                        nodes=tuple(nodes),
                        parent=parent,
                        merged_sequence=merged_sequence,
                        paths=(path,),
                        merged_sequences=(merged_sequence,),
                        predecessors={
                            node: tuple(node_predecessors)
                            for node, node_predecessors in predecessors.items()
                        },
                        iterations=iteration + 1,
                        component_orders=component_orders,
                    )

        @lru_cache(maxsize=None)
        def paths_to(node: RRTNode) -> Tuple[Tuple[RRTNode, ...], ...]:
            if node == root:
                return ((root,),)
            return tuple(
                prefix + (node,)
                for predecessor in predecessors[node]
                for prefix in paths_to(predecessor)
            )

        paths = tuple(
            list(path)
            for node in complete_nodes
            for path in paths_to(node)
        )
        merged_sequences = tuple(
            extract_reach_avoid_sequence(path, components, sequences)
            for path in paths
        )
        return RRTResult(
            path=paths[0] if paths else None,
            nodes=tuple(nodes),
            parent=parent,
            merged_sequence=merged_sequences[0] if merged_sequences else None,
            paths=paths,
            merged_sequences=merged_sequences,
            predecessors={
                node: tuple(node_predecessors)
                for node, node_predecessors in predecessors.items()
            },
            iterations=max_iterations,
            component_orders=component_orders,
        )


def rrt_search(
    sequences: Mapping[Component, Sequence[Any]],
    max_iterations: int,
    seed: Optional[int] = None,
    stop_after_first_path: bool = True,
) -> RRTResult:
    return ReachAvoidRRTSearch(seed).search_tree(
        sequences,
        max_iterations,
        stop_after_first_path=stop_after_first_path,
    )


def product_bfs_search(
    specs: Mapping[Component, str],
    current_states: Optional[Mapping[Component, int]] = None,
    num_loops: int = 1,
    max_depth: Optional[int] = None,
    stop_after_first_path: bool = False,
) -> ProductBFSResult:
    """Search the synchronous product of component Spot automata directly."""

    from ltl.automata import ltl2spotldba
    from sequence.search import SpotBFS
    import spot

    if current_states:
        raise NotImplementedError(
            "product_bfs_search currently starts from the Spot product automaton initial state."
        )

    automata = []
    for component in specs:
        automata.append(ltl2spotldba(component_ltl(component, specs)).automaton)

    if not automata:
        return ProductBFSResult(automaton=None, paths=(), sequences=())

    product_automaton = automata[0]
    for automaton in automata[1:]:
        product_automaton = spot.product(product_automaton, automaton)

    search = SpotBFS(
        None,
        None,
        None,
        num_loops=num_loops,
        max_depth=max_depth,
        stop_after_first_path=stop_after_first_path,
    )
    search._prepare(product_automaton)
    paths = search.bfs(product_automaton.get_init_state_number())
    sequences = tuple(
        sequence
        for path, loop_index in paths
        if (sequence := search._to_sequence(path, loop_index)) is not None
    )
    return ProductBFSResult(
        automaton=product_automaton,
        paths=tuple(path for path, _loop_index in paths),
        sequences=sequences,
    )


def describe_node(
    node: RRTNode,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
    planned_components: Optional[Sequence[bool]] = None,
    satisfied_components: Optional[Sequence[bool]] = None,
) -> str:
    rows = []
    component_width = max((len(component) for component in components), default=0)
    for i, component in enumerate(components):
        sequence_index = node.sequence_indices[i]
        component_prefix = f"{component:>{component_width}}:"
        if sequence_index < 0:
            if _is_agent_component(component):
                agent_index = component.rsplit("_", 1)[-1]
                tuple_text = f"({agent_index}, 0, 0)"
            elif component in {"req", "cnt"}:
                tuple_text = f"({component}, 0, 0)"
            else:
                tuple_text = "<unselected>"
            rows.append((component_prefix, tuple_text, "none"))
            continue

        sequence = sequences[component][sequence_index]
        done = node.progress[i]
        component_index = (
            component.rsplit("_", 1)[-1]
            if _is_agent_component(component)
            else component
        )
        if planned_components is not None and planned_components[i]:
            chosen = node.chosen_subgoals[i] if done else ()
            if chosen:
                selected = chosen[-1]
                subtask = (
                    f"reach: {selected.reach}; "
                    f"avoid: {selected.avoid or 'none'}"
                )
            else:
                subtask = "none"
        elif node.satisfied[i]:
            subtask = "satisfied"
        else:
            subtask = "none"
        rows.append((
            component_prefix,
            f"({component_index}, {sequence_index + 1}, {done})",
            subtask,
        ))

    tuple_width = max((len(tuple_text) for _, tuple_text, _ in rows), default=0)
    entries = [
        f"{component_prefix} {tuple_text:>{tuple_width}}; subtask: {subtask}"
        for component_prefix, tuple_text, subtask in rows
    ]
    return "\n".join(entries)


def visualize_tree(
    result: RRTResult,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
) -> str:
    node_ids = {node: index for index, node in enumerate(result.nodes)}
    lines = []
    predecessors = _result_predecessors(result)
    for node in result.nodes:
        parent_ids = [node_ids[parent] for parent in predecessors[node]]
        label = "root" if not parent_ids else f"parents={parent_ids}"
        lines.append(f"node {node_ids[node]} ({label})")
        for detail in describe_node(node, components, sequences).splitlines():
            lines.append(f"  {detail}")
    return "\n".join(lines)


def export_tree_graphviz(
    result: RRTResult,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
    dot_path: Path,
    png_path: Optional[Path] = None,
) -> Optional[Path]:
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(tree_to_dot(result, components, sequences), encoding="utf-8")

    if png_path is None or shutil.which("dot") is None:
        return None

    subprocess.run(
        ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    return png_path


def tree_to_dot(
    result: RRTResult,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
) -> str:
    node_ids = {node: index for index, node in enumerate(result.nodes)}
    rendered_paths = result.paths or ((result.path,) if result.path else ())
    path_nodes = {
        node
        for path in rendered_paths
        for node in path
    }
    path_edges = {
        (parent, child)
        for path in rendered_paths
        for parent, child in zip(path, path[1:])
    }
    predecessors = _result_predecessors(result)
    lines = [
        "digraph RRT {",
        "  graph [rankdir=LR, overlap=false, splines=true];",
        "  node [shape=box, fontname=Courier, fontsize=9];",
        "  edge [color=\"#666666\"];",
    ]

    for node in result.nodes:
        node_id = node_ids[node]
        parent = predecessors[node][0] if predecessors[node] else None
        planned_order = (
            result.component_orders.get((parent, node), ())
            if parent is not None
            else ()
        )
        grandparent = predecessors[parent][0] if parent and predecessors[parent] else None
        satisfied_order = (
            result.component_orders.get((grandparent, parent), ())
            if grandparent is not None
            else ()
        )
        planned_components = tuple(
            i in planned_order for i in range(len(components))
        ) if parent is not None else None
        satisfied_components = tuple(
            i in satisfied_order for i in range(len(components))
        ) if parent is not None else None
        attrs = {
            "label": _dot_label(
                node_id,
                node,
                components,
                sequences,
                planned_components,
                satisfied_components,
            ),
            "style": "rounded,filled",
            "fillcolor": "#fff7d6" if node in path_nodes else "#f6f8fa",
            "color": "#d18f00" if node in path_nodes else "#8c959f",
            "penwidth": "2" if node in path_nodes else "1",
        }
        lines.append(f"  n{node_id} [{_dot_attrs(attrs)}];")

    for node in result.nodes:
        for parent in predecessors[node]:
            attrs = {
                "color": "#d18f00" if (parent, node) in path_edges else "#666666",
                "penwidth": "2" if (parent, node) in path_edges else "1",
            }
            lines.append(
                f"  n{node_ids[parent]} -> n{node_ids[node]} [{_dot_attrs(attrs)}];"
            )

    lines.append("}")
    return "\n".join(lines) + "\n"


def _result_predecessors(result: RRTResult) -> Mapping[RRTNode, Tuple[RRTNode, ...]]:
    if result.predecessors is not None:
        return result.predecessors
    return {
        node: (() if result.parent[node] is None else (result.parent[node],))
        for node in result.nodes
    }


def _remaining_options(
    node: RRTNode,
    component_index: int,
    component_sequences: Sequence[Any],
) -> bool:
    if node.satisfied[component_index]:
        return False
    sequence_index = node.sequence_indices[component_index]
    if sequence_index < 0:
        return bool(component_sequences)
    return node.progress[component_index] < len(component_sequences[sequence_index])


def _sample_component_orders(
    expandable: Sequence[int],
    components: Sequence[Component],
    rng: random.Random,
) -> List[List[int]]:
    """Sample candidate joint orders.

    Req is always active through validation. It is attempted first when
    expandable, but if its reach is currently blocked, fallback orders let
    agents advance without moving req.
    """

    expandable_set = set(expandable)
    agent_indices = [
        i for i, component in enumerate(components)
        if i in expandable_set and _is_agent_component(component)
    ]
    req_indices = [
        i for i, component in enumerate(components)
        if i in expandable_set and component == "req"
    ]
    cnt_indices = [
        i for i, component in enumerate(components)
        if i in expandable_set and component == "cnt"
    ]
    other_indices = [
        i for i in expandable
        if i not in set(agent_indices + req_indices + cnt_indices)
    ]

    rng.shuffle(agent_indices)
    rng.shuffle(cnt_indices)
    rng.shuffle(other_indices)

    optional_cnt = [index for index in cnt_indices if rng.random() < 0.7]
    sampled_agents = _sample_many(agent_indices, rng)
    all_agents = list(agent_indices)
    random_order = _sample_random_order(expandable, rng)

    orders = [
        req_indices + optional_cnt + sampled_agents + other_indices,
        random_order,
        # optional_cnt + sampled_agents + other_indices,
        # all_agents + optional_cnt + other_indices,
    ]
    return [order for order in orders if order]


def _sample_many(indices: Sequence[int], rng: random.Random) -> List[int]:
    if not indices:
        return []
    subset_sizes = list(range(1, len(indices) + 1))
    weights = [size * size for size in subset_sizes]
    subset_size = rng.choices(subset_sizes, weights=weights, k=1)[0]
    return rng.sample(list(indices), subset_size)


def _sample_random_order(indices: Sequence[int], rng: random.Random) -> List[int]:
    sampled = _sample_many(list(indices), rng)
    rng.shuffle(sampled)
    return sampled


def _sample_joint_candidate(
    parent: RRTNode,
    component_order: Sequence[int],
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
    rng: random.Random,
    return_component_order: bool = False,
) -> Optional[Any]:
    child = parent
    applied_order = []
    for component_index in component_order:
        previous_subgoal_count = len(child.chosen_subgoals[component_index])
        candidate = _advance_component(child, component_index, components, sequences, rng)
        if candidate is None:
            continue
        child = candidate
        if len(child.chosen_subgoals[component_index]) > previous_subgoal_count:
            applied_order.append(component_index)
    if return_component_order:
        return child, tuple(applied_order)
    return child


def _advance_component(
    parent: RRTNode,
    component_index: int,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
    rng: random.Random,
) -> Optional[RRTNode]:
    component = components[component_index]
    sequence_index = parent.sequence_indices[component_index]

    if sequence_index < 0:
        # A component chooses one BFS sequence the first time it is expanded.
        # After that, all descendants must keep using that same sequence.
        sequence_index = rng.randrange(len(sequences[component]))

    selected_sequence = sequences[component][sequence_index]
    if parent.progress[component_index] >= len(selected_sequence):
        # Also record the chosen alternative: an empty sequence is selected
        # and finished in one step, and complete() requires a selection.
        next_sequence_indices = list(parent.sequence_indices)
        next_sequence_indices[component_index] = sequence_index
        next_satisfied = list(parent.satisfied)
        next_satisfied[component_index] = True
        return RRTNode(
            tuple(next_sequence_indices),
            parent.progress,
            tuple(next_satisfied),
            parent.chosen_subgoals,
        )

    raw_subgoal = selected_sequence[parent.progress[component_index]]
    reach, avoid, mode = _subgoal_reach_avoid_mode(raw_subgoal)
    reach_candidates = _split_dnf_terms(reach)
    if not reach_candidates:
        return None

    next_sequence_indices = list(parent.sequence_indices)
    next_progress = list(parent.progress)
    next_satisfied = list(parent.satisfied)
    next_chosen_subgoals = [list(subgoals) for subgoals in parent.chosen_subgoals]

    next_sequence_indices[component_index] = sequence_index
    next_progress[component_index] += 1
    next_satisfied[component_index] = next_progress[component_index] >= len(selected_sequence)
    next_chosen_subgoals[component_index].append(
        ChosenSubgoal(
            reach=rng.choice(reach_candidates),
            avoid=avoid,
            mode=mode,
        )
    )

    for i, parent_sequence_index in enumerate(parent.sequence_indices):
        if parent_sequence_index >= 0:
            assert next_sequence_indices[i] == parent_sequence_index

    return RRTNode(
        tuple(next_sequence_indices),
        tuple(next_progress),
        tuple(next_satisfied),
        tuple(tuple(subgoals) for subgoals in next_chosen_subgoals),
    )


def _is_agent_component(component: Component) -> bool:
    return bool(re.fullmatch(r"agent_\d+", component))


def _subgoal_reach_avoid_mode(subgoal) -> Tuple[str, str, str]:
    if hasattr(subgoal, "reach") and hasattr(subgoal, "avoid"):
        return subgoal.reach, subgoal.avoid, getattr(subgoal, "mode", "once")
    raise TypeError(f"Unsupported subgoal type in RRT prototype: {type(subgoal)!r}")


@lru_cache(maxsize=4096)
def _split_dnf_terms(expression: str) -> Tuple[str, ...]:
    """Split DNF reach formula into top-level disjunctive candidates."""

    expression = expression.strip()
    if not expression:
        return ("",)

    terms = []
    depth = 0
    start = 0

    for index, char in enumerate(expression):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "|" and depth == 0:
            terms.append(expression[start:index].strip())
            start = index + 1

    terms.append(expression[start:].strip())
    return tuple(term for term in terms if term)


@lru_cache(maxsize=16384)
def _extract_props(expression: str) -> Tuple[str, ...]:
    if not expression:
        return ()
    return tuple(re.findall(r"[A-Za-z][A-Za-z0-9]*_\d+", expression))


def _at_most_one_proposition_per_agent(propositions: Iterable[str]) -> bool:
    seen_agent_indices = set()
    for proposition in propositions:
        agent_index = _agent_index(proposition)
        if agent_index is None:
            continue
        if agent_index in seen_agent_indices:
            return False
        seen_agent_indices.add(agent_index)
    return True


def _merge_reach_preserving_first_agent_ap(*expressions: str) -> str:
    """Merge reach terms while keeping only the earliest AP per agent index."""

    propositions = []
    seen_propositions = set()
    seen_agent_indices = set()
    for expression in expressions:
        for proposition in _extract_props(expression):
            if proposition in seen_propositions:
                continue
            agent_index = _agent_index(proposition)
            if agent_index is not None:
                if agent_index in seen_agent_indices:
                    continue
                seen_agent_indices.add(agent_index)
            seen_propositions.add(proposition)
            propositions.append(proposition)
    return _format_conjunction(propositions)


def _agent_index(proposition: str) -> Optional[str]:
    match = re.search(r"_(\d+)$", proposition)
    if match is None:
        return None
    return match.group(1)


def _active_avoid_formulas(
    parent: RRTNode,
    child: RRTNode,
    component_index: int,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
) -> Tuple[str, ...]:
    new_subgoals = child.chosen_subgoals[component_index][len(parent.chosen_subgoals[component_index]):]
    if new_subgoals:
        return tuple(subgoal.avoid for subgoal in new_subgoals if subgoal.avoid)
    if child.satisfied[component_index]:
        return ()

    sequence_index = child.sequence_indices[component_index]
    component = components[component_index]
    if sequence_index < 0:
        return tuple(
            _subgoal_reach_avoid_mode(sequence[0])[1]
            for sequence in sequences[component]
            if sequence and _subgoal_reach_avoid_mode(sequence[0])[1]
        )

    sequence = sequences[component][sequence_index]
    if child.progress[component_index] >= len(sequence):
        return ()
    avoid = _subgoal_reach_avoid_mode(sequence[child.progress[component_index]])[1]
    return (avoid,) if avoid else ()


def _formula_matches(formula: str, propositions: set[str]) -> bool:
    return any(
        set(_extract_props(term)).issubset(propositions)
        for term in _split_dnf_terms(formula)
        if _extract_props(term)
    )


def _minimum_blocking_propositions(
    formulas: Sequence[str],
    protected: set[str],
    preferred: Optional[set[str]] = None,
) -> Tuple[str, ...]:
    """Choose executable individual avoids while preserving DNF semantics."""
    clauses = [
        set(_extract_props(term))
        for formula in formulas
        for term in _split_dnf_terms(formula)
        if _extract_props(term)
    ]
    if not clauses:
        return ()

    preferred = preferred or set()
    candidates = sorted(set().union(*clauses))
    best = None
    best_score = None
    for size in range(len(candidates) + 1):
        for selection in combinations(candidates, size):
            selected = set(selection)
            if not all(selected & clause for clause in clauses):
                continue
            score = (
                len(selected & preferred),
                len(selected & protected),
                len(selected),
                selection,
            )
            if best_score is None or score < best_score:
                best, best_score = selection, score
        if best_score is not None and best_score[:2] == (0, 0):
            break
    return tuple(best or ())


def _new_subgoals(
    parent: RRTNode,
    child: RRTNode,
    components: Sequence[Component],
) -> List[ChosenSubgoal]:
    subgoals = []
    for i, _component in enumerate(components):
        subgoals.extend(child.chosen_subgoals[i][len(parent.chosen_subgoals[i]):])
    return subgoals


def _format_conjunction(propositions: Iterable[str]) -> str:
    return " & ".join(_unique_sorted(propositions))


def _format_disjunction(propositions: Iterable[str]) -> str:
    return " | ".join(_unique_sorted(propositions))


def _merge_reach_if_subgoals_compatible(left, right) -> Optional[str]:
    left_reach = set(_extract_props(left.reach))
    left_avoid = set(_extract_props(left.avoid))
    right_reach = set(_extract_props(right.reach))
    right_avoid = set(_extract_props(right.avoid))
    if not left_reach.isdisjoint(right_avoid):
        return None
    if not left_avoid.isdisjoint(right_reach):
        return None

    left_by_agent = {
        _agent_index(proposition): proposition
        for proposition in left_reach
        if _agent_index(proposition) is not None
    }
    if any(
        (agent_index := _agent_index(proposition)) in left_by_agent
        and left_by_agent[agent_index] != proposition
        for proposition in right_reach
    ):
        return None

    return _merge_reach_preserving_first_agent_ap(
        left.reach,
        right.reach,
    )


def _overlap_compatible_reach(left, right) -> Optional[str]:
    """Copy safe reaches for idle agents; the caller must retain right in full."""
    left_reach = set(_extract_props(left.reach))
    combined_avoid = set(_extract_props(left.avoid)) | set(_extract_props(right.avoid))
    if not left_reach.isdisjoint(combined_avoid):
        return None

    occupied_agents = {_agent_index(proposition) for proposition in left_reach}
    additions = {
        proposition
        for proposition in _extract_props(right.reach)
        if proposition not in combined_avoid
        and _agent_index(proposition) not in occupied_agents
    }
    if not additions or not _at_most_one_proposition_per_agent(left_reach | additions):
        return None
    return _format_conjunction(left_reach | additions)


def _unique_sorted(propositions: Iterable[str]) -> List[str]:
    return sorted(set(propositions), key=lambda item: (item.rsplit("_", 1)[-1], item))


def _dot_label(
    node_id: int,
    node: RRTNode,
    components: Sequence[Component],
    sequences: Mapping[Component, Sequence[Any]],
    planned_components: Optional[Sequence[bool]] = None,
    satisfied_components: Optional[Sequence[bool]] = None,
) -> str:
    component_width = max((len(component) for component in components), default=0)
    rows = [
        '<TR><TD COLSPAN="3" ALIGN="LEFT">'
        f"node {node_id}"
        "</TD></TR>"
    ]
    parsed_rows = []
    for line in describe_node(
        node,
        components,
        sequences,
        planned_components,
        satisfied_components,
    ).splitlines():
        component, _, remainder = line.partition(": ")
        tuple_text, _, subtask = remainder.partition("; subtask: ")
        parsed_rows.append((component.strip(), tuple_text.strip(), subtask))

    tuple_width = max((len(tuple_text) for _, tuple_text, _ in parsed_rows), default=0)
    for component, tuple_text, subtask in parsed_rows:
        component_text = (
            html.escape(component.rjust(component_width)).replace(" ", "&#160;")
            + ":"
        )
        tuple_display = html.escape(tuple_text.rjust(tuple_width)).replace(" ", "&#160;")
        rows.append(
            "<TR>"
            f'<TD ALIGN="LEFT">{component_text}</TD>'
            f'<TD ALIGN="LEFT">{tuple_display}</TD>'
            f'<TD ALIGN="LEFT">; subtask: {html.escape(subtask)}</TD>'
            "</TR>"
        )
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        + "".join(rows)
        + "</TABLE>"
    )


def _dot_attrs(attrs: Mapping[str, str]) -> str:
    rendered = []
    for key, value in attrs.items():
        if key == "label" and value.startswith("<") and value.endswith(">"):
            rendered.append(f"{key}=<{value}>")
        else:
            rendered.append(f'{key}="{_dot_escape(value)}"')
    return ", ".join(rendered)


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-paths",
        action="store_true",
        help="Keep sampling until max iterations and return every distinct complete path.",
    )
    parser.add_argument("--max-iterations", type=int, default=100_000)
    args = parser.parse_args()
    seed = 1
    formula = {
        "agent_0": "!(blue_0 | yellow_0) U (green_0 & F magenta_0)",
        "agent_1": "(F blue_1) & (!blue_1 U (green_1 & F yellow_1))",
        "req": "(green_0 Req green_1) & (yellow_1 Req magenta_0)",
        "cnt": "Sync[{0, 1}, green, 2] & Async[{0, 1}, magenta, 2]",
    }
    component_sequences = build_component_sequences(formula)
    components = tuple(component_sequences.keys())

    for component, sequences in component_sequences.items():
        print(f"\n=== {component}: {len(sequences)} sequence candidate(s) ===")
        for index, sequence in enumerate(sequences):
            print(f"[{index}] {sequence:pretty}")

    search = ReachAvoidRRTSearch(seed=seed)
    start_time = time.perf_counter()
    search_output = search(
        component_sequences,
        max_iterations=args.max_iterations,
        stop_after_first_path=not args.all_paths,
    )
    final_sequence = (
        search_output[0] if args.all_paths and search_output else search_output
    )
    elapsed = time.perf_counter() - start_time
    result = search.last_result
    assert result is not None

    if result.path is None:
        print(f"\nRRT result: FAILURE after {elapsed:.3f}s")
    else:
        print(f"\nRRT result: found path with {len(result.path)} node(s) in {elapsed:.3f}s")
        print(f"\n=== final merged reach-avoid sequence ===")
        print(f"{final_sequence:pretty}")
        for step, node in enumerate(result.path):
            print(f"\n--- path node {step} ---")
            print(describe_node(node, components, component_sequences))
        if args.all_paths:
            print(f"\n=== all complete paths: {len(result.paths)} ===")
            for index, sequence in enumerate(result.merged_sequences):
                print(f"\n--- complete path {index} ---")
                print(f"{sequence:pretty}")

    dot_path = Path(f"vis/rrt_tree_s{seed}.dot")
    png_path = Path(f"vis/rrt_tree_s{seed}.png")
    rendered_path = export_tree_graphviz(result, components, component_sequences, dot_path, png_path)
    print(f"\nGraphviz DOT written to: {dot_path}")
    if rendered_path is None:
        print("Graphviz PNG not rendered because the 'dot' executable was not found.")
    else:
        print(f"Graphviz PNG written to: {rendered_path}")
