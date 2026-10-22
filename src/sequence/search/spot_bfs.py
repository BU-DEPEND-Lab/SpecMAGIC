import os
import re
import resource
import time
from collections import deque
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from torch import nn

from ltl.req_cnt import OrSubtasks, SeqSubtasks
from ltl.req_cnt.types import StrRA
from .sequence_search import SequenceSearch


Valuation = FrozenSet[str]
Transition = Tuple[int, int, bool, object]


class SpotBFS(SequenceSearch):
    """Breadth-first reach/avoid search over a raw Spot automaton."""

    def __init__(
        self,
        env,
        model: Optional[nn.Module] = None,
        propositions: Optional[Iterable[str]] = None,
        num_loops: int = 1,
        max_depth: Optional[int] = None,
        stop_after_first_path: bool = False,
    ):
        super().__init__(model, set(propositions or ()))
        self.env = env
        self.num_loops = num_loops
        self.max_depth = max_depth
        self.stop_after_first_path = stop_after_first_path
        self.automaton = None
        self.scc_info = None
        self.accepts_all = False
        self.propositions: Set[str] = set(propositions or ())
        self._prepared_automaton = None
        # These values depend only on the prepared automaton.  Keeping them
        # here avoids repeating BDD and SCC work while leaving BFS path
        # enumeration unchanged.
        self._outgoing_cache: Dict[int, List[Transition]] = {}
        self._successor_cache: Dict[int, List[int]] = {}
        self._condition_terms_cache = {}
        self._bottom_rejecting_scc_cache: Dict[int, bool] = {}
        self._bottom_scc_cache: Dict[int, bool] = {}
        self._finite_specification_cache: Optional[bool] = None
        self.debug = os.environ.get("SPOT_BFS_DEBUG", "0").lower() not in {"", "0", "false", "no"}
        self.debug_interval = max(1, int(os.environ.get("SPOT_BFS_DEBUG_INTERVAL", "1000")))

    def __call__(self, automaton, current_states, obs=None):
        sequences = self.all_sequences(automaton, current_states)
        if not sequences:
            raise ValueError("No accepting Spot BFS path was found from the current automaton states.")
        return OrSubtasks(*sequences)

    def all_sequences(self, automaton, current_states) -> List[SeqSubtasks]:
        self._prepare(automaton)
        states = self._current_states(current_states)

        unknown = {state for state in states if state < 0 or state >= automaton.num_states()}
        if unknown:
            raise ValueError(f"Unknown current Spot automaton states: {sorted(unknown)}")

        paths = []
        for state in states:
            paths.extend(self.bfs(state))
            if self.stop_after_first_path and paths:
                break
        sequences = [self._to_sequence(path, loop_index) for path, loop_index in paths]
        return [sequence for sequence in sequences if sequence is not None]

    def bfs(self, start_state: int):
        queue = deque([(start_state, tuple(), {start_state: 0})])
        accepted = []
        min_length = None
        expanded = 0
        generated = 0
        started = time.perf_counter()

        if self.debug:
            print(
                f"SPOT_BFS start state={start_state} states={self.automaton.num_states()} "
                f"edges={self.automaton.num_edges()} aps={len(self.propositions)}",
                flush=True,
            )

        while queue:
            state, path, state_to_path_index = queue.popleft()
            expanded += 1
            if self.debug and (expanded == 1 or expanded % self.debug_interval == 0):
                elapsed = time.perf_counter() - started
                max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                print(
                    f"SPOT_BFS progress expanded={expanded} generated={generated} "
                    f"queue={len(queue)} depth={len(path)} accepted={len(accepted)} "
                    f"cached_states={len(self._outgoing_cache)} elapsed={elapsed:.1f}s "
                    f"max_rss={max_rss_mb:.1f}MB",
                    flush=True,
                )
            if min_length is not None and len(path) >= min_length:
                continue

            outgoing = self._outgoing(state)
            rejecting = [
                transition
                for transition in outgoing
                if self._is_bottom_rejecting(transition[1])
            ]
            if self.debug and (expanded == 1 or expanded % self.debug_interval == 0):
                print(
                    f"SPOT_BFS state={state} outgoing={len(outgoing)} "
                    f"rejecting={len(rejecting)}",
                    flush=True,
                )

            for transition in outgoing:
                source, target, accepting, _ = transition
                if transition in rejecting:
                    continue

                step = (transition, self._avoid_terms_for_transition(transition, rejecting))
                next_path = path + (step,)
                generated += 1

                if target in state_to_path_index:
                    loop_index = state_to_path_index[target]
                    loop = next_path[loop_index:]
                    if accepting or any(loop_step[0][2] for loop_step in loop):
                        if self._path_is_representable(next_path):
                            accepted.append((next_path, loop_index))
                            if self.stop_after_first_path:
                                return accepted
                            min_length = len(next_path)
                    continue

                if self.max_depth is not None and len(next_path) >= self.max_depth:
                    continue

                next_state_to_path_index = dict(state_to_path_index)
                next_state_to_path_index[target] = len(next_path)
                queue.append((target, next_path, next_state_to_path_index))

        return accepted

    def _prepare(self, automaton) -> None:
        if automaton is self._prepared_automaton:
            return

        import spot

        acceptance = automaton.acc()
        if not (acceptance.is_buchi() or acceptance.is_t()):
            raise RuntimeError(f"Expected Buchi acceptance, received: {acceptance.get_acceptance()}")

        self.automaton = automaton
        self.scc_info = spot.scc_info(automaton)
        self.accepts_all = acceptance.is_t()
        self.propositions = {str(ap) for ap in automaton.ap()}
        self._outgoing_cache.clear()
        self._successor_cache.clear()
        self._condition_terms_cache.clear()
        self._bottom_rejecting_scc_cache.clear()
        self._bottom_scc_cache.clear()
        self._finite_specification_cache = None
        self._prepared_automaton = automaton

    def _outgoing(self, state: int) -> List[Transition]:
        import buddy

        cached = self._outgoing_cache.get(state)
        if cached is not None:
            return cached

        transitions = []
        for edge in self.automaton.out(state):
            if edge.cond == buddy.bddfalse:
                continue
            transitions.append((state, edge.dst, self.accepts_all or edge.acc.has(0), edge.cond))
        self._outgoing_cache[state] = transitions
        return transitions

    def _successors(self, state: int) -> List[int]:
        cached = self._successor_cache.get(state)
        if cached is not None:
            return cached

        successors = [edge.dst for edge in self.automaton.out(state)]
        self._successor_cache[state] = successors
        return successors

    def _to_sequence(self, path, loop_index: int) -> SeqSubtasks:
        loops = 0 if self._is_finite_specification() else self.num_loops
        prefix = [self._to_subgoal(step, "once") for step in path[:loop_index]]
        loop = [self._to_subgoal(step, "repeat") for step in path[loop_index:]]
        if any(subgoal is None for subgoal in prefix) or any(subgoal is None for subgoal in loop):
            return None
        return SeqSubtasks(*(prefix + loop * loops))

    def _to_subgoal(self, step, mode: str) -> StrRA:
        transition, avoid_terms = step
        reach_terms = self._condition_terms(transition[3])
        reach_terms = {
            reach_term
            for reach_term in reach_terms
            if not any(avoid_term.issubset(reach_term) for avoid_term in avoid_terms)
        }
        if not reach_terms:
            return None
        reach = self._terms_to_formula(reach_terms)
        avoid = self._terms_to_formula(avoid_terms)
        return StrRA(reach=reach, avoid=avoid, mode=mode)

    def _avoid_terms_for_transition(self, transition: Transition, rejecting: Set[Transition]) -> FrozenSet[Valuation]:
        return frozenset(
            term
            for reject_transition in rejecting
            for term in self._condition_terms(reject_transition[3])
        )

    def _path_is_representable(self, path) -> bool:
        subgoals = [self._to_subgoal(step, "once") for step in path]
        if self._is_finite_specification():
            return all(subgoal is not None for subgoal in subgoals)

        return all(
            subgoal is not None
            and not (subgoal.reach == "" and subgoal.avoid == "")
            for subgoal in subgoals)

    def _is_accepting_scc(self, state: int) -> bool:
        return self.scc_info.is_accepting_scc(self.scc_info.scc_of(state))

    def _is_bottom_rejecting(self, state: int) -> bool:
        scc = self.scc_info.scc_of(state)
        cached = self._bottom_rejecting_scc_cache.get(scc)
        if cached is not None:
            return cached

        if self.scc_info.is_accepting_scc(scc):
            result = False
        else:
            states = set(self.scc_info.states_of(scc))
            result = all(
                target in states
                for s in states
                for target in self._successors(s)
            )
        self._bottom_rejecting_scc_cache[scc] = result
        return result

    def _is_finite_specification(self) -> bool:
        if self._finite_specification_cache is not None:
            return self._finite_specification_cache

        accepting = [
            frozenset(self.scc_info.states_of(scc))
            for scc in range(self.scc_info.scc_count())
            if self.scc_info.is_accepting_scc(scc)
        ]
        if len(accepting) != 1 or len(accepting[0]) != 1:
            result = False
        else:
            state = next(iter(accepting[0]))
            result = self._is_bottom_scc(self.scc_info.scc_of(state))
        self._finite_specification_cache = result
        return result

    def _is_bottom_scc(self, scc: int) -> bool:
        cached = self._bottom_scc_cache.get(scc)
        if cached is not None:
            return cached

        states = set(self.scc_info.states_of(scc))
        result = all(
            target in states
            for state in states
            for target in self._successors(state)
        )
        self._bottom_scc_cache[scc] = result
        return result

    @staticmethod
    def _current_states(current_states) -> List[int]:
        if isinstance(current_states, int):
            return [current_states]
        return list(current_states)

    def _condition_terms(self, condition) -> FrozenSet[Valuation]:
        import buddy
        import spot

        if condition == buddy.bddfalse:
            return frozenset()
        if condition == buddy.bddtrue:
            return frozenset({frozenset()})

        cached = self._condition_terms_cache.get(condition)
        if cached is not None:
            return cached

        formula = spot.bdd_format_formula(self.automaton.get_dict(), condition)
        terms = frozenset(self._formula_to_terms(formula))
        self._condition_terms_cache[condition] = terms
        return terms

    @staticmethod
    def _valuations_to_formula(valuations: Iterable[Valuation]) -> str:
        return SpotBFS._terms_to_formula(valuations)

    @staticmethod
    def _formula_to_terms(formula: str) -> Set[Valuation]:
        formula = formula.strip()
        if not formula or formula == "0":
            return set()
        if formula == "1":
            return {frozenset()}

        terms = set()
        for disjunct in SpotBFS._split_top_level(formula, "|"):
            literals = SpotBFS._split_top_level(
                SpotBFS._strip_wrapping_parentheses(disjunct),
                "&",
            )
            propositions = []
            for literal in literals:
                literal = SpotBFS._strip_wrapping_parentheses(literal).strip()
                if not literal or literal == "1" or literal.startswith("!"):
                    continue
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*_\d+", literal):
                    propositions.append(literal)
            term = frozenset(propositions)
            if SpotBFS._term_respects_agent_mutex(term):
                terms.add(term)
        if frozenset() in terms:
            return {frozenset()}
        return terms

    @staticmethod
    def _term_respects_agent_mutex(term: Valuation) -> bool:
        seen_agent_indices = set()
        for proposition in term:
            match = re.search(r"_(\d+)$", proposition)
            if match is None:
                continue
            agent_index = match.group(1)
            if agent_index in seen_agent_indices:
                return False
            seen_agent_indices.add(agent_index)
        return True

    @staticmethod
    def _split_top_level(expression: str, delimiter: str) -> List[str]:
        parts = []
        depth = 0
        start = 0
        for index, char in enumerate(expression):
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(depth - 1, 0)
            elif char == delimiter and depth == 0:
                parts.append(expression[start:index].strip())
                start = index + 1
        parts.append(expression[start:].strip())
        return [part for part in parts if part]

    @staticmethod
    def _strip_wrapping_parentheses(expression: str) -> str:
        expression = expression.strip()
        while expression.startswith("(") and expression.endswith(")"):
            depth = 0
            wraps = True
            for index, char in enumerate(expression):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(expression) - 1:
                        wraps = False
                        break
            if not wraps:
                break
            expression = expression[1:-1].strip()
        return expression

    @staticmethod
    def _terms_to_formula(terms: Iterable[Valuation]) -> str:
        return " | ".join(
            " & ".join(sorted(term))
            for term in sorted(terms, key=lambda item: (len(item), tuple(sorted(item))))
            if term
        )


SpotBFSSafety = SpotBFS
