from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import tempfile

from utils import memory, timeit
from .owl import run_owl
from .rabinizer import run_rabinizer
from .ldba import LDBA, LDBATransition
from .ldba_sequence import LDBASequence


class SpotLDBA:
    """Small adapter exposing the LDBA methods used by LDBAWrapper."""

    def __init__(self, automaton, propositions: list[str] = None):
        import spot

        self.automaton = automaton
        self.propositions = tuple(sorted(str(ap) for ap in automaton.ap())) if propositions is None else tuple(sorted(propositions))
        self.initial_state = automaton.get_init_state_number()
        self.accepts_all = automaton.acc().is_t()
        self.scc_info = spot.scc_info(automaton)

    def __getattr__(self, name):
        return getattr(self.automaton, name)

    def get_next_states(self, state: int, propositions: set[str], take_epsilon=False):
        import buddy

        if take_epsilon:
            raise ValueError('Spot automata do not expose epsilon transitions through SpotLDBA.')

        valuation = self._valuation_bdd(set(propositions))
        outcomes = []
        for edge in self.automaton.out(state):
            if edge.cond & valuation != buddy.bddfalse:
                outcomes.append((edge.dst, self._edge_accepting(edge)))
        if not outcomes:
            raise ValueError('Invalid transition.')
        return outcomes

    def is_finite_specification(self) -> bool:
        accepting = [
            frozenset(self.scc_info.states_of(scc))
            for scc in range(self.scc_info.scc_count())
            if self.scc_info.is_accepting_scc(scc)
        ]
        if len(accepting) != 1 or len(accepting[0]) != 1:
            return False
        return self._is_bottom_scc(self.scc_info.scc_of(next(iter(accepting[0]))))

    def is_state_violating(self, state: int) -> bool:
        scc = self.scc_info.scc_of(state)
        return not self.scc_info.is_accepting_scc(scc) and self._is_bottom_scc(scc)

    def _is_bottom_scc(self, scc: int) -> bool:
        states = set(self.scc_info.states_of(scc))
        return all(
            edge.dst in states
            for state in states
            for edge in self.automaton.out(state)
        )

    def _edge_accepting(self, edge) -> bool:
        if self.accepts_all or edge.acc.has(0):
            return True
        state_is_accepting = getattr(self.automaton, 'state_is_accepting', None)
        if state_is_accepting is None:
            return False
        return state_is_accepting(edge.src) or state_is_accepting(edge.dst)

    def _valuation_bdd(self, true_propositions: set[str]):
        import buddy

        result = buddy.bddtrue
        for proposition in self.propositions:
            var = self.automaton.register_ap(proposition)
            result &= buddy.bdd_ithvar(var) if proposition in true_propositions else buddy.bdd_nithvar(var)
        return result


@memory.cache
def ltl2ldba(formula: str, propositions: list[str] = None, simplify_labels=True, print_time=False) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    from ltl.hoa import HOAParser
    with timeit(f'Converting LTL formula "{formula}" to LDBA') if print_time else nullcontext():
        # print(f"DEBUG: starting construction")
        hoa = run_rabinizer(formula)
        # print(f"DEBUG: starting exporting hoa")
    # print(f"DEBUG: Rabinizer HOA for {formula}: {hoa}")
    return HOAParser(formula, hoa, propositions, simplify_labels=simplify_labels).parse_hoa()


@memory.cache
def ltl2nba(formula: str, propositions: list[str] = None, simplify_labels=True, print_time=False) -> LDBA:
    """Converts an LTL formula to a NBA using the OWL tool."""
    from ltl.hoa import HOAParser
    with timeit(f'Converting LTL formula "{formula}" to NBA') if print_time else nullcontext():
        hoa = run_owl(formula)
    return HOAParser(formula, hoa, propositions, simplify_labels=simplify_labels).parse_hoa()


# @memory.cache
# def ltl2spotldba(formula: str, propositions: list[str] = None, simplify_labels=True, print_time=False) -> LDBA:
#     """Converts an LTL formula to a Buchi automaton using Spot and stores it as an LDBA."""
#     from ltl.hoa import HOAParser
#     with timeit(f'Converting LTL formula "{formula}" to Buchi automaton with Spot') if print_time else nullcontext():
#         import spot
#         print(f"DEBUG: starting construction")
#         aut = spot.translate(formula, 'Buchi', 'state-based', 'complete', 'high')
#         print(f"DEBUG: starting exporting hoa")
#         hoa = aut.to_str('hoa')
#     print(f"DEBUG: Spot HOA for {formula}: {hoa}")
#     return HOAParser(formula, hoa, propositions, simplify_labels=simplify_labels).parse_hoa()


def ltl2spotldba(formula: str, propositions: list[str] = None, simplify_labels=True, print_time=False) -> LDBA:
    """Loads or creates a cached Spot Buchi automaton."""
    with timeit(f'Converting LTL formula "{formula}" to Buchi automaton with Spot') if print_time else nullcontext():
        import spot

        cache_dir = Path('/home/zjguo/code/Spec-Guided-MARL/cache')
        cache_key = hashlib.sha256(json.dumps({
            'version': 1,
            'formula': formula,
            'translation': ['Buchi', 'state-based', 'complete'],
        }, sort_keys=True).encode('utf-8')).hexdigest()
        cache_path = cache_dir / f'{cache_key}.hoa'

        if cache_path.is_file():
            # print(f"DEBUG: loading cached Spot automaton for {formula} from {cache_path}")
            aut = spot.automaton(str(cache_path))
        else:
            # print(f"DEBUG: creating new Spot automaton for {formula} from {cache_path}")
            aut = spot.translate(formula, 'Buchi', 'state-based', 'complete')
            cache_dir.mkdir(parents=True, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(
                prefix=f'.{cache_key}.',
                suffix='.hoa',
                dir=cache_dir,
                text=True,
            )
            try:
                with os.fdopen(fd, 'w') as cache_file:
                    cache_file.write(aut.to_str('hoa'))
                os.replace(temporary_path, cache_path)
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)

        print(f"DEBUG: automaton num_states = {aut.num_states()}, num_edges = {aut.num_edges()}")
    return SpotLDBA(aut, propositions)


def spot2ldba(aut, formula: str = None, propositions: list[str] = None, simplify_labels=True) -> LDBA:
    """Converts a Spot automaton object directly to this repo's LDBA representation."""
    import spot

    if aut.num_sets() > 1:
        raise ValueError('spot2ldba only supports Buchi automata with at most one acceptance set.')

    aut_propositions = [str(ap) for ap in aut.ap()]
    propositions = set(aut_propositions if propositions is None else propositions)
    ldba = LDBA(propositions, formula=formula, simplify_labels=simplify_labels)

    for state in range(aut.num_states()):
        ldba.add_state(state, initial=(state == aut.get_init_state_number()))

    for source in range(aut.num_states()):
        for edge in aut.out(source):
            accepting = str(edge.acc) != '{}' or aut.state_is_accepting(source)
            label = spot.bdd_format_formula(aut.get_dict(), edge.cond)
            if label == '1':
                label = 't'
            elif label == '0':
                continue
            ldba.add_transition(edge.src, edge.dst, label, accepting)

    return ldba


__all__ = ['LDBASequence', 'LDBA', 'LDBATransition', 'SpotLDBA', 'ltl2ldba', 'ltl2nba', 'ltl2spotldba', 'spot2ldba']


# __all__ = ['LDBASequence', 'LDBA', 'LDBATransition', 'ltl2ldba', 'ltl2nba', 'ltl2spotldba']
