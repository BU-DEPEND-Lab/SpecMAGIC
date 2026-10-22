from dataclasses import dataclass
from typing import Optional, List, Tuple, Set
import numpy as np
from torch import nn

from ltl.automata import LDBA, LDBATransition, LDBASequence
from ltl.logic import Assignment, FrozenAssignment
from ltl.req_cnt import AndSubtasks, OrSubtasks, SeqSubtasks, StrRA
from sequence.search import SequenceSearch


class NoPathsException(Exception):
    """
    Raised when no paths are found during search.
    """
    pass


@dataclass
class Path:
    reach_avoid: list[tuple[LDBATransition, set[LDBATransition]]]
    loop_index: int

    def __len__(self):
        return len(self.reach_avoid)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Path(self.reach_avoid[item], self.loop_index)
        return self.reach_avoid[item]

    def __str__(self):
        p = [(reach.source, {a.target for a in avoid}) for reach, avoid in self[:self.loop_index]]
        loop = [(reach.source, {a.target for a in avoid}) for reach, avoid in self[self.loop_index:]]
        return str(p + loop * 3)

    def prepend(self, reach: LDBATransition, avoid: set[LDBATransition]) -> 'Path':
        return Path([(reach, avoid)] + self.reach_avoid, self.loop_index)

    def to_sequence(self, num_loops: int) -> LDBASequence:
        # seq = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[:self.loop_index] if not r.is_epsilon()]
        # loop = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[self.loop_index:] if not r.is_epsilon()]
                
        seq = [self.reach_avoid_to_dict(r, a, "once") for r, a in self.reach_avoid[:self.loop_index] if not r.is_epsilon()]
        loop = [self.reach_avoid_to_dict(r, a, "repeat") for r, a in self.reach_avoid[self.loop_index:] if not r.is_epsilon()]
        # if any element in seq or loop is None, it means we have a conflict between reach and avoid and should return None
        if any(s is None for s in seq) or any(l is None for l in loop):
            return None
        seq = seq + loop * num_loops
        return SeqSubtasks(*seq)
        # seq = SeqSubtasks(*seq)
        return LDBASequence(seq)

    @staticmethod
    def reach_avoid_to_dict(reach: LDBATransition, avoid: set[LDBATransition], mode: str) -> dict:
        new_reach = reach.feasible_assignments
        new_avoid = [a.valid_assignments for a in avoid]
        new_avoid = set() if not new_avoid else set.union(*new_avoid)

        # new_reach = [r.to_string()[0] for r in new_reach if r.to_string()] # r.to_string()[0]
        # new_avoid = [a.to_string()[0] for a in new_avoid] # a.to_string()[0]

        new_reach = [" & ".join(rr for rr in r.to_string()) for r in new_reach if r.to_string()]
        # new_avoid = [" & ".join(aa for aa in a.to_string()) for a in new_avoid if a.to_string()]
        new_avoid = [a.to_string() for a in new_avoid] # a.to_string()[0]
        
        # print(f"DEBUG: new_reach = {new_reach}, new_avoid = {new_avoid}")
        # Remove any set that is a strict superset of another (keep only minimal sets)
        filtered_avoid = []
        avoid_sets = [set(av) for av in new_avoid]
        for i, s1 in enumerate(avoid_sets):
            if not any((s1 > s2) for j, s2 in enumerate(avoid_sets) if i != j):
                filtered_avoid.append(new_avoid[i])
        new_avoid = filtered_avoid
        new_avoid = [" & ".join(a for a in na) for na in new_avoid]

        COLORS = {"green", "blue", "yellow", "magenta"}

        def parse_term(term: str) -> frozenset[str]:
            return frozenset(x.strip() for x in term.split("&") if x.strip())

        def ap_agent(ap: str):
            import re
            m = re.fullmatch(r"(green|blue|yellow|magenta)_(\d+)", ap)
            if m is None:
                return None
            return m.group(1), int(m.group(2))

        def simplify_reach_terms(new_reach: list[str]) -> list[str]:
            from collections import defaultdict
            """
            Only applies:
                (A & green_i) | (A & blue_i) | (A & yellow_i) | (A & magenta_i) = A

            where A can itself be a conjunction.
            """

            terms = {parse_term(t) for t in new_reach if t.strip()}

            changed = True
            while changed:
                changed = False
                groups = defaultdict(set)

                for term in terms:
                    for ap in term:
                        parsed = ap_agent(ap)
                        if parsed is None:
                            continue

                        color, agent = parsed
                        base = frozenset(x for x in term if x != ap)
                        groups[(base, agent)].add(color)

                for (base, agent), colors in groups.items():
                    if colors == COLORS:
                        old_terms = {
                            frozenset(set(base) | {f"{c}_{agent}"})
                            for c in COLORS
                        }

                        if old_terms.issubset(terms):
                            terms -= old_terms
                            terms.add(base)
                            changed = True
                            break

            return [
                " & ".join(sorted(term))
                for term in sorted(terms, key=lambda t: (len(t), sorted(t)))
            ]
        new_reach = simplify_reach_terms(new_reach)

        # TODO: check conflicts between reach and avoid, e.g., if avoid containt a, then we can remove any reach that contains a
        # new_reach = [str], each str is "a & b & ..."
        # new_avoid = [str], each str is "a & b & ..."
        
        a_set_list = [set(a.split(" & ")) for a in new_avoid]
        valid_reach = []
        for r in new_reach:
            r_set = set(r.split(" & "))
            # e.g., reach = "a & b", avoid = "a", then we have conflict and should remove this reach
            if not any(a_set.issubset(r_set) for a_set in a_set_list):
                valid_reach.append(r)
        if not valid_reach:
            return None
        new_reach = valid_reach

        # print(f"DEBUG: new_reach = {new_reach}, new_avoid = {new_avoid}")
        # TODO: mat need the Req node to represent the subgoal
        subgoal = {
            'reach': " | ".join(r for r in new_reach),
            'avoid': " | ".join(a for a in new_avoid),
            'mode': mode
        }
        # print(f"DEBUG: mode = {mode}, reach = {new_reach}, avoid = {new_avoid}")
        return subgoal

    @staticmethod
    def reach_avoid_to_assignments(reach: LDBATransition, avoid: set[LDBATransition]) -> tuple[frozenset, frozenset]:
        avoid = [a.valid_assignments for a in avoid]
        avoid = set() if not avoid else set.union(*avoid)
        if reach.is_epsilon():
            new_reach = LDBASequence.EPSILON
        else:
            new_reach = frozenset(reach.feasible_assignments)
            assert new_reach  # at least one feasible assignment
        # print(f"DEBUG: reach = {new_reach}, avoid = {frozenset(avoid)}")
        return new_reach, frozenset(avoid)


class ExhaustiveSearch(SequenceSearch):
    def __init__(self, model: nn.Module, propositions, num_loops: int, value_threshold: float = 0.4):
        super().__init__(model, propositions)
        self.num_loops = num_loops
        self.value_threshold = value_threshold

    def __call__(self, ldba: LDBA, ldba_state: int, obs=None) -> LDBASequence:
        seqs = self.all_sequences(ldba, ldba_state, obs, self.num_loops)
        return max(seqs, key=lambda s: self.get_value(s, obs))

    def all_sequences(self, ldba: LDBA, ldba_state: int, obs=None, num_loops=1) -> list[LDBASequence]:
        num_loops = 0 if ldba.is_finite_specification() else num_loops
        paths = self.dfs(ldba, ldba_state, [], {}, None, obs, num_loops)
        return [path.to_sequence(num_loops) for path in paths]

    def dfs(self, ldba: LDBA, state: int, current_path: list[LDBATransition], state_to_path_index: dict[int, int],
            accepting_transition: Optional[LDBATransition], obs=None, num_loops=1) -> list[Path]:
        """
        Performs a depth-first search on the LDBA to find all simple paths leading to an accepting loop.
        """
        state_to_path_index[state] = len(current_path)
        neg_transitions = set()
        paths = []
        transition_to_max_value = {}
        for transition in ldba.state_to_transitions[state]:
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting:
                neg_transitions.add(transition)
            else:
                current_path.append(transition)
                stays_in_scc = scc == ldba.state_to_scc[transition.source]
                updated_accepting_transition = accepting_transition
                if transition.accepting and stays_in_scc:
                    updated_accepting_transition = transition
                if transition.target in state_to_path_index:  # found cycle
                    if updated_accepting_transition in current_path[state_to_path_index[transition.target]:]:
                        # found accepting cycle
                        path = Path(reach_avoid=[], loop_index=state_to_path_index[transition.target])
                        future_paths = [path]
                    else:
                        # found non-accepting cycle
                        current_path.pop()
                        if transition.source != transition.target:
                            neg_transitions.add(transition)
                        continue
                else:
                    future_paths = self.dfs(ldba, transition.target, current_path, state_to_path_index,
                                            updated_accepting_transition, obs)
                    if len(future_paths) == 0:
                        neg_transitions.add(transition)
                    else:
                        if obs is not None:
                            future_seqs = [fp.to_sequence(num_loops) for fp in future_paths]
                            max_value = max([self.get_value(s, obs) for s in future_seqs])
                            transition_to_max_value[transition] = max_value
                for fp in future_paths:
                    # avoid transitions can only be added once the recursion is finished, so only set() for now
                    paths.append(fp.prepend(transition, set()))
                current_path.pop()

        del state_to_path_index[state]
        paths = ExhaustiveSearch.prune_paths(paths)
        # for path in paths:
        #     path[0][1].update(neg_transitions)  # now we update the negative transitions
        for path in paths:
            path[0][1].update(neg_transitions)  # now we update the negative transitions
            if obs is None:
                continue
            chosen_seq = path.to_sequence(num_loops)
            chosen_value = self.get_value(chosen_seq, obs)
            for transition in ldba.state_to_transitions[state]:
                if transition in neg_transitions or transition.source == transition.target or transition == path[0][0]:
                    continue
                if transition not in transition_to_max_value:
                    continue
                alternative_value = transition_to_max_value[transition]
                if chosen_value - alternative_value > self.value_threshold:
                    path[0][1].add(transition)
        return paths

    @staticmethod
    def prune_paths(paths: list[Path]) -> list[Path]:
        to_remove = set()
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if i in to_remove or j in to_remove:
                    continue
                if len(paths[i]) < len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[j], paths[i]):
                        to_remove.add(j)
                elif len(paths[i]) > len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[i], paths[j]):
                        to_remove.add(i)
                if i in to_remove:
                    break
        paths = [paths[i] for i in range(len(paths)) if i not in to_remove]
        return paths

    @staticmethod
    def check_path_contained(path1: Path, path2: Path) -> bool:
        assert len(path2) < len(path1)
        path1 = [t[0].valid_assignments for t in path1]
        path2 = [t[0].valid_assignments for t in path2]
        acc_pos = 0
        found = False
        for p in path1:
            if p.issubset(path2[acc_pos]):
                acc_pos += 1
                if acc_pos == len(path2):
                    found = True
                    break
        return found


class ExhaustiveSearchSafety(SequenceSearch):
    def __init__(self, env, model: nn.Module, propositions, num_loops: int):
        super().__init__(model, propositions)
        self.env = env
        self.num_loops = num_loops

    def __call__(self, ldba: LDBA, ldba_states: List[int], obs=None) -> LDBASequence:
        seqs = self.all_sequences(ldba, ldba_states, obs, self.num_loops)
        valid_seqs = [seq for seq in seqs if seq is not None]
        seqs = valid_seqs
        # for seq in seqs:
        #     print(f"DEBUG: seq = {seq}")
        # processed_seqs = []
        # for seq in seqs:
        #     processed_seq = []
        #     for reach, avoid in seq:
        #         # TODO: may use the structure of the Req node
        #         subgoal = {'reach': " | ".join(r.to_string()[0] for r in reach),
        #                    'avoid': " | ".join(a.to_string()[0] for a in avoid)}
        #         processed_seq.append(subgoal)
        #     processed_seqs.append(SeqSubtasks(*processed_seq))
        #     # print(f"DEBUG: seqsubtask = {seqsubtask}")
        # processed_seqs = OrSubtasks(*processed_seqs)
        #         # print(f"DEBUG: reach = {[r.to_string()[0] for r in reach]}, avoid = {[a.to_string()[0] for a in avoid]}")
        # return processed_seqs
        return OrSubtasks(*seqs)
        processed_seqs = []
        for seq in seqs:
            reach_list, avoid = seq[0]
            suffix = seq[1:]

            # Eliminate avoid assignments that are subsets of another
            # E.g., if we have green, we can eliminate green & magenta
            avoid_assignments = []
            avoid_sets = []
            # Sort by length of true propositions: start from smallest (most general) assignments
            for a, s in sorted([(a, a.get_true_propositions()) for a in avoid], key=lambda x: len(x[1])):
                if not any(other <= s for other in avoid_sets):
                    avoid_sets.append(s)
                    avoid_assignments.append(a)
            new_avoid = frozenset(avoid_assignments)

            for reach in reach_list:
                true_props = reach.get_true_propositions()
                # Check conflicts with avoid set
                if not any(avoid_set <= true_props for avoid_set in avoid_sets):
                    new_reach = frozenset(
                        [Assignment.single_proposition(p[0], self.propositions).to_frozen()
                        for p in reach if p[1]]
                    )
                    new_seq = [(new_reach, new_avoid)] + list(suffix)
                    processed_seqs.append(LDBASequence(new_seq))

        if not processed_seqs:
            # raise NoPathsException()
            return None
        # return the fisrt subgoal regardless of value
        # print(f"DEBUG: processed_seqs = {processed_seqs}")
        # return processed_seqs[0]
        # print(f"=" * 50)
        # for s in processed_seqs:
        #     print(f"DEBUG: processed_seq = {s[0]}")

        # TODO: need return all the possible sequences
        return processed_seqs
        # return [seq[0] for seq in processed_seqs]
        
        # return min(processed_seqs, key=lambda s: len(s))
        # return max(processed_seqs, key=lambda s: self.get_value_safety([s[0]], obs))

    def all_sequences(self, ldba: LDBA, ldba_states: List[int], obs=None, num_loops=1) -> List[LDBASequence]:
        num_loops = 0 if ldba.is_finite_specification() else num_loops
        # seqs = [
        #     path.to_sequence(num_loops)
        #     for ldba_state in ldba_states
        #     for path in self.dfs(ldba, ldba_state, [], {}, None, obs, num_loops)
        # ]
        # return OrSubtasks(*seqs)
        return [
            self.path_to_strra_sequence(path, num_loops)
            for ldba_state in ldba_states
            for path in self.dfs(ldba, ldba_state, [], {}, None, obs, num_loops)
        ]

    @staticmethod
    def path_to_strra_sequence(path: Path, num_loops: int) -> SeqSubtasks:
        seq = [
            ExhaustiveSearchSafety.reach_avoid_to_strra(r, a, "once")
            for r, a in path.reach_avoid[:path.loop_index]
            if not r.is_epsilon()
        ]
        loop = [
            ExhaustiveSearchSafety.reach_avoid_to_strra(r, a, "repeat")
            for r, a in path.reach_avoid[path.loop_index:]
            if not r.is_epsilon()
        ]
        if any(s is None for s in seq) or any(l is None for l in loop):
            return None
        return SeqSubtasks(*(seq + loop * num_loops))

    @staticmethod
    def reach_avoid_to_strra(reach: LDBATransition, avoid: set[LDBATransition], mode: str) -> StrRA:
        subgoal = Path.reach_avoid_to_dict(reach, avoid, mode)
        if subgoal is None:
            return None
        return StrRA(reach=subgoal['reach'], avoid=subgoal['avoid'], mode=subgoal['mode'])

    def dfs(self, ldba: LDBA, state: int, current_path: list[LDBATransition], state_to_path_index: dict[int, int],
            accepting_transition: Optional[LDBATransition], obs=None, num_loops=1) -> list[Path]:
        """
        Performs a depth-first search on the LDBA to find all simple paths leading to an accepting loop.
        """
        state_to_path_index[state] = len(current_path)
        neg_transitions = set()
        paths = []
        # transition_to_max_value = {}
        for transition in ldba.state_to_transitions[state]:
            if not transition.is_feasible():
                # If all valid assignments are marked as unfeasible, skip this transition.
                continue
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting:
                neg_transitions.add(transition)
            else:
                current_path.append(transition)
                stays_in_scc = scc == ldba.state_to_scc[transition.source]
                updated_accepting_transition = accepting_transition
                if transition.accepting and stays_in_scc:
                    updated_accepting_transition = transition
                if transition.target in state_to_path_index:  # found cycle
                    if updated_accepting_transition in current_path[state_to_path_index[transition.target]:]:
                        # found accepting cycle
                        path = Path(reach_avoid=[], loop_index=state_to_path_index[transition.target])
                        future_paths = [path]
                    else:
                        # found non-accepting cycle
                        current_path.pop()
                        if transition.source != transition.target:
                            neg_transitions.add(transition)
                        continue
                else:
                    future_paths = self.dfs(ldba, transition.target, current_path, state_to_path_index,
                                            updated_accepting_transition, obs)
                    if len(future_paths) == 0:
                        neg_transitions.add(transition)
                for fp in future_paths:
                    # avoid transitions can only be added once the recursion is finished, so only set() for now
                    paths.append(fp.prepend(transition, set()))
                current_path.pop()

        del state_to_path_index[state]
        paths = ExhaustiveSearch.prune_paths(paths)
        for path in paths:
            path[0][1].update(neg_transitions)  # now we update the negative transitions
        return paths

    @staticmethod
    def prune_paths(paths: list[Path]) -> list[Path]:
        to_remove = set()
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if i in to_remove or j in to_remove:
                    continue
                if len(paths[i]) < len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[j], paths[i]):
                        to_remove.add(j)
                elif len(paths[i]) > len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[i], paths[j]):
                        to_remove.add(i)
                if i in to_remove:
                    break
        paths = [paths[i] for i in range(len(paths)) if i not in to_remove]
        return paths

    @staticmethod
    def check_path_contained(path1: Path, path2: Path) -> bool:
        assert len(path2) < len(path1)
        path1 = [t[0].valid_assignments for t in path1]
        path2 = [t[0].valid_assignments for t in path2]
        acc_pos = 0
        found = False
        for p in path1:
            if p.issubset(path2[acc_pos]):
                acc_pos += 1
                if acc_pos == len(path2):
                    found = True
                    break
        return found
