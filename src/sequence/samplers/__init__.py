from .curriculum import ZONES_CURRICULUM, ZONES_SAFETY_CURRICULUM, LETTER_CURRICULUM, LETTER_SAFETY_CURRICULUM, FLATWORLD_CURRICULUM, MA_ZONES_SAFETY_CURRICULUM
from .curriculum_sampler import CurriculumSampler
from .rrt_sampler import ReachAvoidRRTSearch, post_process_reach_avoid_sequence

curricula = {
    'PointLtl2-v0': ZONES_CURRICULUM,
    'PointLtlSafety2-v0': ZONES_SAFETY_CURRICULUM,
    'SafetyPointMultiGoal0-v0.ma3': MA_ZONES_SAFETY_CURRICULUM,
    'SafetyPointMultiGoal0-v0.ma5': MA_ZONES_SAFETY_CURRICULUM,
    'SafetyPointMultiGoal0-v0.ma10': MA_ZONES_SAFETY_CURRICULUM,
    'LetterEnv-v0': LETTER_CURRICULUM,
    'LetterSafetyEnv-v0': LETTER_SAFETY_CURRICULUM,
    'FlatWorld-v0': FLATWORLD_CURRICULUM,
}
