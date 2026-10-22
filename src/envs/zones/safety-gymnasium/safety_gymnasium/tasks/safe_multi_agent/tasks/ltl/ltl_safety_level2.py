"""LTL level 2."""

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.zones import Zones
from safety_gymnasium.tasks.safe_multi_agent.tasks.ltl.ltl_base_task import LtlBaseTask
# from safety_gymnasium.world import World


class MultiAgentLtlSafetyLevel2(LtlBaseTask):
    """Two green, two yellow, two blue, and two magenta zones."""

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)

        self._add_geoms(
            Zones(color='green', size=self.zone_size, num=2),
            Zones(color='yellow', size=self.zone_size, num=2),
            Zones(color='blue', size=self.zone_size, num=2),
            Zones(color='magenta', size=self.zone_size, num=2)
        )
