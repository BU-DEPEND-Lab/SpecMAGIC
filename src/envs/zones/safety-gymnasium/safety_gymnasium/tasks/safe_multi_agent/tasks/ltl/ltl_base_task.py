"""LTL level 0."""

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask


class LtlBaseTask(BaseTask):
    """Base task for LTL tasks."""

    def __init__(self, config, zone_size: float, walls=True) -> None:
        super().__init__(config=config)
        self.zone_size = zone_size
        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = None
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False
        if walls:
            self._add_geoms(LtlWalls())

    def calculate_reward(self):
        return 0

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        return False
