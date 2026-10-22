# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multi Goal level 0."""

# from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.goal import GoalBlue, GoalGreen, GoalMagenta, GoalYellow
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.zones import Zones
from safety_gymnasium.tasks.safe_multi_agent.assets.mocaps.gremlins import Gremlins
# from safety_gymnasium.tasks.safe_multi_agent.tasks.ltl.ltl_base_task import LtlBaseTask


# class MultiGoalLevel0(LtlBaseTask):
class MultiGoalLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-2.7, -2.7, 2.7, 2.7] # ma3 and ma5
        # self.placements_conf.extents = [-3.2, -3.2, 3.2, 3.2] # ma10
        # self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]
        # self.placements_conf.extents = [-4.0, -4.0, 4.0, 4.0]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = None
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False

        # env boundary walls
        self._add_geoms(LtlWalls())
        # self._add_geoms(LtlWalls(locate_factor=3.5, collision_threshold=3.4, size=3.5)) # ma10

        # colored zones
        zone_size = 0.5; keepout = 0.7 # ma3 and ma5
        # zone_size = 0.6; keepout = 0.9 # ma10
        # zone_size = 0.5; keepout = 0.65
        self._add_geoms(
            Zones(color='green', size=zone_size, num=2, keepout=keepout),
            Zones(color='yellow', size=zone_size, num=2, keepout=keepout),
            Zones(color='blue', size=zone_size, num=2, keepout=keepout),
            Zones(color='magenta', size=zone_size, num=2, keepout=keepout),
            # Zones(color='green', size=zone_size, num=1, keepout=keepout),
            # Zones(color='yellow', size=zone_size, num=1, keepout=keepout),
            # Zones(color='blue', size=zone_size, num=1, keepout=keepout),
            # Zones(color='magenta', size=zone_size, num=1, keepout=keepout),
        )

        # used to represent agents, just for lidar observation of other agents
        self._add_mocaps(
            # Gremlins(num=config['agent_num'], size=0.28, dist_threshold=0.0, keepout=0.1) # ma3 and ma5
            Gremlins(num=config['agent_num'], size=0.28, dist_threshold=0.0, keepout=0.2)
        )


    def calculate_reward(self):
        # return {'agent_0': 0.0, 'agent_1': 0.0}
        return {f'agent_{i}': 0.0 for i in range(self.agent.agent_num)}

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        return (False, False)