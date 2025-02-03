#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random

import magnum as mn
import numpy as np
import quaternion

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.navmesh_utils import (
    embodied_unoccluded_navmesh_snap,
)
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import (
    place_agent_at_dist_from_pos,
    rearrange_logger,
    set_agent_base_via_obj_trans,
)


@registry.register_task(name="RearrangePickTask-v0")
class RearrangePickTaskV1(RearrangeTask):
    DISTANCE_TO_RECEPTACLE = 1.0
    """
    Rearrange Pick Task with Fetch robot interacting with objects and environment.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            should_place_articulated_agent=False,
            **kwargs,
        )

        self.prev_colls = None
        self.force_set_idx = None
        self._base_angle_noise = self._config.base_angle_noise
        self._spawn_max_dist_to_obj = self._config.spawn_max_dist_to_obj
        self._num_spawn_attempts = self._config.num_spawn_attempts
        self._filter_colliding_states = self._config.filter_colliding_states
        self._spawn_max_dist_to_obj_delta = (
            self._config.spawn_max_dist_to_obj_delta
        )
        self._spawn_type = self._config.get("spawn_type", "orig_snap")
        self.initial_base_pos = None
        self.initial_base_rot = None

    def set_args(self, obj, **kwargs):
        self.force_set_idx = obj

    def _get_targ_pos(self, sim):
        scene_pos = sim.get_scene_pos()
        targ_idxs = sim.get_targets()[0]
        return scene_pos[targ_idxs]

    def _sample_idx(self, sim):
        if self.force_set_idx is not None:
            idxs = self._sim.get_targets()[0]
            sel_idx = self.force_set_idx
            sel_idx = list(idxs).index(sel_idx)
        else:
            sel_idx = np.random.randint(0, len(self._get_targ_pos(sim)))
        return sel_idx

    def _gen_start_pos(self, sim, episode, sel_idx):
        if np.all(episode.start_position != [0.0, 0.0, 0.0]):
            start_pos = mn.Vector3(*episode.start_position)
            angle_to_obj = episode.start_rotation[-1]
            return start_pos, angle_to_obj
        target_positions = self._get_targ_pos(sim)
        targ_pos = target_positions[sel_idx]

        was_fail = True
        spawn_attempt_count = 0

        while was_fail and spawn_attempt_count < self._num_spawn_attempts:
            if self._spawn_type == "embodied_unoccluded_navmesh_snap":
                (
                    start_pos,
                    angle_to_obj,
                    success,
                ) = embodied_unoccluded_navmesh_snap(
                    target_position=mn.Vector3(targ_pos),
                    height=1.5,  # NOTE: this is default agent max height. This parameter is used to determine whether or not a point is occluded.
                    sim=sim,
                    island_id=sim._largest_indoor_island_idx,
                    search_offset=self._spawn_max_dist_to_obj
                    + spawn_attempt_count * self._spawn_max_dist_to_obj_delta,
                    orientation_noise=self._base_angle_noise,
                    max_samples=self._num_spawn_attempts,
                    target_object_ids=[
                        sel_idx
                    ],  # TODO: this must be the integer id of the target object or no unoccluded state will be found because this object will be considered occluding
                    agent_embodiment=(
                        sim.articulated_agent
                        if self._filter_colliding_states
                        else None
                    ),
                )
                print(f"UNOCCLUDED NAVMESH SNAP POS: {start_pos}")

                was_fail = not success
                if (
                    (start_pos is None)
                    or (angle_to_obj is None)
                    or (was_fail is None)
                ):
                    start_pos, angle_to_obj, was_fail = (
                        place_agent_at_dist_from_pos(
                            targ_pos,
                            self._base_angle_noise,
                            self._spawn_max_dist_to_obj
                            + spawn_attempt_count
                            * self._spawn_max_dist_to_obj_delta,
                            sim,
                            self._num_spawn_attempts,
                            self._filter_colliding_states,
                        )
                    )
                    print(
                        f"UNOCCLUDED NAVMESH SNAP FAILED. GENEARING NEW START_POS WITH OLD METHOD: {start_pos}"
                    )
            else:
                start_pos, angle_to_obj, was_fail = (
                    place_agent_at_dist_from_pos(
                        targ_pos,
                        self._base_angle_noise,
                        self._spawn_max_dist_to_obj
                        + spawn_attempt_count
                        * self._spawn_max_dist_to_obj_delta,
                        sim,
                        self._num_spawn_attempts,
                        self._filter_colliding_states,
                    )
                )
            spawn_attempt_count += 1

        if was_fail:
            rearrange_logger.error(
                f"Episode {episode.episode_id} failed to place robot"
            )
        print("start_pos: ", start_pos, angle_to_obj)
        return start_pos, angle_to_obj

    def _should_prevent_grip(self, action_args):
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] < 0
        )

    def step(self, action, episode):
        action_args = action["action_args"]

        if self._should_prevent_grip(action_args):
            # No releasing the object once it is held.
            action_args["grip_action"] = None
        obs = super().step(action=action, episode=episode)
        return obs

    def _set_arm_to_target_pose(self):
        """Set the arm to the target pose."""
        # Set the arm based on the random selection
        robot_init_arm_angle = np.copy(
            self._sim.get_agent_data(None).articulated_agent.arm_joint_pos
        )
        # Set the arm to a pose
        self._sim.get_agent_data(None).articulated_agent.arm_joint_pos = (
            np.array(random.choice(self._config.semantic_pick_target_arm_pose))
        )
        # Get the initial EE orientation at the time of begining of placing
        _, ee_orientation = self._sim.get_agent_data(
            None
        ).articulated_agent.get_ee_local_pose()  # type: ignore
        self.target_obj_orientation = quaternion.quaternion(ee_orientation)
        # Revert the robot arm to the initial pose
        self._sim.get_agent_data(None).articulated_agent.arm_joint_pos = (
            robot_init_arm_angle
        )

    def reset(self, episode: Episode, fetch_observations: bool = True):
        sim = self._sim
        assert isinstance(
            episode, RearrangeEpisode
        ), "Provided episode needs to be of type RearrangeEpisode for RearrangePickTaskV1"

        super().reset(episode, fetch_observations=False)

        self.prev_colls = 0

        sel_idx = self._sample_idx(sim)
        start_pos, start_rot = self._gen_start_pos(sim, episode, sel_idx)

        set_agent_base_via_obj_trans(
            start_pos, start_rot, sim.articulated_agent
        )

        self.initial_base_pos = self._sim.articulated_agent.base_pos
        self.initial_base_rot = self._sim.articulated_agent.base_rot

        self._targ_idx = sel_idx

        # Set the arm to the target pose, and the revert it
        if self._config.get("semantic_pick_training", False):
            self._set_arm_to_target_pose()
        if self._config.get("topdown_side_training", False):
            self.grasping_type = np.random.choice(["topdown", "side"])

        if fetch_observations:
            self._sim.maybe_update_articulated_agent()
            return self._get_observations(episode)
        return None
