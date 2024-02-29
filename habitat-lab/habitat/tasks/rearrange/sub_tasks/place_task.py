#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np
import quaternion

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.pick_task import RearrangePickTaskV1
from habitat.tasks.rearrange.utils import get_angle_to_pos_xyz


@registry.register_task(name="RearrangePlaceTask-v0")
class RearrangePlaceTaskV1(RearrangePickTaskV1):
    def _get_targ_pos(self, sim):
        return sim.get_targets()[1]

    def _should_prevent_grip(self, action_args):
        # Never allow regrasping
        return (
            not self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] >= 0
        )

    def get_object_pose_by_id(self, obj_idx):
        """Get the object quaternion for a given object index."""
        # Get the object transformation
        rom = self._sim.get_rigid_object_manager()
        obj_transform = rom.get_object_by_id(obj_idx).transformation
        base_transform = self._sim.get_agent_data(
            None
        ).articulated_agent.base_transformation
        # Get the local ee location (x,y,z)
        base_T_obj_transform = base_transform.inverted() @ obj_transform

        # Get the local ee orientation (roll, pitch, yaw)
        local_obj_quat = quaternion.from_rotation_matrix(
            base_T_obj_transform.rotation()
        )
        return local_obj_quat

    def get_object_pose_by_target(self):
        """Get the object quaternion for a given object index."""

        # Get the object transformation
        obj_transform = self._sim._get_target_trans()[self.targ_idx][-1]

        # Get the base transformation
        base_transform = self._sim.get_agent_data(
            None
        ).articulated_agent.base_transformation

        # Get the ee transformation
        ee_transform = self._sim.get_agent_data(
            None
        ).articulated_agent.ee_transform()

        # Compute the rotation offset from the base to target object
        local_obj = base_transform.inverted() @ obj_transform
        angle_to_object = get_angle_to_pos_xyz(local_obj.translation)

        # Rotate the base transform by the angle
        base_rotate_trans = mn.Matrix4.rotation_z(mn.Rad(angle_to_object))
        base_transform = base_transform @ base_rotate_trans

        # Get the local ee location relative to base
        base_T_ee_transform = base_transform.inverted() @ ee_transform

        # Get the local ee orientation (roll, pitch, yaw)
        local_obj_quat = quaternion.from_rotation_matrix(
            base_T_ee_transform.rotation()
        )
        return local_obj_quat

    def get_keep_T(self, abs_obj_idx):
        """This is simulate top down grasping"""
        rom = self._sim.get_rigid_object_manager()
        ro = rom.get_object_by_id(abs_obj_idx)
        ee_T = self._sim.get_agent_data(None).articulated_agent.ee_transform()
        obj_in_ee_T = ee_T.inverted() @ ro.transformation
        obj_in_ee_T.translation = mn.Vector3(0, 0, 0)
        return obj_in_ee_T

    def random_arm(self):
        """This function randomizes the arm joints to get a diverse training data"""
        arm_joint_pos = self._sim.get_agent_data(
            None
        ).articulated_agent.arm_joint_pos
        arm_joint_limit = self._config.actions.arm_action.arm_joint_limit
        arm_joint_mask = self._config.actions.arm_action.arm_joint_mask
        assert (
            len(arm_joint_mask)
            == len(self._config.joint_start_noise_multiplier)
            or self._config.joint_start_noise_multiplier is None
        ), "Arm noise size mismatch"
        new_arm_joint_pos = []
        j = 0
        for i in range(len(arm_joint_pos)):
            if arm_joint_mask[i]:
                # Can change the arm joint angle
                target_arm = (
                    arm_joint_pos[i]
                    + np.random.uniform(-1, 1)
                    * self._config.joint_start_noise
                    * self._config.joint_start_noise_multiplier[i]
                )
                _min = arm_joint_limit[j][0]
                _max = arm_joint_limit[j][1]
                target_arm = np.clip(target_arm, _min, _max)
                j += 1
            else:
                # Cannot change the arm joint angle
                target_arm = arm_joint_pos[i]
            new_arm_joint_pos.append(target_arm)

        # Set the arm
        self._sim.get_agent_data(
            None
        ).articulated_agent.arm_joint_pos = new_arm_joint_pos
        # Update the initial ee orientation
        _, self.init_ee_orientation = self._sim.get_agent_data(
            None
        ).articulated_agent.get_ee_local_pose()  # type: ignore

    def reset(self, episode: Episode):
        sim = self._sim
        # Remove whatever the agent is currently holding.
        sim.grasp_mgr.desnap(force=True)

        super().reset(episode, fetch_observations=False)

        abs_obj_idx = sim.scene_obj_ids[self.abs_targ_idx]

        # Get the initial object orientation
        # This is the object orientation (in ee frame) at grasping moment
        # We will like the robot to match such init_obj_orientation when dropping the object
        # The sensor will be relative orientation to the initial object orientation
        if self._config.enable_rotation_target:
            self.target_obj_orientation = self.get_object_pose_by_target()
        else:
            self.target_obj_orientation = self.get_object_pose_by_id(
                abs_obj_idx
            )

        # Here, we teleport the target object to the gripper
        # The place task is to let Spot place the object in the original
        # object location
        top_down_grasp = (
            np.random.random() > 1.0 - self._config.top_down_grasp_ratio
            and self._config.top_down_grasp
        )
        if top_down_grasp:
            # We do top down grasping here
            sim.grasp_mgr._keep_T = self.get_keep_T(abs_obj_idx)
        else:
            # We do side grasping here
            sim.grasp_mgr._keep_T = None
        sim.grasp_mgr.snap_to_obj(abs_obj_idx, force=True)

        self.was_prev_holding = self.targ_idx

        sim.internal_step(-1)
        self._sim.maybe_update_articulated_agent()

        # Get the initial EE orientation at the time of begining of placing
        _, self.init_ee_orientation = self._sim.get_agent_data(
            None
        ).articulated_agent.get_ee_local_pose()  # type: ignore

        # We update the initial object orientation here to be the gripper orientation if
        # it is top down grasping
        if top_down_grasp and not self._config.enable_rotation_target:
            self.target_obj_orientation = quaternion.quaternion(
                self.init_ee_orientation
            )

        # We want to change the arm joint after doing top_down_grasp
        if self._config.fix_obj_rotation_change_arm_joint and top_down_grasp:
            self.random_arm()

        return self._get_observations(episode)
