# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.spatial.transform import Rotation as R

import magnum as mn

# todo: add guard to ensure SimulatorApp is created, or give nice error message, so we don't get weird import errors here
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView

from pxr import UsdGeom, Sdf, Usd, UsdPhysics, PhysxSchema
import omni.physx.scripts.utils as physxUtils

from habitat.isaac_sim import isaac_prim_utils

class RobotWrapper:
    """Isaac-internal wrapper for a robot.
    
    The goal with this wrapper is convenience but not encapsulation. See also (public) IsaacMobileManipulator, which has the goal of exposing a minimal public interface to the rest of Habitat-lab.
    """
    def __init__(self, isaac_service, instance_id=0):

        self._isaac_service = isaac_service
        asset_path = "./data/usd/robots/hab_spot_arm.usda"
        robot_prim_path = f"/World/env_{instance_id}/Spot"
        self._robot_prim_path = robot_prim_path

        add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)
        self._isaac_service.usd_visualizer.on_add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)

        robot_prim = self._isaac_service.world.stage.GetPrimAtPath(robot_prim_path)

        if not robot_prim.IsValid():
            raise ValueError(f"Prim at {robot_prim_path} is not valid.")

        # Traverse only the robot's prim hierarchy
        for prim in Usd.PrimRange(robot_prim):

            if prim.HasAPI(PhysxSchema.PhysxJointAPI):
                joint_api = PhysxSchema.PhysxJointAPI(prim)
                joint_api.GetMaxJointVelocityAttr().Set(1000.0)

            if prim.HasAPI(UsdPhysics.DriveAPI):
                # Access the existing DriveAPI
                drive_api = UsdPhysics.DriveAPI(prim, "angular")
                if drive_api:

                    # Modify drive parameters
                    drive_api.GetStiffnessAttr().Set(10.0)  # Position gain
                    drive_api.GetDampingAttr().Set(0.1)     # Velocity gain
                    drive_api.GetMaxForceAttr().Set(1000)  # Maximum force/torque

                drive_api = UsdPhysics.DriveAPI(prim, "linear")
                if drive_api:

                    drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
                    drive_api.GetStiffnessAttr().Set(1000)  # Example for linear stiffness

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):

                # UsdPhysics.RigidBodyAPI doesn't support damping but PhysxRigidBodyAPI does
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                else:
                    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

                # todo: decide hard-coded values here
                physx_api.CreateLinearDampingAttr(50.0)
                physx_api.CreateAngularDampingAttr(10.0)

        # todo: investigate if this is needed for kinematic base
        # todo: resolve off-by-100 scale issue
        self.scale_prim_mass_and_inertia(f"{robot_prim_path}/base", 100.0)

        self._name = f"robot_{instance_id}"
        self._robot = self._isaac_service.world.scene.add(Robot(prim_path=robot_prim_path, name=self._name))
        self._robot_controller = self._robot.get_articulation_controller()

        self._step_count = 0
        self._lateral_vel = [0.0, 0.0]
        self._instance_id = instance_id

        self.arm_target_joint_pos = None

    @property
    def robot(self) -> Robot:
        """Get Isaac Sim Robot"""
        return self._robot

    def get_prim_path(self):
        return self._robot_prim_path

    def post_reset(self):
        # todo: just do a single callback
        self._isaac_service.world.add_physics_callback(f"{self._name}_physics_callback", callback_fn=self.physics_callback)

        # todo: specify this in isaac_spot_robot.py
        arm_joint_names = ["arm0_sh0", "arm0_sh1", "arm0_hr0", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"]
        gripper_joint_names = ["arm0_f1x"]

        arm_joint_indices = []
        dof_names = self._robot.dof_names
        assert len(dof_names) > 0
        for arm_joint_name in arm_joint_names:
            arm_joint_indices.append(dof_names.index(arm_joint_name))

        self._arm_joint_indices = np.array(arm_joint_indices)

    def scale_prim_mass_and_inertia(self, path, scale):

        prim = self._isaac_service.world.stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.MassAPI)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.GetMassAttr().Set(mass_api.GetMassAttr().Get() * scale)
        mass_api.GetDiagonalInertiaAttr().Set(mass_api.GetDiagonalInertiaAttr().Get() * scale)


    def fix_base_orientation_via_angular_vel(self, step_size):

        curr_angular_velocity = self._robot.get_angular_velocity()

        _, base_orientation = self._robot.get_world_pose()

        # Constants
        max_angular_velocity = 3.0  # Maximum angular velocity (rad/s)

        # wxyz to xyzw
        base_orientation_xyzw = np.array([base_orientation[1], base_orientation[2], base_orientation[3], base_orientation[0]])
        base_rotation = R.from_quat(base_orientation_xyzw)

        # Define the local "up" axis and transform it to world space
        local_up = np.array([0, 0, 1])  # Object's local up axis (hack: -1?)
        world_up = base_rotation.apply(local_up)  # Local up in world space

        # Define the global up axis
        global_up = np.array([0, 0, 1])  # Global up direction

        # Compute the axis of rotation to align world_up to global_up
        rotation_axis = np.cross(world_up, global_up)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        # Handle special cases where no rotation is needed
        if rotation_axis_norm < 1e-6:  # Already aligned
            desired_angular_velocity = np.array([0, 0, 0])  # No correction needed
        else:
            # Normalize the rotation axis
            rotation_axis /= rotation_axis_norm

            # Compute the angle of rotation using the dot product
            rotation_angle = np.arccos(np.clip(np.dot(world_up, global_up), -1.0, 1.0))

            # Calculate the angular velocity to correct the tilt in one step
            hack_scale = 1.0
            tilt_correction_velocity = ((rotation_axis * rotation_angle) / step_size) * hack_scale

            # Cap the angular velocity to the maximum allowed value
            angular_velocity_magnitude = np.linalg.norm(tilt_correction_velocity)
            if angular_velocity_magnitude > max_angular_velocity:
                tilt_correction_velocity *= max_angular_velocity / angular_velocity_magnitude

            desired_angular_velocity = tilt_correction_velocity

        # don't change rotation about z
        desired_angular_velocity[2] = curr_angular_velocity[2]

        self._robot.set_angular_velocity(desired_angular_velocity)


    def fix_base_height_via_linear_vel_z(self, step_size):

        curr_linear_velocity = self._robot.get_linear_velocity()

        z_target = 0.7  # todo: get from navmesh or assume ground_z==0
        max_linear_vel = 3.0

        base_position, _ = self._robot.get_world_pose()

        # Extract the vertical position and velocity
        z_current = base_position[2]

        # Compute the position error
        position_error = z_target - z_current    

        desired_linear_vel_z = position_error / step_size
        desired_linear_vel_z = max(-max_linear_vel, min(max_linear_vel, desired_linear_vel_z))

        self._robot.set_linear_velocity([curr_linear_velocity[0], curr_linear_velocity[1], desired_linear_vel_z])

    def fix_base(self, step_size):

        # temp: introduce random xy vel periodically
        # if self._step_count % 120 == 0:
        #     base_position, _ = self._robot.get_world_pose()
        #     self._lateral_vel = np.array([-base_position[0], -base_position[1]])
        #     self._lateral_vel += np.array([-self._lateral_vel[1], self._lateral_vel[0]]) * 0.2
        #     self._lateral_vel /= np.linalg.norm(self._lateral_vel)
        #     self._lateral_vel *= 20.0

        # self._lateral_vel = np.array([0.0, 0.0])

        self.fix_base_height_via_linear_vel_z(step_size)
        self.fix_base_orientation_via_angular_vel(step_size)

    def physics_callback(self, step_size):
        self.fix_base(step_size)
        self._step_count += 1

