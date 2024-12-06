# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
from dataclasses import dataclass

# todo: clean up how RenderInstanceHelper is exposed from habitat_sim extension
from habitat_sim._ext.habitat_sim_bindings import RenderInstanceHelper

LOCAL_ROOT_KEY = "[root]"

@dataclass
class RenderAsset:
    """A render asset that can be provided to Habitat-sim ResourceManager::loadAndCreateRenderAssetInstance."""
    filepath: str
    # todo: possible color override


import numpy as np


def apply_isaac_to_habitat_orientation(orientations):
    """
    Assume wxyz quaternions. Apply 90-degree rotation about X (from Isaac z-up to Habitat y-up) and 180-degree rotation about Y (from Isaac +z-forward to Habitat -z-forward).
    """
    w_o = orientations[:, 0]
    x_o = orientations[:, 1]
    y_o = orientations[:, 2]
    z_o = orientations[:, 3]

    HALF_SQRT2 = 0.70710678  # √0.5

    w_h = -HALF_SQRT2 * (y_o + z_o)
    x_h =  HALF_SQRT2 * (z_o - y_o)
    y_h =  HALF_SQRT2 * (w_o + x_o)
    z_h =  HALF_SQRT2 * (w_o - x_o)

    new_orientations = np.stack([w_h, x_h, y_h, z_h], axis=1)

    return new_orientations


def isaac_to_habitat(positions, orientations):
    """
    Convert from Isaac (Z-up) to Habitat (Y-up) coordinate system for positions and orientations.
    """

    # Positions:
    # From Isaac (Z-up) to Habitat (Y-up): Isaac (x, y, z) → Habitat (-x, z, y)
    # We can do this in a single step with array slicing:
    new_positions = positions[:, [0, 2, 1]]  # Rearrange: (x, y, z) -> (x, z, y)
    new_positions[:, 0] *= -1  # Negate the X-axis

    # Orientations:
    # Apply the fixed transform derived above
    new_orientations = apply_isaac_to_habitat_orientation(orientations)

    return new_positions, new_orientations



class UsdVisualizer:

    def __init__(self, isaac_stage, hab_sim):

        self._prim_path_to_render_asset = {}
        self._stage = isaac_stage
        self._xform_prim_view = None
        self._were_prims_removed = False
        self._render_instance_helper = RenderInstanceHelper(hab_sim, self._get_isaac_identity_rotation_quaternion())
        pass


    def _get_isaac_identity_rotation_quaternion(self):

        from pxr import UsdGeom, Sdf
        from omni.isaac.core.prims import XFormPrimView

        # Get the current USD stage
        stage = self._stage
        
        # Define a unique path for the dummy Xform
        dummy_xform_path = "/World/DummyXform"
        if stage.GetPrimAtPath(dummy_xform_path):
            raise RuntimeError(f"Prim already exists at {dummy_xform_path}")

        # Create the dummy Xform
        UsdGeom.Xform.Define(stage, dummy_xform_path)

        # Use XFormPrimView to get the world poses of the dummy Xform
        xform_view = XFormPrimView(prim_paths_expr=dummy_xform_path)
        positions, rotations = xform_view.get_world_poses()

        # Extract the identity rotation (assuming one Xform in the view)
        identity_rotation = rotations[0]

        # Clean up: Remove the dummy Xform
        stage.RemovePrim(Sdf.Path(dummy_xform_path))

        return identity_rotation


    def on_add_reference_to_stage(self, usd_path, prim_path, strict=True):

        usd_dir = os.path.dirname(os.path.abspath(usd_path))

        root_prim_path = prim_path
        root_prim = self._stage.GetPrimAtPath(root_prim_path)

        found_count = 0

        # lazy import
        from pxr import Usd, UsdPhysics
        prim_range = Usd.PrimRange(root_prim)
        it = iter(prim_range)
        for prim in it:

            # todo: issue warnings 

            prim_path = str(prim.GetPath())

            # Retrieve habitatVisual attributes
            asset_path_attr = prim.GetAttribute("habitatVisual:assetPath")
            if not asset_path_attr or not asset_path_attr.HasAuthoredValue():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    print(f"UsdVisualizer Warning: no Habitat visual found for RigidBody prim {prim_path} in {usd_path}.")
                continue

            # we found a habitatVisual; it will visualize the entire subtree, so let's ignore children
            it.PruneChildren()

            found_count += 1

            # asset_path should be relative to the project root, which is hopefully our CWD
            asset_path = asset_path_attr.Get()
            asset_scale_attr = prim.GetAttribute("habitatVisual:assetScale")
            asset_scale = asset_scale_attr.Get() if asset_scale_attr and asset_scale_attr.HasAuthoredValue() else None

            asset_abs_path = asset_path
            self._prim_path_to_render_asset[prim_path] = RenderAsset(filepath=asset_abs_path)
            self._set_dirty()

        if not found_count:
            print(f"UsdVisualizer Warning: no Habitat visuals found for {usd_path}.")

    def on_remove_prims(self):
        # 
        self._were_prims_removed = True
        pass

    def _check_were_prims_removed(self): # todo: add underscore and rename
        if not self._were_prims_removed:
            return

        # reset flag    
        self._were_prims_removed = False

        def does_prim_exist(prim_path):
            return self._stage.GetPrimAtPath(prim_path).IsValid()

        keys_to_delete = [path for path in self._prim_path_to_render_asset if not does_prim_exist(path)]

        if len(keys_to_delete) == 0:
            return

        for key in keys_to_delete:
            del self._prim_path_to_render_asset[key]

        self._set_dirty()

    def _set_dirty(self): # todo: add underscore and rename to explain what is dirty
        self._xform_prim_view = None

    def _check_dirty(self): # todo: add underscore
        if self._xform_prim_view is not None:
            return

        # todo: handle case of no prims (empty scene)
        prim_paths = list(self._prim_path_to_render_asset.keys())

        # lazy import
        from omni.isaac.core.prims.xform_prim_view import XFormPrimView
        self._xform_prim_view = XFormPrimView(prim_paths)

        self._render_instance_helper.clear_all_instances()
        for prim_path in prim_paths:
            render_asset = self._prim_path_to_render_asset[prim_path]
            self._render_instance_helper.add_instance(render_asset.filepath)

    def flush_to_hab_sim(self, method_id):

        self._check_were_prims_removed()

        self._check_dirty()

        positions, orientations = self._xform_prim_view.get_world_poses()

        positions, orientations = isaac_to_habitat(positions, orientations)

        self._render_instance_helper.set_world_poses(
            np.ascontiguousarray(positions), 
            np.ascontiguousarray(orientations))

        