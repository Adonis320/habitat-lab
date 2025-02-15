#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


# This is a result of moving SimulatorActions away from core
# and into simulators specifically. As a result of that the connection points
# for our tasks and datasets for actions is coming from inside habitat-sim
# which makes it impossible for anyone to use habitat-lab without having
# habitat-sim installed. In a future PR we will implement a base simulator
# action class which will be the connection point for tasks and datasets.
# Post that PR we would no longer need try register blocks.
def _try_register_custompointnavdatasetv1():
    try:
        from habitat.datasets.custom_pointnav.custom_pointnav_dataset import (  # noqa: F401
            CustomPointNavDatasetV1,
        )

    except ImportError as e:
        pointnav_import_error = e

        @registry.register_dataset(name="CustomPointNav-v1")
        class PointnavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise pointnav_import_error
