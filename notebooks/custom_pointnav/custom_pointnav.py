from dataclasses import dataclass

import gym
import habitat.gym
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    MeasurementConfig,
    ThirdRGBSensorConfig,
    TopDownMapMeasurementConfig,
    CollisionsMeasurementConfig,
    FogOfWarConfig,
)
from typing import TYPE_CHECKING, Union, cast

from matplotlib import pyplot as plt
# Imports
import os
from habitat.utils.visualizations import maps
import git
import gym
import imageio
import numpy as np
from hydra.core.config_store import ConfigStore

import habitat
import habitat.gym
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
    images_to_video,
)
from habitat_sim.utils import viz_utils as vut

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def insert_render_options(config):
    # Added settings to make rendering higher resolution for better visualization
    with habitat.config.read_write(config):
        config.habitat.simulator.concur_render = False
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {"third_rgb_sensor": ThirdRGBSensorConfig(height=512, width=512)}
        )
    return config

from stable_baselines3 import PPO

import importlib

# If the import block fails due to an error like "'PIL.TiffTags' has no attribute
# 'IFD'", then restart the Colab runtime instance and rerun this cell and the previous cell.
import PIL

importlib.reload(
    PIL.TiffTags  # type: ignore[attr-defined]
)  # To potentially avoid PIL problem

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(
    dir_path, "notebooks/custom_pointnav/"
)
os.makedirs(output_path, exist_ok=True)
os.chdir(dir_path)

# Load embodied AI task (Pointnav) and a pre-specified virtual robot
config = habitat.get_config(
  "/home/adonis/Documents/Thesis/Habitat/tak/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/custom_pointnav_hssd.yaml",
  overrides=["habitat.environment.max_episode_steps=3000"]
)
env = habitat.gym.make_gym_from_config(config)

vis_frames = []

observations = env.reset()

terminal = False
while not terminal:
    observations, reward, terminal, info = env.step(env.action_space.sample())
    #print(env.step(env.action_space.sample()))
    print(info)
    render_obs = observations_to_image(observations, info)
    render_obs = overlay_frame(render_obs, info)
    #print(observations.keys())
    vis_frames.append(render_obs)

images_to_video(
    vis_frames, output_path, "example_pointnav", fps=9, quality=9
)
