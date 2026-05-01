import mediapy as media
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from datetime import datetime
import importlib

import wandb
import os, json

import Pupper_Loco_Manipulation.utils.export as export

import Pupper_Loco_Manipulation.sim.utils.utils as utils
import Pupper_Loco_Manipulation.sim.domain_randomization as domain_randomization

from Pupper_Loco_Manipulation.configs.config import get_total_config

np.set_printoptions(precision=3, suppress=True, linewidth=100)


def run_train(reward_config=None):
    CONFIG = get_total_config()
    env_name = 'pupper'

    importlib.reload(utils)


    train_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    try:
        wandb.init(entity=None,
                project="pupperv3-mjx-rl",
                config=CONFIG.to_dict(),
                save_code=True,
                settings={
                    "_service_wait": 90,
                    "init_timeout": 90
                })
        print("Using personal W&B to log training progress.")
    except wandb.errors.CommError:
        print("W&B failed to initialize. Training without logging (follow W&B specific cells may fail)")

    try:
        wandb.run.summary["benchmark_physics_steps_per_sec"] = physics_steps_per_sec
    except:
        pass

    # Save and reload params.
    output_folder = f"output_{wandb.run.name}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create and JIT reset and step functions for use in in-training policy
    # video creation if they don't already exist from a previous step
    if ("jit_reset" in globals() or "jit_reset" in locals()) and (
        "jit_step" in globals() or "jit_step" in locals()
    ):
        print("JIT'd step and reset functions already defined. " "Using them for policy visualization.")
    else:
        print("Creating and JIT'ing step and reset functions")
        policy_viz_env = envs.get_environment(env_name)
        jit_reset = jax.jit(policy_viz_env.reset)
        jit_step = jax.jit(policy_viz_env.step)


    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=CONFIG.policy.hidden_layer_sizes,
        activation=utils.activation_fn_map(CONFIG.policy.activation)
    )
    train_fn = functools.partial(
        ppo.train,
        **(CONFIG.training.ppo.to_dict()),
        network_factory=make_networks_factory,
        randomization_fn=functools.partial(
            domain_randomization.domain_randomize,
            friction_range=CONFIG.training.friction_range,
            kp_multiplier_range=CONFIG.training.position_control_kp_multiplier_range,
            kd_multiplier_range=CONFIG.training.position_control_kd_multiplier_range,
            body_com_x_shift_range=CONFIG.training.body_com_x_shift_range,
            body_com_y_shift_range=CONFIG.training.body_com_y_shift_range,
            body_com_z_shift_range=CONFIG.training.body_com_z_shift_range,
            body_mass_scale_range=CONFIG.training.body_mass_scale_range,
            body_inertia_scale_range=CONFIG.training.body_inertia_scale_range,
        ),
        seed=28,
    )

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    env = envs.get_environment(env_name)
    eval_env = envs.get_environment(env_name)

    def policy_params_fn(current_step, make_policy, params):
        utils.visualize_policy(current_step=current_step,
                            make_policy=make_policy,
                            params=params,
                            eval_env=eval_env,
                            jit_step=jit_step,
                            jit_reset=jit_reset,
                            output_folder=output_folder)
        utils.save_checkpoint(current_step=current_step,
                            make_policy=make_policy,
                            params=params,
                            checkpoint_path=output_folder)

    from pathlib import Path
    checkpoint_kwargs = {}
    # if CONFIG.training.checkpoint_run_number is not None:
    #   utils.download_checkpoint(entity_name=ENTITY,
    #                             project_name="pupperv3-mjx-rl",
    #                             run_number=CONFIG.training.checkpoint_run_number,
    #                             save_path="checkpoint")
    #   checkpoint_kwargs["restore_checkpoint_path"]=Path("checkpoint").resolve()

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=functools.partial(
            utils.progress,
            times=times,
            x_data=x_data,
            y_data=y_data,
            ydataerr=ydataerr,
            num_timesteps= CONFIG.training.ppo.num_timesteps,
            min_y=0,
            max_y=40,
        ),
        eval_env=eval_env,
        policy_params_fn=policy_params_fn,
        **checkpoint_kwargs
    )

    print(f"time to jit: {times[1] - times[0]}")
    wandb.run.summary["time_to_jit"] = (times[1] - times[0]).total_seconds()
    wandb.run.summary["time_to_train"] = (times[-1] - times[1]).total_seconds()

    # Save params to a model
    model_path = os.path.join(output_folder, f'mjx_params_{train_datetime}')
    model.save_params(model_path, params)

    params_rtneural = export.convert_params(jax.block_until_ready(params),
                                        activation=CONFIG.policy.activation,
                                        action_scale=CONFIG.policy.action_scale,
                                        kp=CONFIG.training.position_control_kp,
                                        kd=CONFIG.training.dof_damping,
                                        default_pose=CONFIG.training.default_pose,
                                        joint_upper_limits=CONFIG.simulation.joint_upper_limits,
                                        joint_lower_limits=CONFIG.simulation.joint_lower_limits,
                                        use_imu=CONFIG.policy.use_imu,
                                        observation_history=CONFIG.policy.observation_history,
                                        final_activation="tanh",
                                        )

    name = f"policy.json"
    saved_policy_filename = os.path.join(output_folder, name)
    with open(saved_policy_filename, "w") as f:
        json.dump(params_rtneural, f)

    wandb.log_model(path=saved_policy_filename, name=name)
    wandb.log_model(path=model_path, name=f"mjx_policy_network_{wandb.run.name}.pt")
    wandb.finish()