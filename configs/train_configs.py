from ml_collections import config_dict
from pupperv3_mjx import domain_randomization

import importlib

def get_train_configs():
    importlib.reload(domain_randomization)

    training_config = config_dict.ConfigDict()
    training_config.checkpoint_run_number = None

    # Environment timestep
    training_config.environment_dt = 0.02

    # PPO params
    training_config.ppo = config_dict.ConfigDict()
    training_config.ppo.num_timesteps = 200_000_000   # Default: 300M
    training_config.ppo.episode_length = 500         # Default: 1000
    training_config.ppo.num_evals = 11                # Default: 10
    training_config.ppo.reward_scaling = 1            # Default: 1
    training_config.ppo.normalize_observations = True # Default: True
    training_config.ppo.action_repeat = 1             # Default: 1
    training_config.ppo.unroll_length = 20            # Default: 20
    training_config.ppo.num_minibatches = 32          # Default: 32
    training_config.ppo.num_updates_per_batch = 4     # Default: 4
    training_config.ppo.discounting = 0.97            # Default: 0.97
    training_config.ppo.learning_rate = 3.0e-5        # Default: 3.0e-4, 3.0e-5 was better than 3e-4
    training_config.ppo.entropy_cost = 1e-2           # Default: 1e-2
    training_config.ppo.num_envs = 8192               # Default: 8192
    training_config.ppo.batch_size = 256              # Default: 256

    # Command sampling
    training_config.resample_velocity_step = training_config.ppo.episode_length // 2
    training_config.lin_vel_x_range = [-0.75, 0.75]  # min max [m/s]. Default: [-0.75, 0.75]
    training_config.lin_vel_y_range = [-0.5, 0.5]  # min max [m/s]. Default: [-0.5, 0.5]
    training_config.ang_vel_yaw_range = [-2.0, 2.0]  # min max [rad/s]. Default: [-2.0, 2.0]
    training_config.zero_command_probability = 0.02
    training_config.stand_still_command_threshold = 0.05

    # Orientation command sampling in degrees
    training_config.maximum_pitch_command = 0.0
    training_config.maximum_roll_command = 0.0

    # Desired body orientation
    training_config.desired_world_z_in_body_frame = (0.0, 0.0, 1.0) # Default: (0.0, 0.0, 1.0)

    # Termination
    # NOTE: without a body collision geometry, can't train recovery policy
    training_config.terminal_body_z = 0.1  # Episode ends if body center goes below this height [m] Default: 0.10 m
    training_config.terminal_body_angle = 0.52  # Episode ends if body angle relative to vertical is more than this. Default: 0.52 rad (30 deg)
    training_config.early_termination_step_threshold = training_config.ppo.episode_length // 2 # Default: 500

    # Joint PD overrides
    training_config.dof_damping = 0.25  # Joint damping [Nm / (rad/s)] Default: 0.25
    training_config.position_control_kp = 5.5  # Joint stiffness [Nm / rad] Default: 5.0

    # Default joint angles
    training_config.default_pose = jp.array(
        [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52]
    )

    # Desired abduction angles
    training_config.desired_abduction_angles = jp.array(
        [0.0, 0.0, 0.0, 0.0]
    )

    # Height field
    ## Type of height field
    training_config.height_field_random = False
    training_config.height_field_steps = False
    ### Steps type params
    training_config.height_field_step_size = 4
    ## General height field settings
    training_config.height_field_grid_size = 256
    training_config.height_field_group = "0"
    training_config.height_field_radius_x = 10.0 # [m]
    training_config.height_field_radius_y = 10.0 # [m]
    training_config.height_field_elevation_z = 0.02 # [m]
    training_config.height_field_base_z = 0.2 # [m]

    # Domain randomization
    ## Perturbations
    training_config.kick_probability = 0.0        # Kick the robot with this probability. ( ͡° ͜ʖ ͡°) Default: 0.04
    training_config.kick_vel = 0.10               # Change the torso velocity by up to this much in x and y direction to simulate a kick [m/s] Default: 0.1
    training_config.angular_velocity_noise = 0.1  # Default: 0.1 [rad/s]
    training_config.gravity_noise = 0.05            # Default: 0.05 [u]
    training_config.motor_angle_noise = 0.05        # Default: 0.05 [rad]
    training_config.last_action_noise = 0.01       # Default: 0.01 [rad]

    ## Motors
    training_config.position_control_kp_multiplier_range = (0.6, 1.1)
    training_config.position_control_kd_multiplier_range = (0.8, 1.5)

    ## Starting position
    training_config.start_position_config = domain_randomization.StartPositionRandomization(
        x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, z_min=0.15, z_max=0.20
    )

    ## Latency distribution
    # Action latency
    # 0 latency with 20% prob, 1 timestep latency with 80% prob
    training_config.latency_distribution = jp.array([0.2, 0.8])

    # IMU latency
    # 0 latency with 50% prob, 1 timestep latency with 50% prob
    training_config.imu_latency_distribution = jp.array([0.5, 0.5])

    ## Body CoM
    training_config.body_com_x_shift_range = (-0.02, 0.02) # Default: -0.02, 0.02
    training_config.body_com_y_shift_range = (-0.005, 0.005)
    training_config.body_com_z_shift_range = (-0.005, 0.005)

    ## Mass and inertia randomization for all bodies
    training_config.body_mass_scale_range = (0.9, 1.3)
    training_config.body_inertia_scale_range = (0.9, 1.3)

    ## Friction
    training_config.friction_range = (0.6, 1.4)

    # Obstacles
    training_config.n_obstacles = 0
    training_config.obstacle_x_range = (-3.0, 3.0)  # [m]
    training_config.obstacle_y_range = (-3.0, 3.0)  # [m]
    training_config.obstacle_height = 0.04  # [m]
    training_config.obstacle_length = 2.0  # [m]

    return training_config