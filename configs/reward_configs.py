from ml_collections import config_dict


def get_reward_configs():
    reward_config = config_dict.ConfigDict()
    reward_config.rewards = config_dict.ConfigDict()
    reward_config.rewards.scales = config_dict.ConfigDict()

    # Track linear velocity
    # reward_config.rewards.scales.tracking_lin_vel = 20.0

    # Track the angular velocity along z-axis, i.e. yaw rate.
    # reward_config.rewards.scales.tracking_ang_vel = 10
    
    reward_config.rewards.tracking_foot_lin_pos = 10.0
    reward_config.rewards.scales.stand = 5.0

    # Track the given body orientation (desired world z axis in body frame)
    # Not working right nowkick
    reward_config.rewards.scales.tracking_orientation = 0

    # Below are regularization terms, we roughly divide the
    # terms to base state regularizations, joint
    # regularizations, and other behavior regularizations.
    # Penalize the base velocity in z direction, L2 penalty.
    reward_config.rewards.scales.lin_vel_z = -1.0

    # Penalize the base roll and pitch rate. L2 penalty.
    reward_config.rewards.scales.ang_vel_xy = -0.1

    # Penalize non-zero roll and pitch angles. L2 penalty.
    reward_config.rewards.scales.orientation = -0.1

    # L2 regularization of joint torques, sum(|tau|^2).
    reward_config.rewards.scales.torques = -0.0005

    # L2 regularization of joint accelerations sum(|qdd|^2)
    reward_config.rewards.scales.joint_acceleration = -0.001

    # L1 regularization of mechanical work, |v * tau|.
    reward_config.rewards.scales.mechanical_work = -0.01

    # Penalize the change in the action and encourage smooth
    # actions. L1 regularization |action - last_action|^2
    reward_config.rewards.scales.action_rate = -0.01

    # Encourage long swing steps. However, it does not
    # encourage high clearances.
    # reward_config.rewards.scales.feet_air_time = 1.5

    # Encourage joints at default position at zero command, L1 regularization
    # |q - q_default|.
    # reward_config.rewards.scales.stand_still = 1

    # Encourage zero joint velocity at zero command, L1 regularization
    # |q_dot|.
    # Activates when norm(command) < stand_still_command_threshold
    # Commands below this threshold are sampled with probability zero_command_probability
    # reward_config.rewards.scales.stand_still_joint_velocity = 1

    # Encourage zero abduction angle so legs don't spread so far out
    # L2 loss on ||abduction_motors - desired||^2
    reward_config.rewards.scales.abduction_angle = 0.1

    # Early termination penalty.
    reward_config.rewards.scales.termination = 0

    # Penalizing foot slipping on the ground.
    reward_config.rewards.scales.foot_slip = 0

    # Penalize knees hitting the ground
    reward_config.rewards.scales.knee_collision = -1.5

    # Penalize body hitting ground
    reward_config.rewards.scales.body_collision = -3.0

    # Tracking reward = exp(-error^2/sigma).
    reward_config.rewards.tracking_sigma = 0.25

    return reward_config