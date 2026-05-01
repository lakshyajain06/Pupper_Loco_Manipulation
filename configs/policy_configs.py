from ml_collections import config_dict

def get_policy_configs():
    policy_config = config_dict.ConfigDict()

    policy_config.use_imu = True # Whether to use IMU in policy. Default: True

    policy_config.observation_history = 20  # number of stacked observations to give the policy

    policy_config.action_scale = 0.75  # Default 0.75

    policy_config.hidden_layer_sizes = (256, 128, 128, 128) # default (256, 128, 128, 128)

    # RTNeural supports relu, tanh, sigmoid (not great), softmax, elu, prelu
    # Swish was really good in terms of training but not supported in RTNeural rn
    policy_config.activation = "elu"

    return policy_config

