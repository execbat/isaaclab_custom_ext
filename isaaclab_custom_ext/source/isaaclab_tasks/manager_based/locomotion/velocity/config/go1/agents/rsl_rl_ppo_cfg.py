# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg


@configclass
class UnitreeGo1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000000000
    save_interval = 50
    experiment_name = "unitree_go1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    
#    policy = RslRlPpoActorCriticRecurrentCfg(
#        init_noise_std=1.0,
#        actor_hidden_dims=[256],
#        critic_hidden_dims=[256],
#        activation="elu", #gelu
#
#        # — параметры памяти —
#        rnn_type="lstm",        # "lstm" | "gru"
#        rnn_hidden_dim=256,     # размер скрытого состояния
#        rnn_num_layers=1,       # слоев LSTM; начни с 1
#    )



#    algorithm = RslRlPpoAlgorithmCfg(
#        value_loss_coef=1.0,
#        use_clipped_value_loss=True,
#        clip_param=0.2,
#        entropy_coef=0.0001, # 0.008
#        num_learning_epochs=5,
#        num_mini_batches= 4,
#        learning_rate=1.0e-4,
#        schedule="adaptive",   # "adaptive", # "fixed", # fized because lr changes due to schedule from inside of env. not by KL_div "adaptive",
#        gamma=0.99,
#        lam=0.95,
#        desired_kl=0.005, # 0.01,
#        max_grad_norm=0.8, # 1.0
#    )

'''
@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 150000
        self.experiment_name = "g1_flat"
        #self.policy.actor_hidden_dims = [ 256, 128]
        #self.policy.critic_hidden_dims = [ 256, 128]
        
## added by johnny for AnimalMath        
@configclass
class MathG1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 150000
        self.experiment_name = "g1_flat"
        #self.policy.actor_hidden_dims = [ 256, 128]
        #self.policy.critic_hidden_dims = [ 256, 128]
        #self.policy = RslRlPpoActorCriticRecurrentCfg(
        #    init_noise_std=1.0,
        #    actor_hidden_dims=[256, 128],
        #    critic_hidden_dims=[256, 128],
        #    activation="gelu",
        #
        #    # — параметры памяти —
        #    rnn_type="lstm",        # "lstm" | "gru"
        #    rnn_hidden_dim=256,     # размер скрытого состояния
        #    rnn_num_layers=1,       # слоев LSTM; начни с 1
        #)
'''        
