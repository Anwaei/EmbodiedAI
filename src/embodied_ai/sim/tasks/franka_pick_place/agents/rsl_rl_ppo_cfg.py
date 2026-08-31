"""RSL-RL PPO runner configuration bound to the reviewed Stage 9 TOML."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from ..rl_env_cfg import STAGE9_STANDALONE_CONFIG


@configclass
class FrankaPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = STAGE9_STANDALONE_CONFIG.ppo.num_steps_per_env
    max_iterations = STAGE9_STANDALONE_CONFIG.ppo.max_iterations
    save_interval = STAGE9_STANDALONE_CONFIG.ppo.save_interval
    experiment_name = STAGE9_STANDALONE_CONFIG.identity.run_id
    run_name = ""
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=STAGE9_STANDALONE_CONFIG.ppo.init_noise_std,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=list(STAGE9_STANDALONE_CONFIG.ppo.actor_hidden_dims),
        critic_hidden_dims=list(STAGE9_STANDALONE_CONFIG.ppo.critic_hidden_dims),
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=STAGE9_STANDALONE_CONFIG.ppo.clip_param,
        entropy_coef=STAGE9_STANDALONE_CONFIG.ppo.entropy_coef,
        num_learning_epochs=STAGE9_STANDALONE_CONFIG.ppo.num_learning_epochs,
        num_mini_batches=STAGE9_STANDALONE_CONFIG.ppo.num_mini_batches,
        learning_rate=STAGE9_STANDALONE_CONFIG.ppo.learning_rate,
        schedule="adaptive",
        gamma=STAGE9_STANDALONE_CONFIG.ppo.gamma,
        lam=STAGE9_STANDALONE_CONFIG.ppo.lam,
        desired_kl=STAGE9_STANDALONE_CONFIG.ppo.desired_kl,
        max_grad_norm=STAGE9_STANDALONE_CONFIG.ppo.max_grad_norm,
    )


__all__ = ["FrankaPickPlacePPORunnerCfg"]
