"""Strict loader for the reviewed Stage 9 standalone PPO configuration."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from embodied_ai.contracts.rl import (
    RSL_RL_PPO_BACKEND,
    STAGE9_RUN_CONFIG_SCHEMA_VERSION,
    STANDALONE_PPO_ACTION_PROFILE,
    STANDALONE_PPO_OBSERVATION_PROFILE,
    STANDALONE_PPO_REWARD_PROFILE_ID,
    STANDALONE_PPO_TASK_ID,
    RlMode,
    Stage9RunIdentity,
)
from embodied_ai.contracts.tasks.franka_pick_place import (
    CUBE_CENTER_X_BOUNDS_ENV_M,
    CUBE_CENTER_Y_BOUNDS_ENV_M,
    CUBE_RESET_POSITION_ENV_M,
    GOAL_CENTER_X_BOUNDS_ENV_M,
    GOAL_CENTER_Y_BOUNDS_ENV_M,
    GOAL_POSITION_ENV_M,
    SUCCESS_POSITION_TOLERANCE_M,
)


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _range(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two values")
    result = (_finite(value[0], label), _finite(value[1], label))
    if result[0] >= result[1]:
        raise ValueError(f"{label} lower bound must be smaller than upper bound")
    return result


def _within(child: tuple[float, float], parent: tuple[float, float], label: str) -> None:
    if child[0] < parent[0] or child[1] > parent[1]:
        raise ValueError(f"{label} is outside the task contract bounds")


def _resolve_below(base: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a safe relative path")
    resolved_base = base.resolve()
    resolved = (resolved_base / candidate).resolve()
    if not resolved.is_relative_to(resolved_base):
        raise ValueError(f"{label} escapes its configured root")
    return resolved


@dataclass(frozen=True, slots=True)
class Stage9ResetDistribution:
    cube_x_m: tuple[float, float]
    cube_y_m: tuple[float, float]
    goal_x_m: tuple[float, float]
    goal_y_m: tuple[float, float]
    cube_z_m: float
    goal_z_m: float

    def validate(self) -> None:
        _within(self.cube_x_m, CUBE_CENTER_X_BOUNDS_ENV_M, "cube_x_m")
        _within(self.cube_y_m, CUBE_CENTER_Y_BOUNDS_ENV_M, "cube_y_m")
        _within(self.goal_x_m, GOAL_CENTER_X_BOUNDS_ENV_M, "goal_x_m")
        _within(self.goal_y_m, GOAL_CENTER_Y_BOUNDS_ENV_M, "goal_y_m")
        minimum_dx = max(0.0, self.goal_x_m[0] - self.cube_x_m[1])
        minimum_dy = max(
            0.0,
            self.cube_y_m[0] - self.goal_y_m[1],
            self.goal_y_m[0] - self.cube_y_m[1],
        )
        minimum_distance = math.hypot(minimum_dx, minimum_dy)
        if minimum_distance <= SUCCESS_POSITION_TOLERANCE_M:
            raise ValueError("training reset ranges may begin inside success tolerance")
        if not math.isclose(self.cube_z_m, CUBE_RESET_POSITION_ENV_M[2], abs_tol=1.0e-6):
            raise ValueError("cube_z_m must keep the cube on the reviewed table surface")
        if not math.isclose(self.goal_z_m, GOAL_POSITION_ENV_M[2], abs_tol=1.0e-6):
            raise ValueError("goal_z_m must retain cube-centre goal semantics")


@dataclass(frozen=True, slots=True)
class Stage9PhaseThresholds:
    reach_distance_m: float
    grasp_force_n: float
    lift_height_m: float
    place_distance_m: float


@dataclass(frozen=True, slots=True)
class Stage9RewardConfig:
    reach_weight: float
    grasp_weight: float
    lift_weight: float
    place_weight: float
    success_weight: float
    failure_weight: float
    action_magnitude_weight: float
    action_rate_weight: float
    gripper_toggle_weight: float
    reach_std_m: float
    lift_target_height_m: float
    place_std_m: float


@dataclass(frozen=True, slots=True)
class Stage9PpoConfig:
    num_steps_per_env: int
    max_iterations: int
    save_interval: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    init_noise_std: float
    learning_rate: float
    gamma: float
    lam: float
    clip_param: float
    entropy_coef: float
    num_learning_epochs: int
    num_mini_batches: int
    desired_kl: float
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class Stage9StandalonePpoConfig:
    """Reviewed standalone PPO settings shared by task and future trainer."""

    identity: Stage9RunIdentity
    num_envs: int
    max_episode_steps: int
    evaluation_scenarios_path: Path
    output_dir: Path
    reset: Stage9ResetDistribution
    phase: Stage9PhaseThresholds
    reward: Stage9RewardConfig
    ppo: Stage9PpoConfig

    @classmethod
    def from_toml(
        cls,
        path: Path,
        *,
        repository_root: Path,
        data_root: Path,
    ) -> Stage9StandalonePpoConfig:
        with path.open("rb") as stream:
            source = tomllib.load(stream)
        if source.get("schema_version") != STAGE9_RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 9 run config schema")

        profiles = source["profiles"]
        backend = source["backend"]
        training = source["training"]
        reset = source["reset_distribution"]
        phase = source["phase_thresholds"]
        reward = source["reward"]
        ppo = source["ppo"]
        artifacts = source["artifacts"]

        mode = RlMode(source["mode"])
        identity = Stage9RunIdentity(
            run_id=source["run_id"],
            mode=mode,
            task_id=source["task_id"],
            observation_profile_id=profiles["observation"],
            action_profile_id=profiles["action"],
            reward_profile_id=profiles["reward"],
            backend=RSL_RL_PPO_BACKEND,
            seed=int(training["seed"]),
        )
        if mode is not RlMode.STANDALONE_PPO:
            raise ValueError("this configuration loader accepts standalone PPO only")
        if identity.observation_profile_id != STANDALONE_PPO_OBSERVATION_PROFILE.profile_id:
            raise ValueError("standalone observation profile identity changed")
        if identity.action_profile_id != STANDALONE_PPO_ACTION_PROFILE.profile_id:
            raise ValueError("standalone action profile identity changed")
        if identity.reward_profile_id != STANDALONE_PPO_REWARD_PROFILE_ID:
            raise ValueError("standalone reward profile identity changed")
        if identity.task_id != STANDALONE_PPO_TASK_ID:
            raise ValueError("standalone PPO task identity changed")
        expected_backend = RSL_RL_PPO_BACKEND.to_dict()
        if {key: str(backend[key]) for key in expected_backend} != expected_backend:
            raise ValueError("Stage 9 backend identity does not match the Isaac lock")

        reset_distribution = Stage9ResetDistribution(
            cube_x_m=_range(reset["cube_x_m"], "cube_x_m"),
            cube_y_m=_range(reset["cube_y_m"], "cube_y_m"),
            goal_x_m=_range(reset["goal_x_m"], "goal_x_m"),
            goal_y_m=_range(reset["goal_y_m"], "goal_y_m"),
            cube_z_m=_finite(reset["cube_z_m"], "cube_z_m"),
            goal_z_m=_finite(reset["goal_z_m"], "goal_z_m"),
        )
        reset_distribution.validate()

        phase_thresholds = Stage9PhaseThresholds(
            reach_distance_m=_finite(phase["reach_distance_m"], "reach_distance_m", positive=True),
            grasp_force_n=_finite(phase["grasp_force_n"], "grasp_force_n", positive=True),
            lift_height_m=_finite(phase["lift_height_m"], "lift_height_m", positive=True),
            place_distance_m=_finite(phase["place_distance_m"], "place_distance_m", positive=True),
        )
        reward_config = Stage9RewardConfig(
            reach_weight=_finite(reward["reach_weight"], "reach_weight", positive=True),
            grasp_weight=_finite(reward["grasp_weight"], "grasp_weight", positive=True),
            lift_weight=_finite(reward["lift_weight"], "lift_weight", positive=True),
            place_weight=_finite(reward["place_weight"], "place_weight", positive=True),
            success_weight=_finite(reward["success_weight"], "success_weight", positive=True),
            failure_weight=_finite(reward["failure_weight"], "failure_weight", positive=True),
            action_magnitude_weight=_finite(
                reward["action_magnitude_weight"], "action_magnitude_weight", positive=True
            ),
            action_rate_weight=_finite(
                reward["action_rate_weight"], "action_rate_weight", positive=True
            ),
            gripper_toggle_weight=_finite(
                reward["gripper_toggle_weight"], "gripper_toggle_weight", positive=True
            ),
            reach_std_m=_finite(reward["reach_std_m"], "reach_std_m", positive=True),
            lift_target_height_m=_finite(
                reward["lift_target_height_m"], "lift_target_height_m", positive=True
            ),
            place_std_m=_finite(reward["place_std_m"], "place_std_m", positive=True),
        )
        if reward_config.lift_target_height_m <= phase_thresholds.lift_height_m:
            raise ValueError("lift reward target must exceed the lift phase threshold")

        ppo_config = Stage9PpoConfig(
            num_steps_per_env=_positive_int(ppo["num_steps_per_env"], "num_steps_per_env"),
            max_iterations=_positive_int(ppo["max_iterations"], "max_iterations"),
            save_interval=_positive_int(ppo["save_interval"], "save_interval"),
            actor_hidden_dims=tuple(
                _positive_int(value, "actor_hidden_dims") for value in ppo["actor_hidden_dims"]
            ),
            critic_hidden_dims=tuple(
                _positive_int(value, "critic_hidden_dims") for value in ppo["critic_hidden_dims"]
            ),
            init_noise_std=_finite(ppo["init_noise_std"], "init_noise_std", positive=True),
            learning_rate=_finite(ppo["learning_rate"], "learning_rate", positive=True),
            gamma=_finite(ppo["gamma"], "gamma", positive=True),
            lam=_finite(ppo["lam"], "lam", positive=True),
            clip_param=_finite(ppo["clip_param"], "clip_param", positive=True),
            entropy_coef=_finite(ppo["entropy_coef"], "entropy_coef"),
            num_learning_epochs=_positive_int(
                ppo["num_learning_epochs"], "num_learning_epochs"
            ),
            num_mini_batches=_positive_int(ppo["num_mini_batches"], "num_mini_batches"),
            desired_kl=_finite(ppo["desired_kl"], "desired_kl", positive=True),
            max_grad_norm=_finite(ppo["max_grad_norm"], "max_grad_norm", positive=True),
        )
        if not 0.0 < ppo_config.gamma <= 1.0 or not 0.0 < ppo_config.lam <= 1.0:
            raise ValueError("PPO gamma and lambda must be in (0, 1]")

        config = cls(
            identity=identity,
            num_envs=_positive_int(training["num_envs"], "num_envs"),
            max_episode_steps=_positive_int(
                training["max_episode_steps"], "max_episode_steps"
            ),
            evaluation_scenarios_path=_resolve_below(
                repository_root,
                training["evaluation_scenarios_path"],
                "evaluation_scenarios_path",
            ),
            output_dir=_resolve_below(data_root, artifacts["output_dir"], "output_dir"),
            reset=reset_distribution,
            phase=phase_thresholds,
            reward=reward_config,
            ppo=ppo_config,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_episode_steps != 200:
            raise ValueError("standalone PPO v1 uses the reviewed 200-step horizon")
        if not self.evaluation_scenarios_path.is_file():
            raise ValueError("frozen evaluation scenario manifest does not exist")
        if "/stage9/" not in self.output_dir.as_posix():
            raise ValueError("Stage 9 output directory must remain below runs/stage9")


def reviewed_stage9_config_path(repository_root: Path) -> Path:
    return repository_root / "configs" / "rl" / "franka_pick_place_standalone_ppo_v2.toml"


__all__ = [
    "Stage9PhaseThresholds",
    "Stage9PpoConfig",
    "Stage9ResetDistribution",
    "Stage9RewardConfig",
    "Stage9StandalonePpoConfig",
    "reviewed_stage9_config_path",
]
