"""Strict loader for the reviewed Stage 8 evaluation configuration."""

from __future__ import annotations

import ipaddress
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from embodied_ai.contracts.evaluation import (
    STAGE8_SCENARIOS_SCHEMA_VERSION,
    EvaluationScenario,
)


@dataclass(frozen=True, slots=True)
class Stage8RunConfig:
    run_id: str
    host: str
    port: int
    timeout_s: float
    max_payload_bytes: int
    control_hz: int
    prediction_horizon: int
    execute_horizon: int
    max_steps: int
    hard_action_limit: float
    scenarios_path: Path
    noise_seeds: tuple[int, ...]
    expected_scenario_count: int
    record_video: bool
    stage7_config_path: Path
    processor_dir: Path
    adapter_dir: Path
    expert_config_path: Path

    @classmethod
    def from_toml(
        cls,
        path: Path,
        *,
        repository_root: Path,
        data_root: Path,
    ) -> Stage8RunConfig:
        with path.open("rb") as stream:
            source = tomllib.load(stream)
        if source.get("schema_version") != "embodied-ai.stage8-run-config/v1":
            raise ValueError("unsupported Stage 8 run config schema")
        server = source["server"]
        scheduler = source["scheduler"]
        safety = source["safety"]
        evaluation = source["evaluation"]
        artifacts = source["artifacts"]

        host = str(server["host"])
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("Stage 8 transport must bind to loopback")
        config = cls(
            run_id=str(source["run_id"]),
            host=host,
            port=int(server["port"]),
            timeout_s=float(server["request_timeout_s"]),
            max_payload_bytes=int(server["max_payload_bytes"]),
            control_hz=int(scheduler["control_hz"]),
            prediction_horizon=int(scheduler["prediction_horizon"]),
            execute_horizon=int(scheduler["execute_horizon"]),
            max_steps=int(scheduler["max_episode_steps"]),
            hard_action_limit=float(safety["hard_action_limit"]),
            scenarios_path=(repository_root / evaluation["scenarios_path"]).resolve(),
            noise_seeds=tuple(int(seed) for seed in evaluation["policy_noise_seeds"]),
            expected_scenario_count=int(evaluation.get("expected_scenario_count", 5)),
            record_video=bool(evaluation["record_video"]),
            stage7_config_path=(repository_root / artifacts["stage7_run_config"]).resolve(),
            processor_dir=(data_root / artifacts["processor_dir"]).resolve(),
            adapter_dir=(data_root / artifacts["adapter_dir"]).resolve(),
            expert_config_path=(repository_root / artifacts["expert_config"]).resolve(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.run_id or not 1 <= self.port <= 65535:
            raise ValueError("invalid run_id or server port")
        if self.timeout_s <= 0 or self.max_payload_bytes < 200_000:
            raise ValueError("invalid transport timeout or payload limit")
        if (self.control_hz, self.prediction_horizon, self.execute_horizon) != (20, 50, 5):
            raise ValueError("reviewed Stage 8 scheduler is 20 Hz, horizon 50, execute 5")
        if self.max_steps != 200 or self.hard_action_limit != 1.5:
            raise ValueError("reviewed Stage 8 safety envelope changed")
        if len(self.noise_seeds) != 3 or len(set(self.noise_seeds)) != 3:
            raise ValueError("Stage 8 requires exactly three distinct noise seeds")
        if self.expected_scenario_count not in (5, 10):
            raise ValueError("reviewed Stage 8 suites contain either five or ten scenarios")


def load_scenarios(
    path: Path,
    *,
    expected_count: int = 5,
) -> tuple[EvaluationScenario, ...]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("schema_version") != STAGE8_SCENARIOS_SCHEMA_VERSION:
        raise ValueError("unsupported Stage 8 scenarios schema")
    scenarios = tuple(EvaluationScenario.from_dict(item) for item in source["scenarios"])
    if (
        len(scenarios) != expected_count
        or len({item.scenario_id for item in scenarios}) != expected_count
    ):
        raise ValueError(
            f"reviewed Stage 8 suite requires {expected_count} unique scenarios"
        )
    return scenarios


__all__ = ["Stage8RunConfig", "load_scenarios"]
