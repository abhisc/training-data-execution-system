"""Curriculum stages, lane weights, protected floors, mixture schedule."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


TRAIN_LANES = ["web", "code", "math", "indic", "agentic", "reasoning"]


@dataclass
class StageConfig:
    name: str
    start_step: int
    end_step: int  # exclusive
    lane_weights: Dict[str, float]
    protected_floors: Dict[str, float] = field(default_factory=dict)
    annealing_reserve_lanes: List[str] = field(default_factory=list)
    packing_policy: str = "best_fit"


DEFAULT_STAGES: List[StageConfig] = [
    StageConfig(
        name="early-train",
        start_step=0,
        end_step=20,
        lane_weights={
            "web": 0.35,
            "code": 0.20,
            "math": 0.15,
            "indic": 0.10,
            "agentic": 0.10,
            "reasoning": 0.10,
        },
        protected_floors={"indic": 0.08, "agentic": 0.08},
        packing_policy="greedy",
    ),
    StageConfig(
        name="mid-train",
        start_step=20,
        end_step=40,
        lane_weights={
            "web": 0.25,
            "code": 0.20,
            "math": 0.20,
            "indic": 0.12,
            "agentic": 0.13,
            "reasoning": 0.10,
        },
        protected_floors={"indic": 0.10, "agentic": 0.10},
        packing_policy="best_fit",
    ),
    StageConfig(
        name="annealing",
        start_step=40,
        end_step=80,
        lane_weights={
            "web": 0.15,
            "code": 0.20,
            "math": 0.20,
            "indic": 0.15,
            "agentic": 0.15,
            "reasoning": 0.15,
        },
        protected_floors={"indic": 0.12, "agentic": 0.12},
        annealing_reserve_lanes=["math", "reasoning", "agentic"],
        packing_policy="structure_preserving",
    ),
]


@dataclass
class StepPlan:
    global_step: int
    stage: str
    lane: str
    lane_weights: Dict[str, float]
    protected_floors: Dict[str, float]
    packing_policy: str


class MixtureSchedule:
    """Compiled per-step lane assignment with protected floors."""

    def __init__(self, stages: Optional[Sequence[StageConfig]] = None, total_steps: int = 60):
        self.stages = list(stages or DEFAULT_STAGES)
        self.total_steps = total_steps
        self.steps: List[StepPlan] = []

    def stage_for_step(self, step: int) -> StageConfig:
        for stage in self.stages:
            if stage.start_step <= step < stage.end_step:
                return stage
        return self.stages[-1]

    def _pick_lane(self, step: int, stage: StageConfig) -> str:
        """Deterministic lane selection from cumulative weights + step index."""
        weights = stage.lane_weights
        lanes = [l for l in TRAIN_LANES if l in weights]
        # Ensure protected floors by reserving periodic slots
        floors = stage.protected_floors
        floor_lanes = [l for l in lanes if floors.get(l, 0) > 0]
        if floor_lanes:
            # Every N steps force a protected lane in round-robin
            period = max(2, int(1.0 / max(floors.values())))
            if step % period == 0:
                return floor_lanes[(step // period) % len(floor_lanes)]

        total = sum(weights[l] for l in lanes)
        # Deterministic fractional selection
        r = ((step * 2654435761) % 10000) / 10000.0
        acc = 0.0
        for lane in lanes:
            acc += weights[lane] / total
            if r <= acc:
                return lane
        return lanes[-1]

    def compile(self) -> List[StepPlan]:
        self.steps = []
        for step in range(self.total_steps):
            stage = self.stage_for_step(step)
            lane = self._pick_lane(step, stage)
            self.steps.append(
                StepPlan(
                    global_step=step,
                    stage=stage.name,
                    lane=lane,
                    lane_weights=dict(stage.lane_weights),
                    protected_floors=dict(stage.protected_floors),
                    packing_policy=stage.packing_policy,
                )
            )
        return self.steps

    def get(self, step: int) -> StepPlan:
        if not self.steps:
            self.compile()
        if step < 0 or step >= len(self.steps):
            # Extend deterministically beyond compiled range using last stage
            stage = self.stage_for_step(step)
            lane = self._pick_lane(step, stage)
            return StepPlan(
                global_step=step,
                stage=stage.name,
                lane=lane,
                lane_weights=dict(stage.lane_weights),
                protected_floors=dict(stage.protected_floors),
                packing_policy=stage.packing_policy,
            )
        return self.steps[step]

    def planned_shares(self, start: int = 0, end: Optional[int] = None) -> Dict[str, float]:
        if not self.steps:
            self.compile()
        end = end if end is not None else len(self.steps)
        counts = {l: 0 for l in TRAIN_LANES}
        n = 0
        for plan in self.steps[start:end]:
            counts[plan.lane] = counts.get(plan.lane, 0) + 1
            n += 1
        if n == 0:
            return {l: 0.0 for l in TRAIN_LANES}
        return {l: counts.get(l, 0) / n for l in TRAIN_LANES}

    def save(self, path: Path) -> None:
        if not self.steps:
            self.compile()
        payload = {
            "total_steps": self.total_steps,
            "stages": [asdict(s) for s in self.stages],
            "steps": [asdict(s) for s in self.steps],
            "planned_shares": self.planned_shares(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MixtureSchedule":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stages = [StageConfig(**s) for s in data["stages"]]
        sched = cls(stages=stages, total_steps=data["total_steps"])
        sched.steps = [StepPlan(**s) for s in data["steps"]]
        return sched
