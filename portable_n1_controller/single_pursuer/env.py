"""Pursuit capture environment (gymnasium.Env), parameterized by minion count.

``num_pursuers`` (N) pursuer cars chase one evader inside a bordered arena under a
single *central* controller (one policy outputs all 2*N actions). Uses the validated
bicycle dynamics and the vehicle/arena parameters from
``configs/hive_foundation_experiment.json`` (the project's single source of truth).
Designed for single-agent PPO and cheap vectorized local training; the same
N-parameterized observation/action/reward contract is intended to carry into Isaac.

Observation (clamped to [-3, 3]) is the central controller's full-state view:
  - global evader block (6): evader pos + velocity + predicted-lead pos (bird's-eye)
  - per-pursuer block (14) x N: pose, speed, range/bearing to evader, wall clearances,
    and the nearest-teammate relative offset (zeros when N == 1)
Action is [throttle, steering] per pursuer in [-1, 1], length 2*N.
Capture is configurable: 'all' (every pursuer within the capture radius simultaneously,
the hive-surround intent) or 'any' (any single pursuer reaches), the easier curriculum.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from vehicle_dynamics import next_longitudinal_speed

from .reward import REWARD_WEIGHTS, build_reward_contributions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "hive_foundation_experiment.json"

GLOBAL_OBS = 6              # evader pos (2) + velocity (2) + predicted-lead pos (2) [overhead bird's-eye]
OVERHEAD_PER_PURSUER = 14   # pose(4) + speed(1) + range/bearing(3) + wall clearances(4) + nearest-teammate(2) [overhead]
POV_PER_PURSUER = 4         # ADDITIVE per-car CAMERA detection of the evader: visible + range + bearing(sin,cos), FOV-gated
OBS_CLAMP = 3.0
DT = 0.1
LEAD_TIME_S = 0.6
TTC_THRESHOLD_S = 1.2  # near-miss horizon for the opt-in proactive `ttc` reward penalty
EVADER_FLEE_SPEED_FRAC = 0.6  # fleeing cruise speed as a fraction of the evader's max

EVADER_STATIC = "static_evader"

CAPTURE_ALL = "all"          # every pursuer within the capture radius (can be satisfied by piling)
CAPTURE_ANY = "any"          # any single pursuer within the radius (easy curriculum)
CAPTURE_SURROUND = "surround"  # all within radius AND the evader is enclosed (the real win condition)
CAPTURE_MODES = (CAPTURE_ALL, CAPTURE_ANY, CAPTURE_SURROUND)


def observation_dim(num_pursuers: int, use_car_cameras: bool = True) -> int:
    per = OVERHEAD_PER_PURSUER + (POV_PER_PURSUER if use_car_cameras else 0)
    return GLOBAL_OBS + per * int(num_pursuers)


def action_dim(num_pursuers: int) -> int:
    return 2 * int(num_pursuers)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SinglePursuerEnv(gym.Env):
    """Central-controller pursuit env for N pursuers (N=1 is the original case)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any] | str | Path | None = None,
        *,
        stage: str = EVADER_STATIC,
        num_pursuers: int = 1,
        capture_mode: str = CAPTURE_SURROUND,
        use_car_cameras: bool | None = None,
        enclosure_max_gap_deg: float = 180.0,
        evader_speed_frac: float = EVADER_FLEE_SPEED_FRAC,
        reward_weights: dict[str, float] | None = None,
        capture_hold_steps: int = 5,
        immobile_speed_mps: float = 0.5,
        require_immobility: bool = True,
        max_steps: int = 200,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if config is None or isinstance(config, (str, Path)):
            config_path = Path(config) if config is not None else DEFAULT_CONFIG
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if int(num_pursuers) < 1:
            raise ValueError("num_pursuers must be >= 1")
        if capture_mode not in CAPTURE_MODES:
            raise ValueError(f"capture_mode must be one of {CAPTURE_MODES}")
        self.config = config
        self.stage = stage
        self.num_pursuers = int(num_pursuers)
        self.capture_mode = capture_mode
        # Surround win condition: the evader is "enclosed" when no angular gap between
        # consecutive pursuers (as seen from the evader) exceeds this threshold. <= 180
        # means the evader sits inside/on the pursuers' hull -> no open escape lane.
        self.enclosure_max_gap = math.radians(float(enclosure_max_gap_deg))
        # Corral-and-pin win: the evader must be enclosed AND held for capture_hold_steps
        # consecutive steps. Two toggleable variants (only meaningful for 'surround'):
        #   require_immobility=True  ("stop motion", the default) -- the evader must ALSO
        #     stay under immobile_disp displacement per step: a true stationary pin.
        #   require_immobility=False ("stay in radius") -- the ring just has to persist;
        #     the evader may keep moving/wriggling inside it.
        self.capture_hold_steps = int(capture_hold_steps)
        self.immobile_disp = float(immobile_speed_mps) * DT  # per-step displacement threshold (m)
        self.require_immobility = bool(require_immobility)
        self.max_steps = int(max_steps)

        scene = config["scene"]
        attacker = config["vehicles"]["attackers"]
        evader = config["vehicles"]["evader"]
        metrics = config["success_metrics"]
        sensing = config.get("sensing", {})
        pov = sensing.get("pov_scalar_detection", {})
        prediction = sensing.get("prediction", {})

        # --- Per-car CAMERA POV lever -------------------------------------------------
        # The team keeps per-car forward cameras as extra detection data. This toggles
        # whether the controller receives that per-car POV block. It defaults to the
        # Step 1 sensing contract (`sensing.pov_scalar_detection.enabled`) and can be
        # overridden per run, so the team can disable it for ablations any time.
        self.use_car_cameras = bool(pov.get("enabled", True)) if use_car_cameras is None else bool(use_car_cameras)
        self.pov_fov = math.radians(float(pov.get("fov_deg", 90.0)))
        self.pov_range = float(pov.get("range_m", 8.0))
        self.pov_dropout = float(pov.get("dropout_probability", 0.0))
        self.pov_noise = float(pov.get("noise_std", 0.0))
        self.lead_time = float(prediction.get("evader_lead_time_s", LEAD_TIME_S))

        # Scale-aware constants (new keys; defaults preserve the validated
        # big-arena behavior exactly -- see configs/test_arena_6ft.json for a world
        # where the old hardcoded values would exceed the whole arena).
        self.spawn_min_from_evader_floor = float(scene.get("spawn_min_from_evader_floor_m", 3.0))
        self.evader_wall_margin = float(scene.get("evader_wall_margin_m", 2.5))
        self.spawn_wall_inset = float(scene.get("spawn_wall_inset_m", 0.5))
        self.half_width = float(scene["arena_width_m"]) * 0.5
        self.half_height = float(scene["arena_height_m"]) * 0.5
        self.diagonal = math.hypot(2.0 * self.half_width, 2.0 * self.half_height)
        self.capture_radius = float(metrics["capture_radius_m"])

        self.body_length = float(attacker["body_length_m"])
        self.wheelbase = float(attacker["wheelbase_m"])
        self.max_speed = float(attacker["max_speed_mps"])
        self.max_reverse_speed = float(attacker["max_reverse_speed_mps"])
        self.max_accel = float(attacker["max_accel_mps2"])
        self.max_reverse_accel = float(attacker["max_reverse_accel_mps2"])
        self.max_brake = float(attacker["max_brake_mps2"])
        self.max_steer = math.radians(float(attacker["max_steering_angle_deg"]))
        self.vehicle_collision_distance = self.body_length * 0.76

        self.evader_max_speed = float(evader["max_speed_mps"])
        self.evader_max_accel = float(evader["max_accel_mps2"])
        self.evader_max_reverse_speed = float(evader.get("max_reverse_speed_mps", self.evader_max_speed * 0.38))
        self.evader_max_reverse_accel = float(evader.get("max_reverse_accel_mps2", self.evader_max_accel))
        self.evader_max_brake = float(evader.get("max_brake_mps2", self.evader_max_accel * 1.8))
        self.evader_wheelbase = float(evader.get("wheelbase_m", self.wheelbase))
        self.evader_max_steer = math.radians(float(evader.get("max_steering_angle_deg", 28.0)))
        # Evader difficulty dial (curriculum): cruise speed as a fraction of its max.
        # ~0.3 = easy warm-up; 1.0 = equal speed -> uncatchable by chase, must be trapped.
        self.evader_speed_frac = float(evader_speed_frac)
        # Per-stage reward weights (curriculum sets these); merge over the defaults.
        self.reward_weights = dict(REWARD_WEIGHTS)
        if reward_weights:
            self.reward_weights.update(reward_weights)
        self.margin = self.body_length * 0.5

        self.observation_space = gym.spaces.Box(
            low=-OBS_CLAMP,
            high=OBS_CLAMP,
            shape=(observation_dim(self.num_pursuers, self.use_car_cameras),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim(self.num_pursuers),), dtype=np.float32
        )

        self.dt = DT
        self.step_count = 0
        self.pursuers: list[dict[str, float]] = []
        self.evader: dict[str, float] = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0}
        self.previous_distances: list[float] = []
        self.previous_within = 0
        self.previous_gap = 0.0
        self.previous_slot_distance = 0.0
        self._pin_steps = 0
        self.previous_pin_steps = 0
        self._evader_prev = (0.0, 0.0)
        self._rng = np.random.default_rng(seed)
        self.last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    def _next_speed(self, speed: float, throttle: float) -> float:
        # Validated bicycle longitudinal model, shared with every backend via the
        # neutral vehicle_dynamics module (see tests/test_shared_longitudinal_model.py,
        # which pins all three backends to this exact delegation).
        return next_longitudinal_speed(
            speed,
            throttle,
            max_speed=self.max_speed,
            max_reverse_speed=self.max_reverse_speed,
            max_accel=self.max_accel,
            max_reverse_accel=self.max_reverse_accel,
            max_brake=self.max_brake,
            dt=self.dt,
        )

    def _drive(self, vehicle: dict[str, float], throttle: float, steer_norm: float) -> None:
        throttle = _clamp(throttle, -1.0, 1.0)
        steer = _clamp(steer_norm, -1.0, 1.0) * self.max_steer
        vehicle["speed"] = self._next_speed(vehicle["speed"], throttle)
        vehicle["heading"] = _wrap_angle(
            vehicle["heading"] + (vehicle["speed"] / max(self.wheelbase, 0.01)) * math.tan(steer) * self.dt
        )
        vehicle["x"] += math.cos(vehicle["heading"]) * vehicle["speed"] * self.dt
        vehicle["y"] += math.sin(vehicle["heading"]) * vehicle["speed"] * self.dt

    def _keep_inside(self, vehicle: dict[str, float]) -> int:
        contact = 0
        limit_x = self.half_width - self.margin
        limit_y = self.half_height - self.margin
        if vehicle["x"] < -limit_x:
            vehicle["x"], vehicle["speed"], contact = -limit_x, 0.0, 1
        elif vehicle["x"] > limit_x:
            vehicle["x"], vehicle["speed"], contact = limit_x, 0.0, 1
        if vehicle["y"] < -limit_y:
            vehicle["y"], vehicle["speed"], contact = -limit_y, 0.0, 1
        elif vehicle["y"] > limit_y:
            vehicle["y"], vehicle["speed"], contact = limit_y, 0.0, 1
        return contact

    def _flee_heading(self) -> float:
        # Potential-field evasion: repelled by EVERY pursuer (stronger when closer)
        # and by nearby walls, so the evader steers toward open space and actively
        # avoids being cornered/surrounded. This forces the pursuers to coordinate
        # (close the open lanes) rather than win by a simple tail-chase.
        e = self.evader
        fx = fy = 0.0
        for p in self.pursuers:
            dx, dy = e["x"] - p["x"], e["y"] - p["y"]
            d = max(math.hypot(dx, dy), 1e-3)
            weight = 1.0 / (d * d)
            fx += (dx / d) * weight
            fy += (dy / d) * weight
        wall_margin = self.evader_wall_margin
        left, right = e["x"] + self.half_width, self.half_width - e["x"]
        bottom, top = e["y"] + self.half_height, self.half_height - e["y"]
        if left < wall_margin:
            fx += (wall_margin - left) / wall_margin * 1.5
        if right < wall_margin:
            fx -= (wall_margin - right) / wall_margin * 1.5
        if bottom < wall_margin:
            fy += (wall_margin - bottom) / wall_margin * 1.5
        if top < wall_margin:
            fy -= (wall_margin - top) / wall_margin * 1.5
        if abs(fx) < 1e-9 and abs(fy) < 1e-9:
            return e["heading"]
        return math.atan2(fy, fx)

    def _move_evader(self) -> None:
        if self.stage == EVADER_STATIC:
            return
        flee_heading = self._flee_heading()
        heading_error = _wrap_angle(flee_heading - self.evader["heading"])
        steer = _clamp(heading_error, -self.evader_max_steer, self.evader_max_steer)
        target_speed = self.evader_max_speed * self.evader_speed_frac
        self.evader["speed"] = next_longitudinal_speed(
            self.evader["speed"],
            1.0,
            max_speed=target_speed,
            max_reverse_speed=self.evader_max_reverse_speed,
            max_accel=self.evader_max_accel,
            max_reverse_accel=self.evader_max_reverse_accel,
            max_brake=self.evader_max_brake,
            dt=self.dt,
        )
        self.evader["heading"] = _wrap_angle(
            self.evader["heading"]
            + (self.evader["speed"] / max(self.evader_wheelbase, 0.01)) * math.tan(steer) * self.dt
        )
        self.evader["x"] += math.cos(self.evader["heading"]) * self.evader["speed"] * self.dt
        self.evader["y"] += math.sin(self.evader["heading"]) * self.evader["speed"] * self.dt

    def _evader_distances(self) -> list[float]:
        return [math.hypot(self.evader["x"] - p["x"], self.evader["y"] - p["y"]) for p in self.pursuers]

    def _pursuer_collisions(self) -> int:
        collisions = 0
        for i in range(self.num_pursuers):
            for j in range(i + 1, self.num_pursuers):
                a, b = self.pursuers[i], self.pursuers[j]
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < self.vehicle_collision_distance:
                    collisions += 1
        return collisions

    def _ttc_risk(self) -> float:
        # Proactive near-miss severity (archived hive_chase pair-safety pattern): for
        # each pursuer pair on a CLOSING course, time-to-collision = clearance /
        # closing speed, and severity ramps 0..1 as TTC falls below TTC_THRESHOLD_S.
        # Zero for separating, already-overlapping, or comfortably distant pairs.
        # Unlike the after-the-fact collision penalty, penalizing this teaches
        # braking/steering BEFORE contact. Feeds the opt-in `ttc` reward weight.
        if self.num_pursuers == 1:
            return 0.0
        risk = 0.0
        for i in range(self.num_pursuers):
            for j in range(i + 1, self.num_pursuers):
                a, b = self.pursuers[i], self.pursuers[j]
                rpx, rpy = b["x"] - a["x"], b["y"] - a["y"]
                rvx = math.cos(b["heading"]) * b["speed"] - math.cos(a["heading"]) * a["speed"]
                rvy = math.sin(b["heading"]) * b["speed"] - math.sin(a["heading"]) * a["speed"]
                separation = max(math.hypot(rpx, rpy), 1e-5)
                closing_speed = -(rpx * rvx + rpy * rvy) / separation
                if closing_speed > 1e-4 and separation > self.vehicle_collision_distance:
                    ttc = (separation - self.vehicle_collision_distance) / closing_speed
                    risk += max(0.0, min(1.0, 1.0 - ttc / TTC_THRESHOLD_S))
        return risk

    def _teammate_spacing(self) -> float:
        # Soft crowding penalty that grows as teammates get closer than a safe ring
        # spacing (well before an actual collision), so the fleet spreads into a
        # surround instead of piling onto the evader. Zero for N == 1.
        if self.num_pursuers == 1:
            return 0.0
        # Target only genuine crowding just beyond the collision distance, so the
        # penalty discourages pile-ups/collisions without preventing the surround
        # ring (3 cars at ~0.7 m radius sit ~1.2 m apart, well clear of this band).
        safe = self.vehicle_collision_distance * 1.4
        penalty = 0.0
        for i in range(self.num_pursuers):
            for j in range(i + 1, self.num_pursuers):
                a, b = self.pursuers[i], self.pursuers[j]
                sep = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                if sep < safe:
                    penalty += ((safe - sep) / safe) ** 2
        return penalty

    def _evader_velocity(self) -> tuple[float, float]:
        return (
            math.cos(self.evader["heading"]) * self.evader["speed"],
            math.sin(self.evader["heading"]) * self.evader["speed"],
        )

    def _nearest_teammate_offset(self, index: int) -> tuple[float, float]:
        if self.num_pursuers == 1:
            return 0.0, 0.0
        p = self.pursuers[index]
        best = None
        best_d = float("inf")
        for j, other in enumerate(self.pursuers):
            if j == index:
                continue
            d = math.hypot(other["x"] - p["x"], other["y"] - p["y"])
            if d < best_d:
                best_d, best = d, other
        return (best["x"] - p["x"], best["y"] - p["y"])

    def _pov_detection(self, p: dict[str, float]) -> list[float]:
        # Per-car forward-CAMERA detection of the evader: [visible, range, sin/cos bearing],
        # gated by the camera FOV + range from the Step 1 sensing contract, with optional
        # dropout/noise. Zeroed when the evader is out of view. Only included when the
        # car-camera lever is on; the overhead bird's-eye block still localizes everything.
        dx, dy = self.evader["x"] - p["x"], self.evader["y"] - p["y"]
        distance = math.hypot(dx, dy)
        bearing = _wrap_angle(math.atan2(dy, dx) - p["heading"])
        visible = distance <= self.pov_range and abs(bearing) <= self.pov_fov * 0.5
        if visible and self.pov_dropout > 0.0 and float(self._rng.random()) < self.pov_dropout:
            visible = False
        if not visible:
            return [0.0, 0.0, 0.0, 0.0]
        if self.pov_noise > 0.0:
            distance = max(0.0, distance + float(self._rng.normal(0.0, self.pov_noise)))
            bearing = _wrap_angle(bearing + float(self._rng.normal(0.0, self.pov_noise)))
        return [1.0, distance / self.diagonal, math.sin(bearing), math.cos(bearing)]

    def _observation(self) -> np.ndarray:
        e = self.evader
        evx, evy = self._evader_velocity()
        pred_x = e["x"] + evx * self.lead_time
        pred_y = e["y"] + evy * self.lead_time
        raw: list[float] = [
            e["x"] / self.half_width,
            e["y"] / self.half_height,
            evx / self.max_speed,
            evy / self.max_speed,
            pred_x / self.half_width,
            pred_y / self.half_height,
        ]
        for index, p in enumerate(self.pursuers):
            dx, dy = e["x"] - p["x"], e["y"] - p["y"]
            distance = math.hypot(dx, dy)
            bearing = _wrap_angle(math.atan2(dy, dx) - p["heading"])
            tdx, tdy = self._nearest_teammate_offset(index)
            raw.extend([
                p["x"] / self.half_width,
                p["y"] / self.half_height,
                math.sin(p["heading"]),
                math.cos(p["heading"]),
                p["speed"] / self.max_speed,
                distance / self.diagonal,   # overhead range to evader (always available)
                math.sin(bearing),          # overhead bearing to evader
                math.cos(bearing),
                (p["x"] + self.half_width) / (2.0 * self.half_width),
                (self.half_width - p["x"]) / (2.0 * self.half_width),
                (p["y"] + self.half_height) / (2.0 * self.half_height),
                (self.half_height - p["y"]) / (2.0 * self.half_height),
                tdx / (2.0 * self.half_width),
                tdy / (2.0 * self.half_height),
            ])
            if self.use_car_cameras:
                raw.extend(self._pov_detection(p))  # additive FOV-gated camera detection
        return np.asarray([_clamp(v, -OBS_CLAMP, OBS_CLAMP) for v in raw], dtype=np.float32)

    def _spawn(self) -> None:
        inner_x = self.half_width - self.margin - self.spawn_wall_inset
        inner_y = self.half_height - self.margin - self.spawn_wall_inset
        self.evader = {
            "x": float(self._rng.uniform(-inner_x * 0.5, inner_x * 0.5)),
            "y": float(self._rng.uniform(-inner_y * 0.5, inner_y * 0.5)),
            "heading": float(self._rng.uniform(-math.pi, math.pi)),
            "speed": 0.0,
        }
        min_from_evader = max(self.spawn_min_from_evader_floor, self.capture_radius * 3.0)
        min_from_peers = self.vehicle_collision_distance * 1.5
        self.pursuers = []
        for _ in range(self.num_pursuers):
            for _attempt in range(128):
                px = float(self._rng.uniform(-inner_x, inner_x))
                py = float(self._rng.uniform(-inner_y, inner_y))
                if math.hypot(px - self.evader["x"], py - self.evader["y"]) < min_from_evader:
                    continue
                if any(math.hypot(px - q["x"], py - q["y"]) < min_from_peers for q in self.pursuers):
                    continue
                break
            self.pursuers.append(
                {"x": px, "y": py, "heading": float(self._rng.uniform(-math.pi, math.pi)), "speed": 0.0}
            )

    def _max_angular_gap(self) -> float:
        # Largest angular gap between consecutive pursuers as seen FROM the evader.
        # max_gap <= 180 deg <=> the evader is inside/on the pursuers' convex hull
        # (no open escape lane). Meaningless with < 3 pursuers, so reported as 0.
        if self.num_pursuers < 3:
            return 0.0
        angles = sorted(
            math.atan2(p["y"] - self.evader["y"], p["x"] - self.evader["x"]) for p in self.pursuers
        )
        return max(
            (angles[(i + 1) % len(angles)] - angles[i]) % (2.0 * math.pi) for i in range(len(angles))
        )

    def _is_enclosed(self) -> bool:
        return self._max_angular_gap() <= self.enclosure_max_gap

    def _role_slots(self) -> list[tuple[float, float]]:
        # Fixed points evenly spaced around the predicted evader position -- explicit
        # "go here" surround targets (from the archived Isaac hive_chase role_slot()
        # idea), as an alternative to relying on emergent angular spread. Pursuers are
        # matched to slots via the distance-minimizing permutation (cheap for small N)
        # so a fixed index->slot mapping can't force an inefficient detour.
        e = self.evader
        evx, evy = self._evader_velocity()
        cx = e["x"] + evx * self.lead_time
        cy = e["y"] + evy * self.lead_time
        radius = self.capture_radius * 0.75
        n = self.num_pursuers
        slots = [
            (cx + radius * math.cos(2.0 * math.pi * i / n), cy + radius * math.sin(2.0 * math.pi * i / n))
            for i in range(n)
        ]
        if n == 1:
            return slots
        best_perm, best_cost = None, float("inf")
        for perm in itertools.permutations(range(n)):
            cost = sum(
                math.hypot(self.pursuers[i]["x"] - slots[perm[i]][0], self.pursuers[i]["y"] - slots[perm[i]][1])
                for i in range(n)
            )
            if cost < best_cost:
                best_cost, best_perm = cost, perm
        return [slots[best_perm[i]] for i in range(n)]

    def _mean_slot_distance(self) -> float:
        slots = self._role_slots()
        total = sum(
            math.hypot(self.pursuers[i]["x"] - slots[i][0], self.pursuers[i]["y"] - slots[i][1])
            for i in range(self.num_pursuers)
        )
        return total / self.num_pursuers

    def _resolve_evader_blocking(self) -> None:
        # Cars are solid: the evader cannot drive through a pursuer. Push it out of any
        # overlap and kill its speed on contact, so a tight surround physically pins it.
        # This is what turns "prevent movement" into a real, learnable win condition.
        e = self.evader
        blocked = False
        for _ in range(2):  # a couple of relaxation passes for tight surrounds
            for p in self.pursuers:
                dx, dy = e["x"] - p["x"], e["y"] - p["y"]
                d = math.hypot(dx, dy)
                if d < self.vehicle_collision_distance:
                    if d < 1e-6:
                        dx, dy, d = 1.0, 0.0, 1.0
                    push = self.vehicle_collision_distance - d
                    e["x"] += (dx / d) * push
                    e["y"] += (dy / d) * push
                    blocked = True
        if blocked:
            e["speed"] = 0.0

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self._spawn()
        self.previous_distances = self._evader_distances()
        self.previous_within = sum(1 for d in self.previous_distances if d <= self.capture_radius)
        self.previous_gap = self._max_angular_gap()
        self.previous_slot_distance = self._mean_slot_distance()
        self._pin_steps = 0
        self.previous_pin_steps = 0
        self._evader_prev = (self.evader["x"], self.evader["y"])
        info = {
            "stage": self.stage,
            "num_pursuers": self.num_pursuers,
            "capture_mode": self.capture_mode,
            "require_immobility": self.require_immobility,
            "use_car_cameras": self.use_car_cameras,
            "captured": False,
            "step": 0,
            "distances": list(self.previous_distances),
            "mean_distance": sum(self.previous_distances) / self.num_pursuers,
            "min_distance": min(self.previous_distances),
        }
        self.last_info = info
        return self._observation(), info

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        border_contacts = 0
        for index, p in enumerate(self.pursuers):
            self._drive(p, float(action[index * 2]), float(action[index * 2 + 1]))
            border_contacts += self._keep_inside(p)
        self._move_evader()
        self._resolve_evader_blocking()  # pursuers physically block the evader
        self._keep_inside(self.evader)

        # how far the evader actually moved this step (its "freedom"); near-zero => pinned
        evader_disp = math.hypot(self.evader["x"] - self._evader_prev[0], self.evader["y"] - self._evader_prev[1])
        self._evader_prev = (self.evader["x"], self.evader["y"])

        distances = self._evader_distances()
        progress = sum(prev - cur for prev, cur in zip(self.previous_distances, distances)) / self.num_pursuers
        self.previous_distances = distances
        within_now = sum(1 for d in distances if d <= self.capture_radius)
        surround_progress = within_now - self.previous_within
        self.previous_within = within_now
        gap_now = self._max_angular_gap()
        enclosure_progress = self.previous_gap - gap_now
        self.previous_gap = gap_now
        slot_distance_now = self._mean_slot_distance()
        slot_progress = self.previous_slot_distance - slot_distance_now
        self.previous_slot_distance = slot_distance_now
        collisions = self._pursuer_collisions()
        spacing = self._teammate_spacing()
        ttc_risk = self._ttc_risk()

        all_within = all(d <= self.capture_radius for d in distances)
        if self.capture_mode == CAPTURE_ANY:
            captured = any(d <= self.capture_radius for d in distances)
        elif self.capture_mode == CAPTURE_ALL:
            captured = all_within
        else:  # CAPTURE_SURROUND: enclosed AND held for K steps ("stay in radius"), plus
            # optionally motionless too ("stop motion", the require_immobility default).
            contained = all_within and self._is_enclosed()
            pinned = contained and (not self.require_immobility or evader_disp < self.immobile_disp)
            self._pin_steps = self._pin_steps + 1 if pinned else 0
            captured = self._pin_steps >= self.capture_hold_steps
        # Dense, potential-based reward for BUILDING/MAINTAINING the hold streak (not
        # just touching the ring once). Breaking a streak costs exactly what was gained,
        # so it cannot be farmed by hovering -- it specifically teaches "stop and hold".
        pin_progress = self._pin_steps - self.previous_pin_steps
        self.previous_pin_steps = self._pin_steps

        reward_terms = build_reward_contributions(
            progress=progress,
            surround_progress=surround_progress,
            enclosure_progress=enclosure_progress,
            slot_progress=slot_progress,
            pin_progress=pin_progress,
            spacing=spacing,
            ttc_risk=ttc_risk,
            border_contacts=border_contacts,
            collisions=collisions,
            captured=captured,
            action=action,
            weights=self.reward_weights,
        )
        reward = float(sum(reward_terms.values()))

        terminated = bool(captured)
        truncated = bool(self.step_count >= self.max_steps)
        info = {
            "stage": self.stage,
            "num_pursuers": self.num_pursuers,
            "capture_mode": self.capture_mode,
            "require_immobility": self.require_immobility,
            "use_car_cameras": self.use_car_cameras,
            "captured": captured,
            "step": self.step_count,
            "distances": list(distances),
            "mean_distance": sum(distances) / self.num_pursuers,
            "min_distance": min(distances),
            "within_capture": within_now,
            "max_angular_gap": gap_now,
            "enclosed": gap_now <= self.enclosure_max_gap,
            "mean_slot_distance": slot_distance_now,
            "evader_displacement": evader_disp,
            "evader_speed": self.evader["speed"],
            "pin_steps": self._pin_steps,
            "progress": progress,
            "border_contacts": border_contacts,
            "collisions": collisions,
            "ttc_risk": ttc_risk,
            "reward_terms": reward_terms,
        }
        self.last_info = info
        return self._observation(), reward, terminated, truncated, info
