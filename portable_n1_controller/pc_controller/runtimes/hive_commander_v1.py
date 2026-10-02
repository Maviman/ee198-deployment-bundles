"""The commander hive on the Orin: numpy + onnxruntime only (runtime "hive_commander_v1").

Nothing here imports torch or Isaac. It reproduces, operation for operation (geometry in float64, the network in float32), what the training env
(isaac_hive/tasks/capture/role_env.py on capture_env.py) and the playbooks (isaac_hive/roles.py) compute between
"poses at time t" and "[throttle, steer] per car":

    step(pursuers, evader, dt)
      1. observation  4 tokens x 31: base 12 (capture_env._get_observations) + role 12 + forecast 7 (role_env)
      2. policy.onnx  role logits (3, 5), settings (3, 2), residual (3, 2), observation scaler baked in
      3. decode       role action = argmax (0 = KEEP, k = role k - 1); settings -> the role's physical knobs;
                      the role's PLAYBOOK turns them into [throttle, steer]; command = clip(playbook +
                      residual_scale x residual, -1, 1)

Every number the port reads comes from policy.onnx.manifest.json, so a retrained model (another arena, another
car) changes the manifest, not this file. The source of truth is isaac_hive/deploy/hive_runtime.py in AI Training;
artifacts/deploy/<model>/hive_runtime.py is a copy. tests/test_hive_runtime.py pins it to the torch code.

Conventions: arena metres, origin at the arena centre, +x right, +y up, yaw CCW from +x (radians); v is the SIGNED
forward speed (m/s). Command rows are [throttle, steer] in [-1, 1], +steer = left. Throttle is a SPEED command in
training (target speed = throttle x max_speed_mps; the sim car has a speed loop): see the manifest's notes.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

RUNTIME = "hive_commander_v1"
ROLE_NAMES = ("CUTOFF", "FLANK", "PRESSURE", "CLOSE")
CUTOFF, FLANK, PRESSURE, CLOSE = range(4)
NUM_ROLES = 4
NUM_ROLE_ACTIONS = 5          # role action 0 = KEEP; action k >= 1 selects role k - 1
F32 = np.float32              # the network's input and output, and the returned commands
G = np.float64                # all geometry: float32 rounding flips near-tie branches (lanes, slots) between numpy
                              # and torch, and through the settings feedback the gap grows past 1e-5 (2026-10-02)
_BIG = float(np.finfo(np.float32).max) / 4


# ---- geometry (isaac_hive/roles.py) ----------------------------------------------------------------------------
def wrap_to_pi(a):
    """roles.wrap_to_pi: remainder(a + pi, 2 pi) - pi, range [-pi, pi)."""
    return np.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def wrap_atan2(a):
    """scripted_team.wrap_to_pi (used by the CLOSE slot assignment): atan2(sin a, cos a), range (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def unit(a):
    return np.stack((np.cos(a), np.sin(a)), axis=-1)


def headings(k: int):
    return unit(np.arange(k, dtype=G) * (2.0 * math.pi / k))


def escape_per_heading(ppos, epos, hw: float, hh: float, u, walls: bool = True):
    """(B, K) escape distance of the evader along each heading (equal-speed Voronoi bound, optional walls)."""
    diag = 2.0 * math.sqrt(hw * hw + hh * hh)
    d = ppos - epos[:, None, :]                                        # (B, n, 2)
    dot = np.einsum("bnc,kc->bnk", d, u)                               # (B, n, K)
    n2 = (d * d).sum(-1, keepdims=True)
    t_p = np.where(dot > 1e-9, n2 / np.maximum(dot, 1e-9) * 0.5, _BIG).astype(G).min(axis=1)
    if walls:
        ux, uy = u[:, 0][None, :], u[:, 1][None, :]
        okx, oky = np.abs(ux) > 1e-9, np.abs(uy) > 1e-9
        tx = (hw * np.sign(ux) - epos[:, 0:1]) / np.where(okx, ux, 1.0).astype(G)
        ty = (hh * np.sign(uy) - epos[:, 1:2]) / np.where(oky, uy, 1.0).astype(G)
        t_wall = np.maximum(np.minimum(np.where(okx, tx, _BIG), np.where(oky, ty, _BIG)), 0.0).astype(G)
        t_p = np.minimum(t_p, t_wall)
    return np.clip(t_p, 0.0, diag).astype(G)


def intercept_distance(ppos, epos, lane, hw: float, hh: float):
    """(B, n) how far along its lane the evader gets before each car can be there too (|d|^2 / (2 d.u))."""
    diag = 2.0 * math.sqrt(hw * hw + hh * hh)
    d = ppos - epos[:, None, :]
    dot = (d * lane).sum(-1)
    n2 = (d * d).sum(-1)
    t = np.where(dot > 1e-9, n2 / np.maximum(dot, 1e-9) * 0.5, diag).astype(G)
    return np.clip(t, 0.0, diag).astype(G)


def lanes(ppos, epos, hw, hh, walls: bool, k: int, second_lane_min_deg: float, plateau_m: float = 0.05):
    """roles.lanes: (u_star, u2, ang_star, ang2, escape)."""
    u = headings(k)
    t = escape_per_heading(ppos, epos, hw, hh, u, walls=walls)               # (B, K)
    ang_k = np.arange(k, dtype=G) * (2.0 * math.pi / k)
    top = (t >= t.max(axis=1, keepdims=True) - plateau_m).astype(G)
    ang_star = np.arctan2((top * np.sin(ang_k)).sum(1), (top * np.cos(ang_k)).sum(1))
    sep = np.abs(wrap_to_pi(ang_k[None, :] - ang_star[:, None]))
    score = t - 1e-3 * np.abs(sep - 0.5 * math.pi)
    score = np.where(sep >= math.radians(second_lane_min_deg), score, -1e9).astype(G)
    ang2 = ang_k[score.argmax(axis=1)]
    u_star = unit(ang_star)
    esc = intercept_distance(ppos, epos, u_star[:, None, :], hw, hh).min(axis=1)
    return u_star, unit(ang2), ang_star, ang2, esc


def ring_slots(bearings):
    """scripted_team.ring_slots (anchor "mean")."""
    n = bearings.shape[1]
    sx, sy = np.cos(bearings).sum(1), np.sin(bearings).sum(1)
    degenerate = (sx * sx + sy * sy) < (1e-6 * n) ** 2
    base = np.where(degenerate, bearings[:, 0], np.arctan2(sy, sx) / 1.0).astype(G)
    k = np.arange(n, dtype=G)
    return base[:, None] + k[None, :] * (2.0 * math.pi / n)


def assign_slots(bearings, slots):
    """scripted_team.assign_slots: the least-total-travel permutation, ties to the first in itertools order."""
    n = bearings.shape[1]
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)    # (P, n)
    cand = slots[:, perms]                                                       # (B, P, n)
    cost = np.abs(wrap_atan2(cand - bearings[:, None, :])).sum(-1)               # (B, P)
    best = cost.argmin(axis=1)
    return cand[np.arange(bearings.shape[0]), best], perms[best]


def ctrv_predict(pos, yaw, spd, yaw_rate, t: float):
    """roles.ctrv_predict: constant turn rate and velocity, t seconds ahead."""
    w = yaw_rate
    straight = np.abs(w) < 1e-3
    w_safe = np.where(straight, 1.0, w).astype(G)
    th1 = yaw + w * t
    arc_x = spd / w_safe * (np.sin(th1) - np.sin(yaw))
    arc_y = spd / w_safe * (np.cos(yaw) - np.cos(th1))
    lin_x = spd * t * np.cos(yaw)
    lin_y = spd * t * np.sin(yaw)
    dx = np.where(straight, lin_x, arc_x)
    dy = np.where(straight, lin_y, arc_y)
    return (pos + np.stack((dx, dy), axis=-1)).astype(G)


# ---- playbooks (isaac_hive/roles.py) ---------------------------------------------------------------------------
class Playbook:
    """roles.settings_to_physical + role_targets + track for one PlaybookCfg (the manifest's "playbook" block)."""

    def __init__(self, cfg: dict, setting_ranges: dict):
        self.c = dict(cfg)
        self.hw, self.hh = float(cfg["half_extent"][0]), float(cfg["half_extent"][1])
        self.lo = np.array([[setting_ranges[r][k][0] for k in range(2)] for r in ROLE_NAMES], dtype=G)
        self.hi = np.array([[setting_ranges[r][k][1] for k in range(2)] for r in ROLE_NAMES], dtype=G)

    # derived distances (PlaybookCfg properties)
    @property
    def contact_floor(self) -> float:
        return self.c["contact_nose_nose_m"] + self.c["contact_margin_m"]

    @property
    def mate_block(self) -> float:
        return self.c["contact_nose_nose_m"] + self.c["mate_gap_m"]

    @property
    def close_radius(self) -> float:
        return max(self.c["capture_radius"] - self.c["close_margin_m"], self.contact_floor)

    def physical(self, role, z):
        lo, hi = self.lo[role], self.hi[role]
        return (lo + (np.clip(z, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)).astype(G)

    def targets(self, role, phys, ppos, epos, evel):
        """roles.role_targets -> (target (B, n, 2), keepout (B, n), speed_cap (B, n), u_star (B, 2))."""
        c, hw, hh = self.c, self.hw, self.hh
        walls = bool(c["walls_count"])
        u_star, _, ang_star, ang2, _ = lanes(ppos, epos, hw, hh, walls, int(c["num_headings"]),
                                             c["second_lane_min_deg"])
        e = epos[:, None, :]
        rel = ppos - e
        d_e = np.linalg.norm(rel, axis=-1)
        brg = np.arctan2(rel[..., 1], rel[..., 0])
        a1, a2 = phys[..., 0], phys[..., 1]

        if evel is not None and c["cutoff_lead_s"] > 0.0:
            e_lead = epos + evel * c["cutoff_lead_s"]
            e_lead = np.stack((np.clip(e_lead[:, 0], -hw, hw), np.clip(e_lead[:, 1], -hh, hh)), axis=-1).astype(G)
            _, _, ls_star, ls2, _ = lanes(ppos, e_lead, hw, hh, walls, int(c["num_headings"]), c["second_lane_min_deg"])
        else:
            e_lead, ls_star, ls2 = epos, ang_star, ang2
        lane_ang = np.where(role == FLANK, ls2[:, None], ls_star[:, None]) + np.deg2rad(a1)
        lane_u = unit(lane_ang)
        t_i = intercept_distance(ppos, e_lead, lane_u, hw, hh)
        t_lane = e_lead[:, None, :] + np.maximum(a2, np.minimum(t_i, c["cutoff_max_m"]))[..., None] * lane_u

        blocker = ((role == CUTOFF) | (role == FLANK)).astype(G)
        w_b = blocker[..., None]
        nb = w_b.sum(axis=1)                                                    # (B, 1)
        blk_c = (ppos * w_b).sum(axis=1) / np.maximum(nb, 1.0)                  # (B, 2)
        n = ppos.shape[1]
        others = (ppos.sum(axis=1, keepdims=True) - ppos) / max(n - 1, 1)
        toward = np.where((nb > 0)[:, None, :], np.broadcast_to(blk_c[:, None, :], ppos.shape), others) - e
        toward = np.where(np.linalg.norm(toward, axis=-1, keepdims=True) > 1e-3, toward,
                          np.broadcast_to(-u_star[:, None, :], toward.shape))
        if n == 1:
            toward = -u_star[:, None, :]
        push_ang = np.arctan2(toward[..., 1], toward[..., 0]) + math.pi + np.deg2rad(a1)
        t_press = e + a2[..., None] * unit(push_ang)

        r_c = self.close_radius
        slot_ang = assign_slots(brg, ring_slots(brg))[0] if n > 1 else brg
        t_close = e + r_c * unit(slot_ang + np.deg2rad(a1))

        tgt = np.where((role == PRESSURE)[..., None], t_press, t_lane)
        tgt = np.where((role == CLOSE)[..., None], t_close, tgt)
        lim_x, lim_y = hw - c["wall_inset_m"], hh - c["wall_inset_m"]
        tgt = np.stack((np.clip(tgt[..., 0], -lim_x, lim_x), np.clip(tgt[..., 1], -lim_y, lim_y)), axis=-1).astype(G)

        floor = self.contact_floor
        keep = np.where(role == PRESSURE, np.maximum(c["pressure_keepout_frac"] * a2, floor),
                        np.maximum(0.8 * a2, floor))
        keep = np.where(role == CLOSE, r_c - 0.02, keep).astype(G)
        cap = np.ones_like(d_e)
        cap = np.where(role == PRESSURE, c["pressure_speed_cap"], cap)
        cap = np.where(role == CLOSE, np.clip(a2, 0.0, 1.0), cap).astype(G)
        return tgt, keep, cap, u_star

    def track(self, target, keepout, speed_cap, ppos, pyaw, epos, evel):
        """roles.track -> (B, n, 2) [throttle, steer]."""
        c = self.c
        n = ppos.shape[1]
        vmax = c["max_speed_mps"]
        e = epos[:, None, :]
        rel_e = ppos - e
        d_e = np.linalg.norm(rel_e, axis=-1)
        th_p = np.arctan2(rel_e[..., 1], rel_e[..., 0])

        ab = target - ppos
        t = np.clip(((e - ppos) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0.0, 1.0)
        clear = np.linalg.norm(e - (ppos + t[..., None] * ab), axis=-1)
        route = (clear < keepout) & (d_e > keepout - 0.05)
        ring = np.maximum(keepout + c["route_clear_m"], np.minimum(d_e, keepout + 0.6))
        tgt_rel = target - e
        th_t = np.arctan2(tgt_rel[..., 1], tgt_rel[..., 0])
        delta = wrap_to_pi(th_t - th_p)
        alpha = np.arccos(np.minimum(ring / np.maximum(d_e, 1e-6), 1.0))
        adv = np.minimum(np.abs(delta), np.maximum(alpha, math.radians(c["route_arc_step_deg"])))
        wp_route = e + ring[..., None] * unit(th_p + np.sign(delta) * adv)
        wp = np.where(route[..., None], wp_route, target)
        lim_x, lim_y = self.hw - c["wall_inset_m"], self.hh - c["wall_inset_m"]
        wp = np.stack((np.clip(wp[..., 0], -lim_x, lim_x), np.clip(wp[..., 1], -lim_y, lim_y)), axis=-1).astype(G)
        path = np.where(route, np.linalg.norm(wp - ppos, axis=-1) + np.linalg.norm(target - wp, axis=-1),
                        np.linalg.norm(ab, axis=-1))

        dw = wp - ppos
        brg = wrap_to_pi(np.arctan2(dw[..., 1], dw[..., 0]) - pyaw)
        s = np.clip(c["approach_gain_per_m"] * (path - c["stop_tol_m"]), 0.0, 1.0) * speed_cap
        cosb = np.cos(brg)
        fwd = s * np.maximum(cosb, c["min_turn_throttle"])
        rev_zone = (np.abs(brg) > math.radians(c["reverse_deg"])) & (path < c["reverse_within_m"])
        thr = np.where(rev_zone, -(np.minimum(s, c["reverse_max"]) * np.maximum(-cosb, c["min_turn_throttle"])), fwd)
        if c["cramped_room_m"] is not None:
            raise NotImplementedError("cramped_room_m (three-point turn) is off in every trained model")

        to_e = wrap_to_pi(th_p + math.pi - pyaw)
        e_ahead = np.abs(to_e) < 0.5 * math.pi
        room = np.maximum(d_e - c["contact_nose_nose_m"] - 0.02, 0.0)
        e_cap = np.sqrt(2.0 * c["evader_decel_mps2"] * room) / vmax
        if evel is not None:
            u_los = -rel_e / np.maximum(d_e, 1e-6)[..., None]
            e_cap = e_cap + np.maximum((evel[:, None, :] * u_los).sum(-1), 0.0) / vmax
        thr = np.where((thr > 0) & e_ahead, np.minimum(thr, e_cap), thr)
        if evel is not None and c["ff_gain"] > 0.0:
            heading = np.stack((np.cos(pyaw), np.sin(pyaw)), axis=-1)
            ff = c["ff_gain"] * (heading * evel[:, None, :]).sum(-1) / vmax
            thr = np.clip(thr + ff, -c["reverse_max"], 1.0)
            thr = np.where((thr > 0) & e_ahead, np.minimum(thr, e_cap), thr)
        if n > 1:
            rel = ppos[:, None, :, :] - ppos[:, :, None, :]                     # (B, n, n, 2): j relative to i
            dist = np.linalg.norm(rel, axis=-1) + np.eye(n, dtype=G) * 1e6
            ang = wrap_to_pi(np.arctan2(rel[..., 1], rel[..., 0]) - pyaw[..., None])
            in_cone = np.abs(ang) < math.radians(c["mate_cone_deg"])
            gap = np.maximum(dist - self.mate_block, 0.0)
            m_cap = np.where(in_cone, np.sqrt(2.0 * c["mate_decel_mps2"] * gap) / vmax, np.inf).min(axis=-1)
            thr = np.where(thr > 0, np.minimum(thr, m_cap), thr)
            blocked = (in_cone & (dist < self.mate_block + 0.02)).any(axis=-1) & (path > 0.2)
            allp = np.concatenate((ppos, e), axis=1)                             # (B, n+1, 2)
            relb = allp[:, None, :, :] - ppos[:, :, None, :]                     # (B, n, n+1, 2)
            db = np.linalg.norm(relb, axis=-1) + np.concatenate(
                (np.eye(n, dtype=G) * 1e6, np.zeros((n, 1), dtype=G)), axis=1)
            angb = wrap_to_pi(np.arctan2(relb[..., 1], relb[..., 0]) - pyaw[..., None])
            rear_busy = ((db < self.mate_block) & (np.abs(angb) > 0.5 * math.pi)).any(axis=-1)
            thr = np.where(blocked & ~rear_busy, -c["unblock_reverse"], thr)
        if c["edge_margin_m"] is not None:
            raise NotImplementedError("edge_margin_m (edge speed cap) is off in every trained model")

        steer = np.clip(brg / c["max_steer"], -1.0, 1.0)
        backing = thr < 0
        steer = np.where(backing, np.clip(-wrap_to_pi(brg - math.pi) / c["max_steer"], -1.0, 1.0), steer)
        if c["lat_accel_max_mps2"] is not None:
            quant = np.where(np.abs(steer) > 1.0 / 3.0, np.sign(steer), 0.0) if c.get("binary_steer") else steer
            tan_d = np.maximum(np.tan(np.abs(quant) * c["max_steer"]), 1e-3)
            v_ok = np.sqrt(c["lat_accel_max_mps2"] * c["wheelbase_m"] / tan_d)
            cap_t = np.minimum(v_ok / vmax, 1.0)
            thr = np.sign(thr) * np.minimum(np.abs(thr), cap_t)
        return np.stack((np.clip(thr, -1.0, 1.0), steer), axis=-1).astype(G)

    def actions(self, role, z, ppos, pyaw, epos, evel):
        """roles.playbook_actions -> ((B, n, 2) [throttle, steer], target (B, n, 2))."""
        tgt, keep, cap, _ = self.targets(role, self.physical(role, z), ppos, epos, evel)
        return self.track(tgt, keep, cap, ppos, pyaw, epos, evel), tgt


# ---- observation (capture_env + role_env) ----------------------------------------------------------------------
def observation(m: dict, pb: Playbook, ppos, pyaw, pspd, epos, eyaw, espd, role, z, tir, prev_yaw, dt: float):
    """One env's (B = 1 allowed) observation. Returns (obs (B, 124), new prev_yaw (B, n+1))."""
    b, n = ppos.shape[0], ppos.shape[1]
    hw, hh, diag, vmax = m["arena"]["half_w"], m["arena"]["half_h"], m["arena"]["diag"], m["car"]["max_speed_mps"]

    # base 12 per token (capture_env._get_observations)
    d = epos[:, None, :] - ppos
    dist = np.linalg.norm(d, axis=-1)
    bearing = np.arctan2(d[..., 1], d[..., 0]) - pyaw

    def clearances(pos):
        return np.stack(((pos[..., 0] + hw) / (2.0 * hw), (hw - pos[..., 0]) / (2.0 * hw),
                         (pos[..., 1] + hh) / (2.0 * hh), (hh - pos[..., 1]) / (2.0 * hh)), axis=-1)

    pfeat = np.concatenate((ppos[..., 0:1] / hw, ppos[..., 1:2] / hh, np.sin(pyaw)[..., None], np.cos(pyaw)[..., None],
                            (pspd / vmax)[..., None], (dist / diag)[..., None], np.sin(bearing)[..., None],
                            np.cos(bearing)[..., None], clearances(ppos)), axis=-1)               # (B, n, 12)
    zeros = np.zeros((b, 1), dtype=G)
    efeat = np.concatenate((epos[:, 0:1] / hw, epos[:, 1:2] / hh, np.sin(eyaw)[:, None], np.cos(eyaw)[:, None],
                            (espd / vmax)[:, None], zeros, zeros, zeros, clearances(epos)), axis=-1)[:, None, :]

    # role 12 per pursuer (role_env._get_observations)
    evel = (espd[:, None] * np.stack((np.cos(eyaw), np.sin(eyaw)), axis=-1)).astype(G) \
        if m["playbook_evader_velocity"] else None
    pbact, tgt = pb.actions(role, z, ppos, pyaw, epos, evel)
    rel = tgt - ppos
    tdist = np.linalg.norm(rel, axis=-1) / diag
    tbrg = np.arctan2(rel[..., 1], rel[..., 0]) - pyaw
    one_hot = np.eye(NUM_ROLES, dtype=G)[role]
    extra = np.concatenate((one_hot, np.clip(tir / m["time_in_role_scale_s"], 0.0, 3.0)[..., None], z,
                            tdist[..., None], np.sin(tbrg)[..., None], np.cos(tbrg)[..., None], pbact), axis=-1)

    # forecast 7 per token (role_env._forecasts)
    yaw_all = np.concatenate((pyaw, eyaw[:, None]), axis=1)
    spd_all = np.concatenate((pspd, espd[:, None]), axis=1)
    pos_all = np.concatenate((ppos, epos[:, None, :]), axis=1)
    with np.errstate(invalid="ignore"):                               # prev_yaw is NaN on a match's first tick
        rate = np.where(np.isfinite(prev_yaw), wrap_to_pi(yaw_all - prev_yaw) / dt, 0.0).astype(G)
    f05 = ctrv_predict(pos_all, yaw_all, spd_all, rate, m["forecast"]["t_short_s"])
    f10 = ctrv_predict(pos_all, yaw_all, spd_all, rate, m["forecast"]["t_long_s"])
    walls = bool(m["forecast"]["walls_clamp"])
    if walls:
        f05 = np.stack((np.clip(f05[..., 0], -hw, hw), np.clip(f05[..., 1], -hh, hh)), axis=-1).astype(G)
        f10 = np.stack((np.clip(f10[..., 0], -hw, hw), np.clip(f10[..., 1], -hh, hh)), axis=-1).astype(G)
    u_k = headings(int(m["forecast"]["num_headings"]))
    esc_now = escape_per_heading(ppos, epos, hw, hh, u_k, walls=walls).max(axis=1)
    esc_pred = escape_per_heading(f10[:, :n], f10[:, n], hw, hh, u_k, walls=walls).max(axis=1)
    scale = np.array([hw, hh], dtype=G)
    ev_only = np.zeros((b, n + 1, 2), dtype=G)
    ev_only[:, n, 0] = esc_now / diag
    ev_only[:, n, 1] = esc_pred / diag
    forecast = np.concatenate((f05 / scale, f10 / scale, np.clip(rate / 3.0, -3.0, 3.0)[..., None], ev_only), axis=-1)

    ptok = np.concatenate((pfeat, extra, forecast[:, :n]), axis=-1)
    etok = np.concatenate((efeat, np.zeros((b, 1, extra.shape[-1]), dtype=G), forecast[:, n:]), axis=-1)
    obs = np.concatenate((ptok, etok), axis=1).reshape(b, -1)
    obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -m["obs_clip"], m["obs_clip"]).astype(F32)
    return obs, yaw_all.astype(G)


# ---- the runtime -------------------------------------------------------------------------------------------------
class CommanderRuntime:
    """The trained commander driving three pursuers. Keep one instance per match; call reset() between matches."""

    def __init__(self, model_dir, seed=None, session=None):
        self.dir = Path(model_dir)
        self.m = json.loads((self.dir / "policy.onnx.manifest.json").read_text(encoding="utf-8"))
        if self.m.get("runtime") != RUNTIME:
            raise ValueError(f"manifest runtime {self.m.get('runtime')!r} is not {RUNTIME!r}")
        if self.m.get("use_car_cameras"):
            raise ValueError("this runtime is overhead-camera only")
        self.n = int(self.m["num_pursuers"])
        self.pb = Playbook(self.m["playbook"], self.m["setting_ranges"])
        self.residual_scale = float(self.m["residual_scale"])
        if session is None:
            import onnxruntime as ort
            session = ort.InferenceSession(str(self.dir / "policy.onnx"), providers=["CPUExecutionProvider"])
        self.sess = session
        self.rng = np.random.default_rng(seed)
        self.last: dict = {}
        self.reset()

    def reset(self, roles=None) -> None:
        """roles None = what training does (uniformly random roles); else names or indices, one per car."""
        if roles is None:
            r = self.rng.integers(0, NUM_ROLES, size=self.n)
        else:
            r = np.array([ROLE_NAMES.index(x) if isinstance(x, str) else int(x) for x in roles], dtype=np.int64)
        self._role = r.astype(np.int64)[None, :]                       # (1, n)
        self._z = np.zeros((1, self.n, 2), dtype=G)
        self._tir = np.zeros((1, self.n), dtype=G)
        self._prev_yaw = np.full((1, self.n + 1), np.nan, dtype=G)
        self.last = {}

    @property
    def roles(self) -> list[str]:
        return [ROLE_NAMES[int(i)] for i in self._role[0]]

    def policy(self, obs):
        out = self.sess.run(None, {"observations": obs.astype(F32)})
        return [np.asarray(o, dtype=F32) for o in out]                 # role_logits, settings, residual

    def step(self, pursuers, evader, dt: float = 0.1):
        """pursuers (n, 4) rows [x, y, yaw, v_signed]; evader (4,). Returns (n, 2) [throttle, steer] in [-1, 1]."""
        p = np.asarray(pursuers, dtype=G).reshape(self.n, 4)
        ev = np.asarray(evader, dtype=G).reshape(4)
        ppos, pyaw, pspd = p[None, :, 0:2], p[None, :, 2], p[None, :, 3]
        epos, eyaw, espd = ev[None, 0:2], ev[None, 2], ev[None, 3]
        obs, self._prev_yaw = observation(self.m, self.pb, ppos, pyaw, pspd, epos, eyaw, espd, self._role, self._z,
                                          self._tir, self._prev_yaw, float(dt))
        logits, settings, residual = self.policy(obs)
        act = logits.argmax(axis=-1)                                   # (1, n): 0 = KEEP
        new = np.where(act > 0, act - 1, self._role)
        switched = new != self._role
        self._role = new
        self._tir = np.where(switched, 0.0, self._tir + float(dt)).astype(G)
        self._z = np.clip(settings, -1.0, 1.0).astype(G)          # float32 network values, held exactly
        evel = (espd[:, None] * np.stack((np.cos(eyaw), np.sin(eyaw)), axis=-1)).astype(G) \
            if self.m["playbook_evader_velocity"] else None
        pbact, _ = self.pb.actions(self._role, self._z, ppos, pyaw, epos, evel)
        res = np.clip(residual.astype(G), -1.0, 1.0) * self.residual_scale
        cmd = np.clip(pbact + res, -1.0, 1.0).astype(F32)
        self.last = {"obs": obs[0], "role_logits": logits[0], "settings": settings[0], "residual": residual[0],
                     "playbook": pbact[0].astype(F32), "command": cmd[0], "roles": self.roles}
        return cmd[0]
