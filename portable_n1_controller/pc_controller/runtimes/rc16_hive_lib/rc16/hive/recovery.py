"""Out-of-bounds recovery (rc16/hive/CONTRACT.md, "recovery.py"; user decision 4 in DECISIONS D26).

A pawn whose body centre is beyond the tape stops following its play slot and runs this subroutine instead.
Pure torch, one state machine per pawn, batched (N envs x 3 pawns).

Terms (metres, m/s, seconds):
- beyond the tape: geometry.dist_outside(centre) > 0 for the square |x|, |y| <= S.ARENA_HALF_M.
- seen: the cameras report the pawn this tick. While a pawn is NOT seen, the xy / yaw / v passed in are ignored;
  the last seen values are used instead.
- shield: rc16/hive/shield.py, the limiter the command is expected to pass after this module. `turn_radius`
  (constructor) must be the value the shield is run with (None = its default).
- recovering: the pawn is in this subroutine. It starts when the centre is beyond the tape (seen), or when an
  unseen pawn's dead-reckoned centre (last seen position + last seen speed x heading x time unseen, at most
  DEAD_RECKON_MAX_S) is beyond the tape. It ends when the pawn is seen with its centre REJOIN_INSIDE_M (0.3 m)
  or more inside the tape on both axes.
- history: the last HISTORY_S (1 s) of ticks in which the pawn was seen: the (throttle, steer) actually returned
  by step() and the measured speed. Ticks in which the pawn is unseen are not recorded, so the history always
  ends at the last place the cameras saw it.

What a recovering pawn does:
1. seen: brake (reverse throttle against its motion) until |v| < STOPPED_MPS once, then drive toward the arena
   centre at about DRIVE_IN_TARGET_MPS and never above DRIVE_IN_MPS (0.5 m/s), steering by pure pursuit of the
   centre. It goes in LEAD END first: the end (nose or tail) that faced the centre when the drive-in began; the
   choice is kept (it changes only if the centre gets more than 110 deg off that end).
   The command is expected to pass through the shield afterwards, and the shield refuses a throttle that takes
   an outside car further out. So when the lead end has less than GEAR_ROOM_M of room as the shield counts it
   (shield.rooms on the tape minus 0.05 m: 0 when that direction points out of the square), the pawn drives the
   other way on OPPOSITE lock, which swings the lead end toward the centre (a three-point turn), until the lead
   end has room again. (Review 2026-11-17: without this, 19 % of pawns placed outside the tape at a random
   heading stood still for ever once their command went through the shield: recovery asked for the end facing
   the centre and the shield allowed only the other one.)
2. not seen: brake blind for |last seen v| / BRAKE_PLAN seconds, then replay the history backwards, newest
   first: the SAME steer, and a throttle of magnitude DRIVE_IN_MPS / TOP_SPEED pointing AGAINST the direction
   the car was moving at that tick (against the recorded throttle if it was standing still). Each recorded tick
   is held for ceil(|recorded v| / DRIVE_IN_MPS) ticks (1 to REPLAY_HOLD_MAX) so the slow replay covers about
   the same ground. When the history runs out it sends zero. Seen again -> 1;
3. unseen for GIVE_UP_S (3 s) while recovering: zero command and `gave_up` = True, until it is seen again.
"""

from __future__ import annotations

import math

import torch

from rc16 import specs as S
from rc16.hive import geometry as G
from rc16.hive.shield import LINE_INSET_M as SHIELD_INSET_M
from rc16.hive.shield import rooms

GEAR_ROOM_M = 0.08          # m: with less room than this (as the shield counts it) ahead of its lead end the pawn
                            # drives the other way; below the shield's crawl room for a car coming back in
                            # (0.106 m), so a direction the shield allows is never dropped
GEAR_ROOM_KEEP_M = 0.30     # m: once it drives the other way it keeps to that until the lead end has this much room
LEAD_FLIP_COS = 0.34        # the lead end changes only when the centre is more than 110 deg off it (cos 110 = -0.34)
HISTORY_S = 1.0             # s of sent commands kept per pawn
CONTROL_DT = 1.0 / S.CONTROL_HZ
HISTORY_STEPS = int(round(HISTORY_S / CONTROL_DT))      # 10 ticks at 10 Hz
REJOIN_INSIDE_M = 0.30      # m inside the tape (both axes) at which a pawn leaves recovery
DRIVE_IN_MPS = 0.50         # m/s: speed limit while driving back in or replaying
DRIVE_IN_TARGET_MPS = 0.40  # m/s: speed asked for while driving back in
STOPPED_MPS = 0.15          # m/s: under this the pawn counts as stopped
GIVE_UP_S = 3.0             # s unseen (while recovering) after which the pawn stops and stays stopped
DEAD_RECKON_MAX_S = 1.0     # s: longest time an unseen pawn's position is extrapolated
REPLAY_HOLD_MAX = 4         # ticks: longest a single recorded tick is held during the replay
LOW_THROTTLE = DRIVE_IN_MPS / S.TOP_SPEED_MPS           # throttle that settles at DRIVE_IN_MPS
BRAKE_THROTTLE = 0.5        # blind-brake throttle magnitude


class Recovery:
    """Per-pawn out-of-bounds state machine, batch (N, 3)."""

    def __init__(self, n: int, device="cpu", turn_radius=None):
        """n envs; turn_radius: the value the shield is run with (None = its default, the straight ray), so that
        the drive-in gear is chosen with the room the shield will actually allow."""
        self.n, self.device = n, torch.device(device)
        self.turn_radius = turn_radius
        z = lambda *s, dtype=torch.float32: torch.zeros(n, 3, *s, dtype=dtype, device=self.device)   # noqa: E731
        self.recovering = z(dtype=torch.bool)
        self.gave_up = z(dtype=torch.bool)
        self.stopped = z(dtype=torch.bool)          # the seen procedure has finished its "stop" once
        self.lead = z()                             # the end that goes in first: +1 nose, -1 tail, 0 not chosen yet
        self.has_last = z(dtype=torch.bool)         # a last seen pose exists
        self.last_xy, self.last_yaw, self.last_v = z(2), z(), z()
        self.t_unseen = z()                         # s since the pawn was last seen
        self.t_blind = z()                          # s spent unseen while recovering (resets when seen)
        self.brake_left = z(dtype=torch.long)       # blind-brake ticks still to send
        self.rp_idx = z(dtype=torch.long)           # replay position in the history (0 = newest)
        self.rp_hold = z(dtype=torch.long)          # ticks the current history entry has been held
        self.hist_cmd = z(HISTORY_STEPS, 2)         # [..., 0, :] is the newest (throttle, steer)
        self.hist_v = z(HISTORY_STEPS)
        self.hist_len = z(dtype=torch.long)

    def reset(self, ids=None):
        """Forget everything for the envs `ids` (tensor / list of env indices; None = all)."""
        ids = slice(None) if ids is None else ids
        for t in (self.recovering, self.gave_up, self.stopped, self.has_last):
            t[ids] = False
        for t in (self.last_xy, self.last_yaw, self.last_v, self.t_unseen, self.t_blind, self.hist_cmd, self.hist_v,
                  self.lead):
            t[ids] = 0.0
        for t in (self.brake_left, self.rp_idx, self.rp_hold, self.hist_len):
            t[ids] = 0

    # -- the two procedures --------------------------------------------------------------------------------
    def _seen_cmd(self, xy, yaw, v):
        """Procedure 1. Returns (cmd (N,3,2), stopped_now (N,3) bool, lead_now (N,3) +1 nose / -1 tail)."""
        speed = v.abs()
        stopped = self.stopped | (speed < STOPPED_MPS)
        brake = -torch.sign(v) * speed.clamp(0.25, 0.75)
        # the centre of the arena in the body frame: bx along the nose, by to the left
        b = G.body_frame(-xy, yaw)
        bx, by = b[..., 0], b[..., 1]
        one = torch.ones_like(bx)
        face = torch.where(bx >= 0.0, one, -one)                     # the end that faces the centre now
        lead = torch.where(self.lead == 0.0, face, self.lead)
        lead = torch.where(lead * bx < -LEAD_FLIP_COS * b.norm(dim=-1), face, lead)
        # the gear: lead end first, unless the shield would refuse that direction (it points out of the square,
        # or the line is right there): then the other way, until the lead end has room again
        room_f, room_r = rooms(xy, yaw, S.ARENA_HALF_M - SHIELD_INSET_M, self.turn_radius)
        room_l = torch.where(lead > 0, room_f, room_r)
        room_o = torch.where(lead > 0, room_r, room_f)
        other_now = (lead * v) < -0.05                               # already driving the other way
        need = torch.where(other_now, torch.full_like(bx, GEAR_ROOM_KEEP_M), torch.full_like(bx, GEAR_ROOM_M))
        gear = torch.where((room_l < need) & (room_o > room_l), -lead, lead)
        # steering: pure pursuit of the centre with the lead end (full lock toward it while it is behind that
        # end); driving the other way, the opposite full lock, which turns the lead end toward the centre
        kappa = 2.0 * by / (bx * bx + by * by).clamp(min=1e-6)
        delta = torch.atan(S.WHEELBASE_M * kappa).clamp(-S.MAX_STEER_RAD, S.MAX_STEER_RAD)
        side = torch.where(by >= 0.0, one, -one)
        delta = torch.where(lead * bx < 0.0, side * S.MAX_STEER_RAD, delta)
        delta = torch.where(gear != lead, -side * S.MAX_STEER_RAD, delta)
        s = gear * v                                # speed along the chosen gear
        push = gear * ((2.0 * DRIVE_IN_TARGET_MPS - s) / S.TOP_SPEED_MPS).clamp(0.0, 0.30)
        thr = torch.where(s > DRIVE_IN_MPS, -gear * 0.25, push)
        thr = torch.where(s < -0.05, gear * 0.25, thr)
        drive_in = torch.stack([thr, delta / S.MAX_STEER_RAD], -1)
        stop = torch.stack([brake, torch.zeros_like(brake)], -1)
        return torch.where(stopped.unsqueeze(-1), drive_in, stop), stopped, lead

    def _blind_cmd(self, active):
        """Procedure 2 for the pawns in `active` (N,3) bool; advances their replay state. Returns cmd (N,3,2)."""
        braking = self.brake_left > 0
        brake_thr = torch.where(self.last_v.abs() > 0.1, -torch.sign(self.last_v) * BRAKE_THROTTLE,
                                torch.zeros_like(self.last_v))
        idx = self.rp_idx.clamp(max=HISTORY_STEPS - 1)
        h_cmd = torch.gather(self.hist_cmd, 2, idx[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        h_v = torch.gather(self.hist_v, 2, idx[..., None]).squeeze(2)
        h_thr = h_cmd[..., 0]
        moved = torch.where(h_v.abs() > 0.05, torch.sign(h_v),
                            torch.where(h_thr.abs() >= S.PWM_DEADBAND, torch.sign(h_thr), torch.zeros_like(h_thr)))
        has = self.rp_idx < self.hist_len
        rp = torch.stack([-moved * LOW_THROTTLE, h_cmd[..., 1]], -1)
        rp = torch.where(has.unsqueeze(-1), rp, torch.zeros_like(rp))
        cmd = torch.where(braking.unsqueeze(-1), torch.stack([brake_thr, torch.zeros_like(brake_thr)], -1), rp)
        # advance
        replaying = active & ~braking & has
        need = torch.ceil(h_v.abs() / DRIVE_IN_MPS).long().clamp(1, REPLAY_HOLD_MAX)
        hold = self.rp_hold + 1
        nxt = replaying & (hold >= need)
        self.rp_idx = torch.where(nxt, self.rp_idx + 1, self.rp_idx)
        self.rp_hold = torch.where(nxt, torch.zeros_like(hold), torch.where(replaying, hold, self.rp_hold))
        self.brake_left = torch.where(active & braking, self.brake_left - 1, self.brake_left)
        return cmd

    # -- one control tick -------------------------------------------------------------------------------------
    def step(self, xy, yaw, v, seen, cmd_in, dt: float = 0.1):
        """Advance one tick. xy (N,3,2), yaw (N,3), v (N,3) signed m/s, seen (N,3) bool, cmd_in (N,3,2).

        Returns (cmd (N,3,2), recovering (N,3) bool, gave_up (N,3) bool). cmd equals cmd_in for pawns that are
        not recovering. recovering stays True for a pawn that gave up (it is still outside).
        """
        seen = seen.bool()
        fresh = seen | ~self.has_last               # take the inputs when seen, or when nothing is stored yet
        self.last_xy = torch.where(fresh.unsqueeze(-1), xy, self.last_xy)
        self.last_yaw = torch.where(fresh, yaw, self.last_yaw)
        self.last_v = torch.where(fresh, v, self.last_v)
        self.has_last = torch.ones_like(self.has_last)
        self.t_unseen = torch.where(seen, torch.zeros_like(self.t_unseen), self.t_unseen + dt)
        xy_k, yaw_k, v_k = self.last_xy, self.last_yaw, self.last_v

        # enter: centre beyond the tape (seen), or dead-reckoned beyond it (unseen)
        t_dr = self.t_unseen.clamp(max=DEAD_RECKON_MAX_S)
        est = xy_k + (v_k * t_dr).unsqueeze(-1) * torch.stack([torch.cos(yaw_k), torch.sin(yaw_k)], -1)
        enter = ~self.recovering & (G.dist_outside(est) > 0.0)
        # leave: seen and far enough inside on both axes
        inside = (S.ARENA_HALF_M - xy_k.abs().max(dim=-1).values) >= REJOIN_INSIDE_M
        leave = self.recovering & seen & inside
        was = self.recovering
        self.recovering = (was | enter) & ~leave
        self.stopped = self.stopped & self.recovering & ~enter
        self.lead = torch.where(self.recovering & ~enter, self.lead, torch.zeros_like(self.lead))

        # blind bookkeeping: (re)start the blind procedure on the first unseen tick of a recovery
        blind = self.recovering & ~seen
        start_blind = blind & ((self.t_blind <= 0.0) | enter)
        brake_ticks = torch.ceil(v_k.abs() / (G.BRAKE_PLAN * dt) - 1e-6).long().clamp(0, 8)
        brake_ticks = torch.where(v_k.abs() > 0.1, brake_ticks, torch.zeros_like(brake_ticks))
        self.brake_left = torch.where(start_blind, brake_ticks, self.brake_left)
        self.rp_idx = torch.where(start_blind, torch.zeros_like(self.rp_idx), self.rp_idx)
        self.rp_hold = torch.where(start_blind, torch.zeros_like(self.rp_hold), self.rp_hold)
        self.t_blind = torch.where(start_blind, torch.zeros_like(self.t_blind), self.t_blind)
        self.gave_up = blind & (self.t_blind >= GIVE_UP_S - 0.5 * dt)     # half a tick of slack: t_blind is a float sum

        seen_cmd, stopped_now, lead_now = self._seen_cmd(xy_k, yaw_k, v_k)
        self.stopped = torch.where(self.recovering & seen, stopped_now, self.stopped)
        self.lead = torch.where(self.recovering & seen & stopped_now, lead_now, self.lead)
        blind_cmd = self._blind_cmd(blind & ~self.gave_up)
        blind_cmd = torch.where(self.gave_up.unsqueeze(-1), torch.zeros_like(blind_cmd), blind_cmd)
        rec_cmd = torch.where(seen.unsqueeze(-1), seen_cmd, blind_cmd)
        cmd = torch.where(self.recovering.unsqueeze(-1), rec_cmd, cmd_in)

        self.t_blind = torch.where(blind, self.t_blind + dt, torch.zeros_like(self.t_blind))

        # history: only ticks in which the pawn was seen
        new_cmd = torch.cat([cmd.unsqueeze(2), self.hist_cmd[:, :, :-1]], 2)
        new_v = torch.cat([v_k.unsqueeze(2), self.hist_v[:, :, :-1]], 2)
        self.hist_cmd = torch.where(seen[..., None, None], new_cmd, self.hist_cmd)
        self.hist_v = torch.where(seen[..., None], new_v, self.hist_v)
        self.hist_len = torch.where(seen, (self.hist_len + 1).clamp(max=HISTORY_STEPS), self.hist_len)

        return cmd, self.recovering.clone(), self.gave_up.clone()
