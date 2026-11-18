"""What the N=3 playbook commander sees and does (rc16/hive/CONTRACT.md, "commander_io.py").

Pure torch, batched over N envs, no Isaac imports. Imports only math, torch, rc16.specs and rc16.hive.geometry
(plays.py is NOT imported: the number of play choices is hard-coded here and must equal plays.N_CHOICES).

Terms (units in brackets)
    choice        index 0..13 into plays.CHOICES: the play (with its parameter) the team is running.
    action        what the commander outputs every decision: 0 = keep the running choice, a in 1..14 = switch to
                  choice a - 1.
    t             episode time [s] of an env, 0 at the reset. Supplied by the caller.
    t_in_play     how long the running choice has been ticked [s] (sum of the dt passed to tick()).
    progress      the playbook's signed "is it working" rate [1/s] of the running choice, roughly in [-1, 1]
                  (plays.Playbook.progress). Supplied by the caller.
    progress_avg  time-weighted mean of progress over the running choice so far [1/s].
    outcome       how a finished choice is filed in the memory, from its progress_avg at the moment it ended:
                  > +PROGRESS_THRESHOLD "progressed", < -PROGRESS_THRESHOLD "broke_down", otherwise "stalled".
    memory        the last MEMORY_LEN finished choices of this episode and their outcomes, newest first.
    blocked_frac  share [0..1] of the evader's 18 escape arcs that are blocked (boxstop.BoxStopTracker).
"""

from __future__ import annotations

import math

import torch

from rc16 import specs as S
from rc16.hive.geometry import World, body_frame, wrap

# --- the contract's constants ---------------------------------------------------------------------------------
N_CHOICES = 14                  # must equal rc16.hive.plays.N_CHOICES (hard-coded: plays.py is not imported;
                                # tests/test_rc16_hive_commander_io.py checks both this and FIRST_CHOICE against it)
N_ACTIONS = N_CHOICES + 1       # 15: 0 = keep, 1..14 = switch to CHOICES[a - 1]
MIN_COMMIT_S = 1.0              # s: a choice made by the commander runs at least this long before a switch counts
MEMORY_LEN = 4                  # finished choices remembered per episode
OUTCOMES = ("progressed", "stalled", "broke_down")
OUTCOME_PROGRESSED, OUTCOME_STALLED, OUTCOME_BROKE_DOWN = 0, 1, 2      # indices into OUTCOMES

# --- constants added here -------------------------------------------------------------------------------------
PROGRESS_THRESHOLD = 0.1        # 1/s: |mean progress| at or below this is "stalled"
FIRST_CHOICE = N_CHOICES - 1    # 13 = REGROUP, the last entry of plays.CHOICES: what every episode starts in
EMPTY = -1                      # value of an empty memory slot (mem_choice and mem_outcome)
COMMIT_EPS_S = 1e-4             # s: slack on the MIN_COMMIT_S comparison (ten float32 ticks of 0.1 s sum to 0.99999)

# --- observation scaling --------------------------------------------------------------------------------------
POS_SCALE_M = S.ARENA_HALF_M    # arena-frame positions: the tape is at +-1
REL_SCALE_M = S.ARENA_SIDE_M    # evader-relative positions and tape distances: one arena side = 1
SPEED_SCALE_MPS = S.TOP_SPEED_MPS
T_IN_PLAY_SCALE_S = 5.0         # t_in_play of 5 s = 1.0 ...
T_IN_PLAY_CLIP = 2.0            # ... and it saturates at 10 s
OBS_CLIP = 3.0                  # every entry is finally clamped to [-OBS_CLIP, OBS_CLIP]

# --- observation layout (column ranges of the (N, OBS_DIM) tensor) ----------------------------------------------
EVADER_DIM = 7                  # x, y, cos yaw, sin yaw, vx, vy (arena frame), seen
PURSUER_DIM = 7                 # rel x, rel y (evader body frame), cos/sin of heading relative to the evader's,
                                # signed forward speed, seen, recovering
MEM_ENTRY_DIM = N_CHOICES + len(OUTCOMES)     # 17: choice one-hot + outcome one-hot


def _layout():
    sizes = (("evader", EVADER_DIM), ("pursuers", 3 * PURSUER_DIM), ("tape", 4), ("blocked", 1),
             ("choice", N_CHOICES), ("t_in_play", 1), ("can_switch", 1), ("progress", 1),
             ("memory", MEMORY_LEN * MEM_ENTRY_DIM))
    out, o = {}, 0
    for name, k in sizes:
        out[name] = slice(o, o + k)
        o += k
    return out, o


OBS_LAYOUT, OBS_DIM = _layout()             # name -> column slice; OBS_DIM = 7 + 21 + 4 + 1 + 14 + 1 + 1 + 1 + 68 = 118
EVADER_SLICE = OBS_LAYOUT["evader"]
PURSUER_SLICE = OBS_LAYOUT["pursuers"]      # nearest pursuer first
TAPE_SLICE = OBS_LAYOUT["tape"]             # +x, -x, +y, -y lines
BLOCKED_SLICE = OBS_LAYOUT["blocked"]
CHOICE_SLICE = OBS_LAYOUT["choice"]
T_IN_PLAY_SLICE = OBS_LAYOUT["t_in_play"]
CAN_SWITCH_SLICE = OBS_LAYOUT["can_switch"]
PROGRESS_SLICE = OBS_LAYOUT["progress"]
MEMORY_SLICE = OBS_LAYOUT["memory"]         # the block blank_memory zeroes


def classify_outcome(progress_avg):
    """Outcome index (long, into OUTCOMES) of a finished choice from its mean progress [1/s].

    > +PROGRESS_THRESHOLD -> 0 progressed; < -PROGRESS_THRESHOLD -> 2 broke_down; otherwise 1 stalled
    (the thresholds themselves count as stalled).
    """
    p = torch.as_tensor(progress_avg)
    out = torch.full(p.shape, OUTCOME_STALLED, dtype=torch.long, device=p.device)
    out = torch.where(p > PROGRESS_THRESHOLD, torch.full_like(out, OUTCOME_PROGRESSED), out)
    out = torch.where(p < -PROGRESS_THRESHOLD, torch.full_like(out, OUTCOME_BROKE_DOWN), out)
    return out


def _per_env(x, n, device, dtype=torch.float32):
    """x as an (n,) tensor on `device`: a scalar is repeated, anything holding n values ((n,), (n,1), ...) is
    flattened (skrl hands actions over as (num_envs, 1))."""
    x = torch.as_tensor(x, device=device).to(dtype)
    return x.reshape(n) if x.numel() == n else x.expand(n)


class CommanderState:
    """Per-env bookkeeping of the commander: the running choice, its clock and progress, and the episode memory.

    Fields (all on `device`)
        choice        (N,) long    running choice 0..13
        t_in_play     (N,) float   seconds the running choice has been ticked
        progress_avg  (N,) float   time-weighted mean progress of the running choice [1/s] (0 before any tick)
        mem_choice    (N,4) long   finished choices, newest first, EMPTY (-1) where nothing is stored
        mem_outcome   (N,4) long   their outcome index into OUTCOMES, EMPTY (-1) where nothing is stored
        committed     (N,) bool    False from the reset until the commander's first switch (see apply)
        t_switch      (N,) float   episode time [s] of the last switch (-inf before the first one)

    Order of calls per decision: apply(action, t) -> then tick(progress, dt) once per control step until the next
    decision -> build_obs(...).

    The field tensors are created once and only ever updated in place (never rebound), so a reference taken to
    one of them stays current, and stepping under torch.inference_mode() / no_grad() while resetting outside it
    can be mixed freely.
    """

    def __init__(self, n: int, device="cpu"):
        self.n = int(n)
        self.device = torch.device(device)
        self.choice = torch.full((self.n,), FIRST_CHOICE, dtype=torch.long, device=self.device)
        self.t_in_play = torch.zeros(self.n, device=self.device)
        self.progress_avg = torch.zeros(self.n, device=self.device)
        self.mem_choice = torch.full((self.n, MEMORY_LEN), EMPTY, dtype=torch.long, device=self.device)
        self.mem_outcome = torch.full((self.n, MEMORY_LEN), EMPTY, dtype=torch.long, device=self.device)
        self.committed = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.t_switch = torch.full((self.n,), -math.inf, device=self.device)
        self._progress_sum = torch.zeros(self.n, device=self.device)     # integral of progress * dt [-]

    def reset(self, ids=None, first_choice=None):
        """Start a new episode in the envs `ids` (None = all): empty memory, clocks at zero.

        ids: None, a long index tensor/list, or an (N,) bool mask.
        first_choice: None = REGROUP (FIRST_CHOICE); else an int or a long tensor (len(ids),) in 0..13.
        The starting choice is a placeholder, not a commander decision: it is not `committed`, so the first
        switch is allowed at once (see apply).
        """
        if ids is None:
            ids = torch.arange(self.n, device=self.device)
        ids = torch.as_tensor(ids, device=self.device)
        if ids.dtype == torch.bool:
            ids = ids.nonzero(as_tuple=False).squeeze(-1)
        ids = ids.long()
        fc = FIRST_CHOICE if first_choice is None else first_choice
        fc = torch.as_tensor(fc, dtype=torch.long, device=self.device)
        if bool(((fc < 0) | (fc >= N_CHOICES)).any()):
            raise ValueError(f"first_choice must be in 0..{N_CHOICES - 1}")
        self.choice[ids] = fc
        self.t_in_play[ids] = 0.0
        self.progress_avg[ids] = 0.0
        self._progress_sum[ids] = 0.0
        self.mem_choice[ids] = EMPTY
        self.mem_outcome[ids] = EMPTY
        self.committed[ids] = False
        self.t_switch[ids] = -math.inf

    def can_switch(self, t=None):
        """(N,) bool: would a switch be honoured now?

        True while the running choice is the reset placeholder (not committed), otherwise once MIN_COMMIT_S has
        passed since the last switch. With t (N,) [s] the test is t - t_switch (what apply uses); without t it
        is t_in_play (what build_obs uses). The two agree when tick() has been called for all the time elapsed.
        """
        if t is None:
            waited = self.t_in_play
        else:
            waited = _per_env(t, self.n, self.device) - self.t_switch
        return (~self.committed) | (waited >= MIN_COMMIT_S - COMMIT_EPS_S)

    def apply(self, action, t):
        """Apply one commander decision. action (N,) long in 0..14, t (N,) episode time [s].

        Returns (choice (N,) long, switched (N,) bool). A switch is made where ALL of these hold:
          - action >= 1 and the choice it names (action - 1) differs from the running one
            (re-selecting the running choice is a keep: its clock and progress are not restarted);
          - can_switch(t): MIN_COMMIT_S has passed since the last switch, or no switch was made yet this episode.
        Anything else is a keep. Where a switch is made, the ended choice is filed in the memory (newest first,
        the oldest entry drops off) with classify_outcome(progress_avg) -- unless it was never ticked
        (t_in_play == 0), in which case there is nothing to classify and the memory is left alone. Then the
        clock and the progress of the new choice start from zero.
        """
        action = _per_env(action, self.n, self.device, torch.long)
        if bool(((action < 0) | (action >= N_ACTIONS)).any()):
            raise ValueError(f"action must be in 0..{N_ACTIONS - 1}")
        t = _per_env(t, self.n, self.device)
        new = action - 1
        switched = (action >= 1) & (new != self.choice) & self.can_switch(t)

        store = switched & (self.t_in_play > 0.0)
        if bool(store.any()):
            shifted_c = torch.cat([self.choice[:, None], self.mem_choice[:, :-1]], dim=1)
            shifted_o = torch.cat([classify_outcome(self.progress_avg)[:, None], self.mem_outcome[:, :-1]], dim=1)
            self.mem_choice.copy_(torch.where(store[:, None], shifted_c, self.mem_choice))
            self.mem_outcome.copy_(torch.where(store[:, None], shifted_o, self.mem_outcome))

        # in-place updates only (see the class docstring)
        self.choice.copy_(torch.where(switched, new, self.choice))
        self.t_in_play.masked_fill_(switched, 0.0)
        self.progress_avg.masked_fill_(switched, 0.0)
        self._progress_sum.masked_fill_(switched, 0.0)
        self.t_switch.copy_(torch.where(switched, t, self.t_switch))
        self.committed.logical_or_(switched)
        return self.choice.clone(), switched

    def tick(self, progress, dt):
        """Advance the running choice by dt [s] (float or (N,)) with its current progress rate (N,) [1/s].

        progress_avg becomes the time-weighted mean of every progress value ticked since the choice started.
        Non-finite progress values are counted as 0.
        """
        progress = torch.nan_to_num(_per_env(progress, self.n, self.device), nan=0.0, posinf=0.0, neginf=0.0)
        dt = _per_env(dt, self.n, self.device)
        self._progress_sum.add_(progress * dt)
        self.t_in_play.add_(dt)
        self.progress_avg.copy_(torch.where(self.t_in_play > 0.0,
                                            self._progress_sum / self.t_in_play.clamp(min=1e-9),
                                            torch.zeros_like(self._progress_sum)))


def _one_hot(idx, k):
    """(...,) long with EMPTY (-1) allowed -> (..., k) float one-hot, all zeros where idx is EMPTY."""
    valid = idx >= 0
    oh = torch.nn.functional.one_hot(idx.clamp(min=0), k).float()
    return oh * valid[..., None].float()


def _lex_order(keys):
    """(N,3) long: the order that sorts each row by keys[0] ascending, ties broken by keys[1], then keys[2], ...

    keys: list of (N,3) tensors. Stable sorts, least significant key first. NaN sorts last.
    """
    order = torch.argsort(keys[-1], dim=-1, stable=True)
    for key in reversed(keys[:-1]):
        order = torch.gather(order, 1, torch.argsort(torch.gather(key, 1, order), dim=-1, stable=True))
    return order


def build_obs(world: World, cs: CommanderState, recovering, blocked_frac, blank_memory: bool = False):
    """The commander's observation, (N, OBS_DIM) float32 on the World's device, every entry finite and in
    [-OBS_CLIP, OBS_CLIP].

    world         geometry.World for this tick
    cs            the CommanderState (read only)
    recovering    (N,3) bool, per pursuer p0..p2: it is in the out-of-bounds recovery subroutine
    blocked_frac  (N,) share of the evader's 18 escape arcs that are blocked, 0..1
    blank_memory  True zeroes exactly the MEMORY_SLICE columns (the "memory matters" ablation)

    Columns (see OBS_LAYOUT / the *_SLICE constants)
      EVADER_SLICE     x, y / ARENA_HALF; cos yaw, sin yaw; vx, vy / TOP_SPEED (arena frame); seen 0/1
      PURSUER_SLICE    3 x 7, pursuers SORTED by centre distance to the evader, nearest first (never by car
                       index, so the observation does not change when p0, p1, p2 are relabelled):
                       position relative to the evader in the evader's body frame (ahead, left) / ARENA_SIDE;
                       cos, sin of (pursuer yaw - evader yaw); signed forward speed / TOP_SPEED; seen; recovering
      TAPE_SLICE       distance of the evader's centre to the +x, -x, +y, -y tape lines / ARENA_SIDE
                       (negative when it is past that line)
      BLOCKED_SLICE    blocked_frac
      CHOICE_SLICE     one-hot(14) of the running choice
      T_IN_PLAY_SLICE  t_in_play / 5 s, saturating at 2.0
      CAN_SWITCH_SLICE 1 where a switch would be honoured now (cs.can_switch()), else 0
      PROGRESS_SLICE   progress_avg of the running choice
      MEMORY_SLICE     4 x (choice one-hot(14) + outcome one-hot(3)), newest first, zeros for an empty slot

    Unseen cars are NOT masked: the World carries whatever the tracker last reported and the seen flag says so.
    Sorting ties (two pursuers at exactly the same distance, as in a hand-built symmetric world) are broken by
    the pursuers' own 7 observation values in column order, never by car index: the observation is the same for
    every relabelling of p0, p1, p2 even then.
    """
    e_xy, e_yaw, e_v = world.e_xy, world.e_yaw, world.e_v
    n = e_xy.shape[0]
    f, dev = e_xy.dtype, e_xy.device

    ce, se = torch.cos(e_yaw), torch.sin(e_yaw)
    evader = torch.stack([e_xy[:, 0] / POS_SCALE_M, e_xy[:, 1] / POS_SCALE_M, ce, se,
                          e_v * ce / SPEED_SCALE_MPS, e_v * se / SPEED_SCALE_MPS, world.e_seen.to(f)], dim=-1)

    rel_w = world.p_xy - e_xy[:, None, :]                                   # (N,3,2) arena frame
    rel_b = body_frame(rel_w, e_yaw[:, None]) / REL_SCALE_M                  # (N,3,2) evader body frame
    dyaw = wrap(world.p_yaw - e_yaw[:, None])
    pursuers = torch.stack([rel_b[..., 0], rel_b[..., 1], torch.cos(dyaw), torch.sin(dyaw),
                            world.p_v / SPEED_SCALE_MPS, world.p_seen.to(f),
                            torch.as_tensor(recovering, device=dev).to(f)], dim=-1)            # (N,3,7)
    # (N,3) nearest first; exact distance ties fall through to the pursuer's own values, never to its car index
    order = _lex_order([rel_w.norm(dim=-1)] + [pursuers[..., k] for k in range(PURSUER_DIM)])
    pursuers = torch.gather(pursuers, 1, order[:, :, None].expand(-1, -1, PURSUER_DIM)).reshape(n, 3 * PURSUER_DIM)

    h = S.ARENA_HALF_M
    tape = torch.stack([h - e_xy[:, 0], h + e_xy[:, 0], h - e_xy[:, 1], h + e_xy[:, 1]], dim=-1) / REL_SCALE_M

    blocked = torch.as_tensor(blocked_frac, device=dev).to(f).reshape(n, 1)
    choice = _one_hot(cs.choice, N_CHOICES).to(dev, f)
    t_in = (cs.t_in_play / T_IN_PLAY_SCALE_S).clamp(max=T_IN_PLAY_CLIP).to(dev, f).reshape(n, 1)
    can = cs.can_switch().to(dev, f).reshape(n, 1)
    prog = cs.progress_avg.to(dev, f).reshape(n, 1)
    memory = torch.cat([_one_hot(cs.mem_choice, N_CHOICES), _one_hot(cs.mem_outcome, len(OUTCOMES))],
                       dim=-1).to(dev, f).reshape(n, MEMORY_LEN * MEM_ENTRY_DIM)
    if blank_memory:
        memory = torch.zeros_like(memory)

    obs = torch.cat([evader, pursuers, tape, blocked, choice, t_in, can, prog, memory], dim=-1)
    obs = torch.nan_to_num(obs, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP).clamp(-OBS_CLIP, OBS_CLIP)
    return obs.to(torch.float32)        # float32 whatever the World's dtype (a numpy-fed World is float64)
