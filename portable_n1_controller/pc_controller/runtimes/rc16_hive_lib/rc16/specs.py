"""Every physical number of the rc16 car, arena and camera, each with its SOURCE. No torch, no Isaac.

The car is the team's MotorDayz 1:16 MD Elite kit (Walmart R8-16CTRKIT1), run with the Corvette Stingray body on
the GRIP tyres (USER 2026-11-17: "the new and final model of the car ... same scale, same aruco tags"). It is a
hobby-style 4WD chassis: RC380 brushed motor, 7.4 V 1000 mAh pack, 17 g digital servo, "4WD differential",
listed top speed 18 mph. The team drives it from the ESP32 through an H-bridge (firmware `act servo`), not the
stock ESC. It replaces the 2WD toy ZR1 (photo 2026-10-29) that every run before 2026-11-17 modelled; the
geometry below is still that toy's 1:16 Corvette geometry until the new chassis is measured.
All four cars (3 pursuers + 1 evader) are identical in shape and size; only colour differs.

SOURCE tags
    REAL     the full-size 2025 C8 Corvette ZR1 (Chevrolet press specifications), divided by 16
    LABEL    the retail listing of the toy: 11.50 x 5.00 x 3.50 in, 0.68 lb, LiPo pack
    PHOTO    the user's 2026-10-01 tape-measure photos of the bare chassis (+/- 5-10 %)
    USER     a statement by the user (2026-10-01): binary throttle and steering, release = brake, "6 mph"
    PROJECT  a value carried over from this repo on purpose (lab-tile friction, arena size, camera rate)
    LISTING  the MD Elite retail listings (Walmart feature list and box art; Shipt/eBay for weight -- weak)
    FIRMWARE ee198-deployment-bundles esp32_receiver.ino on main (2026-11-13 `act servo`)
    ESTIMATE an engineering guess, flagged for measurement -- see MEASURE_ON_THE_TOY

Where REAL/16 and the toy disagree the TOY wins (it is the thing on the floor); the REAL value is kept beside
it so the disagreement is visible. Units: metres, kilograms, radians, seconds. Body frame: x forward, y left,
z up, origin on the floor plane under the mid-point of the wheelbase (so z of a point = its height above
the floor when the car stands on its wheels).
"""

from __future__ import annotations

import math

IN_M = 0.0254
LB_KG = 0.45359237
MPH_MPS = 0.44704
SCALE = 1.0 / 16.0
G_MPS2 = 9.81

# ---------------------------------------------------------------------------------------------------------
# Full-size 2025 Corvette ZR1 (REAL) and its 1:16 scaling. Only used to cross-check the toy values.
# ---------------------------------------------------------------------------------------------------------
REAL_LENGTH_M = 185.9 * IN_M          # REAL 4.722 m
REAL_WIDTH_M = 79.7 * IN_M            # REAL 2.024 m (ZR1/Z06 wide body; the Stingray is 76.1 in = 1.933 m)
REAL_HEIGHT_M = 48.6 * IN_M           # REAL 1.234 m
REAL_WHEELBASE_M = 107.2 * IN_M       # REAL 2.723 m
REAL_TYRE_OD_FRONT_M = 20 * IN_M + 2 * 0.275 * 0.30   # REAL 275/30ZR20 -> 0.673 m
REAL_TYRE_OD_REAR_M = 21 * IN_M + 2 * 0.345 * 0.25    # REAL 345/25ZR21 -> 0.706 m

SCALED = {  # REAL / 16, for the comparison table in the README
    "length_m": REAL_LENGTH_M * SCALE,            # 0.2951
    "width_m": REAL_WIDTH_M * SCALE,              # 0.1265
    "height_m": REAL_HEIGHT_M * SCALE,            # 0.0771
    "wheelbase_m": REAL_WHEELBASE_M * SCALE,      # 0.1702
    "tyre_od_front_m": REAL_TYRE_OD_FRONT_M * SCALE,   # 0.0421
    "tyre_od_rear_m": REAL_TYRE_OD_REAR_M * SCALE,     # 0.0441
}

# ---------------------------------------------------------------------------------------------------------
# The toy, as built (these drive the CAD, the USD and the controllers)
# ---------------------------------------------------------------------------------------------------------
LENGTH_M = 11.50 * IN_M       # LABEL 0.2921 (REAL/16 0.2951: agrees to 1 %)
WIDTH_M = 5.00 * IN_M         # LABEL 0.1270 overall, tyre outer faces (REAL/16 0.1265; PHOTO ~0.13)
HEIGHT_M = 3.50 * IN_M        # LABEL 0.0889 floor to wing top (REAL/16 roof 0.0771 + toy ride height)
WHEELBASE_M = 0.170           # PHOTO 0.166-0.174; REAL/16 0.1702
MASS_KG = 1.5 * LB_KG         # LISTING (weak: Shipt/eBay "1.5 pounds") 0.680 with pack. MEASURE: kitchen scale.
                              # The old 2WD toy was 0.308 (LABEL).

WHEEL_RADIUS_M = 0.027        # PHOTO 52-55 mm across (toy wheels are OVERSIZED: REAL/16 is 0.021-0.022)
WHEEL_WIDTH_M = 0.020         # PHOTO ~2 cm
TRACK_M = WIDTH_M - WHEEL_WIDTH_M   # 0.107 centre-to-centre, DERIVED from LABEL width and PHOTO tyre width
FRONT_AXLE_X_M = +0.5 * WHEELBASE_M  # ESTIMATE: axles symmetric about the body origin
REAR_AXLE_X_M = -0.5 * WHEELBASE_M
GROUND_CLEARANCE_M = 0.010    # ESTIMATE: toy tub above the floor
BODY_OVERHANG_FRONT_M = 0.5 * (LENGTH_M - WHEELBASE_M)   # 0.061 symmetric split, ESTIMATE
BODY_OVERHANG_REAR_M = 0.5 * (LENGTH_M - WHEELBASE_M)
ROOF_HEIGHT_M = 0.075         # ESTIMATE: shell roof above the floor (REAL/16 0.077)
WING_TOP_HEIGHT_M = HEIGHT_M  # LABEL: overall height is the wing
COM_HEIGHT_M = 0.025          # ESTIMATE: battery box and motor low in the tub
COM_X_M = -0.010              # ESTIMATE: slightly rear-biased (motor + gearbox at the back)

# Inertia about the COM (ESTIMATE): a uniform block 0.26 x 0.11 x 0.045 m of the car's mass.
_IB = (0.26, 0.11, 0.045)
INERTIA_KGM2 = (
    MASS_KG * (_IB[1] ** 2 + _IB[2] ** 2) / 12.0,
    MASS_KG * (_IB[0] ** 2 + _IB[2] ** 2) / 12.0,
    MASS_KG * (_IB[0] ** 2 + _IB[1] ** 2) / 12.0,
)
WHEEL_MASS_KG = 0.012         # ESTIMATE per wheel incl. tyre (4 x 12 g = 16 % of the car)
CHASSIS_MASS_KG = MASS_KG - 4 * WHEEL_MASS_KG

# ---------------------------------------------------------------------------------------------------------
# Drivetrain and steering
# ---------------------------------------------------------------------------------------------------------
TOP_SPEED_MPS = 6.0 * MPH_MPS         # USER 2026-11-17: cap near 2.7 m/s. The speed a command of 1.0 reaches; set
                                      # on the car by the firmware `limit` (duty = |throttle| x limit). MEASURE:
                                      # time over 2 m at full command, adjust `limit` until it reads 2.7 m/s.
MOTOR_TOP_SPEED_MPS = 18.0 * MPH_MPS  # LISTING "up to 18 MPH" = 8.05 m/s at full duty on the stock ESC. Through
                                      # the H-bridge it is lower (driver voltage drop); only the cap is trained on.
MOTOR_TIME_CONSTANT_S = 0.40          # ESTIMATE: full-duty speed-up time constant of the uncapped car
DRIVEN_WHEELS = 4                     # LISTING: 4WD
REAR_AXLE_MODE = "open"               # LISTING "4WD differential": each wheel gets an equal share of the torque
# FIRMWARE: zero throttle drives both H-bridge inputs low = COAST (the old toy model braked on release). The
# only brake is a reverse command, which arrives after a 40 ms both-low gap (not modelled: under one command tick).
MOTOR_RELEASE = "coast"               # "coast" (this car) | "brake" (the pre-2026-11-17 shorted-motor model)
COAST_DECEL_MPS2 = 0.8                # ESTIMATE: drivetrain + rolling drag of a 4WD gear train with the motor
                                      # coasting. MEASURE: roll-out distance from a known speed (v^2 / 2d).
# DC motor seen at the wheels: torque = STALL * (u - omega/OMEGA_FREE), u = duty in [-1, 1]. With "coast" the
# torque is applied only while it pushes the way u points (u = 0: none); drag acts always.
ROLLING_RESISTANCE = COAST_DECEL_MPS2 / G_MPS2
WHEEL_OMEGA_TOP_RAD_S = TOP_SPEED_MPS / WHEEL_RADIUS_M                      # 99.3 rad/s at the cap
FREE_SPEED_MPS = MOTOR_TOP_SPEED_MPS
WHEEL_OMEGA_FREE_RAD_S = FREE_SPEED_MPS / WHEEL_RADIUS_M                    # 298 rad/s
AXLE_STALL_FORCE_N = MASS_KG * FREE_SPEED_MPS / MOTOR_TIME_CONSTANT_S       # 13.7 N, all driven wheels together
AXLE_STALL_TORQUE_NM = AXLE_STALL_FORCE_N * WHEEL_RADIUS_M                  # 0.37 N m, all driven wheels

MAX_STEER_RAD = math.radians(30.0)    # SET 2026-11-17 by the user (was a 28 deg estimate). MEASURE: slow full-lock circle diameter.
STEER_RATE_RAD_S = 6.0                # ESTIMATE: centre to lock in ~0.09 s (a 17 g digital servo, LISTING)
MIN_TURN_RADIUS_M = WHEELBASE_M / math.tan(MAX_STEER_RAD)   # DERIVED 0.294 m at the rear axle
CONTROL_HZ = 10.0                     # PROJECT: the ESP32 command rate
BUTTON_SLOT_S = 0.040                 # PROJECT: firmware button slot (button_mod.h)

# Command interface (USER): the stock remote is BINARY on both axes. Two command modes exist in the sim:
#   "buttons":      throttle in {-1, 0, +1} (reverse / release=brake / forward), steer in {-1, 0, +1}
#   "proportional": throttle, steer in [-1, 1]  (if the team later fits an ESC + servo)
COMMAND_MODE = "pwm"             # USER 2026-11-03: ESP32 + motor driver on each car (was "buttons", D17)
# USER 2026-11-03: the button-remote design is OUTDATED -- each car now carries its ESP32 and a motor driver for
# throttle and steering. Mode "pwm" models that; it becomes the default once stage C run c1 is evaluated (D16).
#   throttle  duty = command (|command| < PWM_DEADBAND -> 0), capped at PWM_THROTTLE_LIMIT; duty 0 = brake
#             (H-bridge low side shorted, as the release behaviour already modelled). PWM at kHz averages out
#             within one 150 Hz physics step, so the duty is applied directly.
#   steering  STEER_ACTUATOR "proportional" (a servo, or the steering motor PWM'd against its return spring)
#             or "binary" (the toy's 2-wire steering motor switched full on: full lock or centre).
# DERIVED: the duty at which the capped car settles at TOP_SPEED_MPS against the coast drag (0.37). It stands for
# the firmware `limit`; the real value is whatever makes the real car reach 2.7 m/s (see TOP_SPEED_MPS).
PWM_THROTTLE_LIMIT = (TOP_SPEED_MPS + COAST_DECEL_MPS2 * MOTOR_TIME_CONSTANT_S) / MOTOR_TOP_SPEED_MPS
PWM_DEADBAND = 0.05                # ESTIMATE (the old firmware used 0.05)
STEER_ACTUATOR = "proportional"    # LISTING + FIRMWARE: a positional servo

# ---------------------------------------------------------------------------------------------------------
# Tyres and floor (PROJECT: lab tile from the 2026-09-29 decision; min() combine makes the floor the limit)
# ---------------------------------------------------------------------------------------------------------
FLOOR_STATIC_FRICTION = 0.70          # PROJECT lab_tile (Oeltjen et al. 2025, rubber RC tyre on dry tile)
FLOOR_DYNAMIC_FRICTION = 0.55
TYRE_STATIC_FRICTION = 0.80           # ESTIMATE: rubber-like toy tyre; the floor's 0.70/0.55 is the limit
TYRE_DYNAMIC_FRICTION = 0.70
BODY_STATIC_FRICTION = 0.35           # ESTIMATE: plastic shell on plastic shell
BODY_DYNAMIC_FRICTION = 0.30
LAT_ACCEL_MAX_MPS2 = FLOOR_DYNAMIC_FRICTION * G_MPS2   # 5.4, the sliding limit

# ---------------------------------------------------------------------------------------------------------
# Arena and overhead camera
# ---------------------------------------------------------------------------------------------------------
ARENA_SIDE_M = 8.0 * 12 * IN_M        # PROJECT 2.4384 m operating square; NO walls
ARENA_HALF_M = 0.5 * ARENA_SIDE_M
CAMERA_WIDTH_PX = 1920                # PROJECT
CAMERA_HEIGHT_PX = 1080
CAMERA_FPS = 30.0
CAMERA_MARGIN_M = 0.30                # ESTIMATE: visible margin outside the square on the short (y) axis
CAMERA_HEIGHT_M = 3.00                # ESTIMATE: lens height above the floor. MEASURE on the real mount.
# The lens is chosen so the short image axis covers the square plus margin at that height (DERIVED):
CAMERA_VFOV_RAD = 2.0 * math.atan((ARENA_HALF_M + CAMERA_MARGIN_M) / CAMERA_HEIGHT_M)      # 53.6 deg
CAMERA_HFOV_RAD = 2.0 * math.atan(math.tan(0.5 * CAMERA_VFOV_RAD) * CAMERA_WIDTH_PX / CAMERA_HEIGHT_PX)  # 83.9
CAMERA_M_PER_PX_AT_FLOOR = 2.0 * (ARENA_HALF_M + CAMERA_MARGIN_M) / CAMERA_HEIGHT_PX        # 2.8 mm/px
# Isaac camera: a 20.955 mm wide sensor (Isaac's default horizontal aperture) and the focal length that gives
# the HFOV above.
CAMERA_HORIZONTAL_APERTURE_MM = 20.955
CAMERA_FOCAL_LENGTH_MM = 0.5 * CAMERA_HORIZONTAL_APERTURE_MM / math.tan(0.5 * CAMERA_HFOV_RAD)   # 11.7 mm

# ---------------------------------------------------------------------------------------------------------
# Two corner cameras (USER 2026-11-03: "two cameras, one on opposite corners", because one top camera gives too
# poor an image of the tags). The single overhead camera above is kept only for the legacy colour-marker demo.
# ---------------------------------------------------------------------------------------------------------
# Resolution and rate: PROJECT, AI_Deployment/portable_orin_perception/config/camera.yaml (1280 x 720 @ 30, chosen
# there for the Orin's detector budget; the 1920 x 1080 above was the brief's number, not the hardware's).
CORNER_CAM_WIDTH_PX = 1920      # 1080p: camera_placement_sweep.py shows 720p cannot reach the measured 25 px tag floor
CORNER_CAM_HEIGHT_PX = 1080     # everywhere with a 78 deg lens (74 %); 1080p covers 100 % (worst tag 32 px). The real
                                # rig runs 720p for the Orin CPU budget -- a hardware trade-off for the team (D13).
CORNER_CAM_FPS = 30.0
CORNER_CAM_HFOV_DEG = 78.0      # ESTIMATE: a typical 720p/1080p USB webcam (C920-class ~70-78). MEASURE (camera model).
CORNER_CAM_HEIGHT_M = 1.8       # SWEEP (camera_placement_sweep.py, 78 deg lens): best worst-case tag size. MEASURE.
CORNER_CAM_SETBACK_M = 0.0      # SWEEP: lens directly above the square's corner (a pole or a wall/ceiling bracket)
CORNER_CAM_CORNERS = ((-1, -1), (+1, +1))   # USER: opposite corners. Camera A at (-x, -y), camera B at (+x, +y).
CORNER_CAM_AIM = (0.0, 0.0, 0.0)            # legacy: the arena centre
CORNER_CAM_AIM_OFFSET_M = 0.9   # SWEEP. Each camera aims at the point this far from the centre TOWARD ITS OWN corner (along
                                # the diagonal); > 0 brings the near corner into view (the far half is the other camera's)

# Tags (PROJECT, AI_Deployment .../config/marker_map.yaml and camera.yaml): ArUco DICT_4X4_50, 7 cm, flat on the roof,
# the tag's canonical TOP edge toward the car's NOSE. Evader id 0, pursuers 1, 2, 3.
TAG_DICT = "DICT_4X4_50"
TAG_SIZE_M = 0.07
TAG_BORDER_M = 0.010            # white quiet zone printed around the black square (each side)
TAG_IDS = {"ev": 0, "p0": 1, "p1": 2, "p2": 3}
TAG_X_M = -0.015                # tag centre in the body frame: the flat of the cabin roof
TAG_HEIGHT_M = ROOF_HEIGHT_M + 0.0008

# Arena calibration tags (PROJECT, AI_Deployment .../config/arena_test_6ft.yaml and CALIBRATION.md): ids 10-13, 10 cm
# sheets, centres 0.115 m in from each corner of the tape. Taping rule assumed here (state it on the rig): each tag's
# top edge toward arena +y. Each corner camera sees 3 of the 4 at 1080p / 78 deg (2 at 720p / 60 deg), so every
# camera is calibrated from the CORNERS of the tags it sees (PnP), not from 4 tag centres.
# USER 2026-11-17 (arena tags): the real 720p cameras could barely see the 10 cm sheets, so the corner tags are
# 16 cm (1.6 x the pixels per side) with a 19 mm quiet zone (0.71 module): the largest plate (198 mm) that fits
# A4 at true scale. The quiet zone is NOT shrinkable: at 175 mm with 10 mm (0.34 module) the detector took the
# quiet zone's outer edge for the tag border on dark and mid-grey floors and the simulated calibration landed
# 9.5-9.9 cm off; at >= 0.7 module it is within 0.6 cm on floors from grey 60 to 190. Deployment must match:
# `corner_tag_m: 0.16` in AI_Deployment .../config/marker_map.yaml (its detector tuning bounds tag size by it).
CALIB_TAG_SIZE_M = 0.160
CALIB_TAG_BORDER_M = 0.019
CALIB_TAG_HEIGHT_M = 0.0008
CALIB_TAGS = {10: (-1.104, 1.104), 11: (1.104, 1.104), 12: (1.104, -1.104), 13: (-1.104, -1.104)}


def corner_camera_positions(height: float = CORNER_CAM_HEIGHT_M, setback: float = CORNER_CAM_SETBACK_M):
    """[(x, y, z)] for each corner camera, in the arena frame."""
    d = ARENA_HALF_M + setback / math.sqrt(2.0)
    return [(sx * d, sy * d, height) for sx, sy in CORNER_CAM_CORNERS]


def corner_camera_aims(aim_offset: float = CORNER_CAM_AIM_OFFSET_M):
    """[(x, y, 0)] aim point of each corner camera: `aim_offset` metres from the centre toward its own corner."""
    a = aim_offset / math.sqrt(2.0)
    return [(sx * a, sy * a, 0.0) for sx, sy in CORNER_CAM_CORNERS]


# ---------------------------------------------------------------------------------------------------------
# Capture: the operational definition (see rc16/eval/capture.py for the code and the ceiling)
# ---------------------------------------------------------------------------------------------------------
CAPTURE_RADIUS_M = 0.45               # PROPOSED: ~1.5 car lengths from the evader's centre
CAPTURE_SECTOR_COUNT = 3              # PROPOSED: 3 equal 120-degree sectors around the evader
CAPTURE_SECTORS_NEEDED = 3            #           all three must hold a pursuer inside the radius
CAPTURE_HOLD_S = 0.5                  # PROPOSED: held for 0.5 s (5 control steps), not a single frame
CAPTURE_MAX_GAP_RAD = math.radians(150.0)  # PROPOSED: no angular gap between neighbours above 150 deg

MEASURE_ON_THE_TOY = [
    "overall length, width (tyre outer faces), height to wing top (mm)",
    "wheelbase: front axle centre to rear axle centre (mm)",
    "track: wheel centre to wheel centre, front and rear (mm)",
    "tyre outer diameter and width, front and rear (mm)",
    "mass on a kitchen scale, with battery (g)",
    "top speed: time over a 3 m straight from a rolling start (s)",
    "full-lock circle: diameter of the outer-tyre track at slow speed (m)",
    "does the rear axle turn as one piece (lift the car, spin one rear wheel: does the other turn)?",
    "camera lens height above the floor and its horizontal field of view (m, deg)",
]


def as_dict() -> dict:
    """Every public number as a flat dict (what cad/build_car.py writes next to the meshes)."""
    out = {}
    for k, v in globals().items():
        if k.isupper() and isinstance(v, (int, float, str, tuple)):
            out[k] = list(v) if isinstance(v, tuple) else v
    out["SCALED_REAL_OVER_16"] = SCALED
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(as_dict(), indent=2))
