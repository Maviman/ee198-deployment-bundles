"""Generate the printable fiducial sheets in markers/ from config/marker_map.yaml.
Handles whichever family the marker map names — ArUco DICT_4X4_50 or AprilTag
DICT_APRILTAG_36h11. The bundle ships with the output pre-generated — re-run
this only if you change the family, ids, or sizes in the marker map.

Tags are rendered from the FULL family codebook at their real ids, so the sheet
is a genuine tag36h11 that NVIDIA cuAprilTags reads. (The detector's subset
dictionary in core.tag_family is a CPU search-space optimisation only; it never
changes what gets printed.)

Family affects the size you need: 36h11 is 8 modules across vs ArUco 4x4's 6,
so at equal physical size and camera resolution an AprilTag resolves ~1.33x
worse. This tool prints the pixel budget for your arena so that trade is visible
before you commit to a print run.

Output per marker id:
  markers/marker_<id>_<role>.pdf  -- print THIS at 100% scale ("actual size",
                                     no fit-to-page); the marker comes out at
                                     exactly the physical size in its caption.
  markers/marker_<id>_<role>.png  -- raw pixels (300 DPI metadata) if you'd
                                     rather print via an image editor.

After printing, ALWAYS ruler-check one marker's black square before trusting
poses — a scaled printout shifts every measurement in the arena.

Run from the bundle root:  python tools/generate_tags.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "hive_perception"))

from hive_perception.core.frame_builder import load_marker_map  # noqa: E402
from hive_perception.core.tag_family import (  # noqa: E402
    family_spec, max_ground_width_m, tag_px_at,
)

VEHICLE_SIDE_M = 0.07   # fits a 8.9 cm-wide car roof with quiet zone on a small plate
CORNER_SIDE_M = 0.10    # calibration markers sit on the floor/jig, can be bigger
RENDER_PX = 840         # crisp binary render; physical size is set at print time
DPI = 300

# For the pixel-budget report only — keep in step with perception.launch.py.
CAPTURE_WIDTHS = (640, 1280)
ARENA_WIDTH_M = 1.83


def report_pixel_budget(marker_map) -> None:
    """Print what this family + tag size can actually resolve, before printing.

    A print run is slow to redo and the failure mode is silent (tags that
    detect on the bench and drop out under motion blur), so the trade belongs
    in front of the user at generation time.
    """
    spec = family_spec(marker_map.dictionary)
    floor = spec.min_detect_px()
    print(f"\n{marker_map.dictionary}: {spec.data_modules}x{spec.data_modules} data "
          f"+ {spec.border_modules} border ring = {spec.total_modules} modules across")
    print(f"rule-of-thumb detection floor: ~{floor:.0f} px/side\n")
    print(f"{'capture':>9} {'tag px @ ' + str(ARENA_WIDTH_M) + 'm':>18} {'verdict':>10}   max arena width")
    print("-" * 62)
    for width in CAPTURE_WIDTHS:
        px = tag_px_at(VEHICLE_SIDE_M, ARENA_WIDTH_M, width)
        widest = max_ground_width_m(VEHICLE_SIDE_M, width, spec)
        verdict = "OK" if px >= floor else "TOO SMALL"
        print(f"{width:>6}px {px:>15.0f} px {verdict:>10}   {widest:.2f} m")
    print(f"\n(vehicle tags {VEHICLE_SIDE_M * 100:.0f} cm, arena {ARENA_WIDTH_M} m. "
          "Numbers are a rule of thumb —\nconfirm against real arena frames, which "
          "are harder than synthetic ones.)")


def role_of(marker_id: int, marker_map) -> tuple[str, float]:
    """(role name, printed side length in metres).

    Calibration corners are tested FIRST and reserved vehicle ids explicitly,
    so that an id listed only in extra_print_ids cannot fall through to the
    corner branch — that would print a future car's tag at the 10 cm corner
    size and label it 'corner', which is exactly the sheet mix-up this tool is
    supposed to prevent.
    """
    if marker_id in marker_map.calibration_corner_ids:
        return "corner", CORNER_SIDE_M
    if marker_id == marker_map.evader_id:
        return "evader", VEHICLE_SIDE_M
    if marker_id in marker_map.pursuer_ids:
        return f"pursuer{marker_map.pursuer_ids.index(marker_id) + 1}", VEHICLE_SIDE_M
    if marker_id in (marker_map.extra_print_ids or []):
        return f"reserved{marker_id}", VEHICLE_SIDE_M
    raise ValueError(
        f"id {marker_id} has no role in marker_map.yaml — it is not the evader, a "
        "pursuer, a calibration corner, or a reserved print id")


def main() -> None:
    import cv2

    marker_map = load_marker_map(ROOT / "config" / "marker_map.yaml")
    spec = family_spec(marker_map.dictionary)
    # Full codebook, real ids: the printed tag must be a genuine family member.
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.opencv_dict))
    out_dir = ROOT / "markers"
    out_dir.mkdir(exist_ok=True)
    report_pixel_budget(marker_map)
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not available -> skipping PDFs, PNGs only")
    try:
        from PIL import Image
    except ImportError:
        Image = None

    written: set[str] = set()
    for marker_id in marker_map.print_ids:
        role, side_m = role_of(marker_id, marker_map)
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, RENDER_PX)
        stem = f"marker_{marker_id}_{role}"
        written.update({f"{stem}.png", f"{stem}.pdf"})

        side_px_at_dpi = int(round(side_m / 0.0254 * DPI))
        resized = cv2.resize(img, (side_px_at_dpi, side_px_at_dpi),
                             interpolation=cv2.INTER_NEAREST)
        quiet = side_px_at_dpi // 4  # generous white quiet zone for detection
        page = np.full((side_px_at_dpi + 2 * quiet,) * 2, 255, dtype=np.uint8)
        page[quiet:quiet + side_px_at_dpi, quiet:quiet + side_px_at_dpi] = resized
        png_path = out_dir / f"{stem}.png"
        if Image is not None:
            Image.fromarray(page).save(png_path, dpi=(DPI, DPI))
        else:
            cv2.imwrite(str(png_path), page)

        if plt is not None:
            side_in = side_m / 0.0254
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([(8.5 - side_in) / 2 / 8.5, (11 - side_in) / 2 / 11,
                               side_in / 8.5, side_in / 11])
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_axis_off()
            # Keyed off the printed size, not the role NAME: a reserved vehicle
            # id ("reserved2") is still a car tag and must carry the nose
            # instruction, not the corner-placement one.
            note = ("this edge = car NOSE" if side_m == VEHICLE_SIDE_M
                    else "place center at the coordinates in arena_test_6ft.yaml")
            fig.text(0.5, 0.5 + side_in / 11 / 2 + 0.02,
                     f"{marker_map.dictionary}  id {marker_id}  ({role})  — canonical top edge: {note}",
                     ha="center", fontsize=10)
            fig.text(0.5, 0.5 - side_in / 11 / 2 - 0.03,
                     f"print at 100% scale, then VERIFY the black square is "
                     f"{side_m * 100:.1f} cm x {side_m * 100:.1f} cm",
                     ha="center", fontsize=10)
            fig.savefig(out_dir / f"{stem}.pdf")
            plt.close(fig)
        print(f"  {stem}: {side_m * 100:.0f} cm "
              f"({'pdf+png' if plt is not None else 'png only'})")

    # Sheets left over from a previous family or id list are a printing hazard:
    # they look like valid tags and are silently undetectable by the configured
    # detector. Name them explicitly rather than leaving them to be discovered
    # at the printer.
    # Exact filenames, not id prefixes: a role rename (pursuer2 -> reserved2)
    # leaves marker_2_pursuer2.pdf behind, and a prefix test would call that
    # current. It is the stalest kind of sheet — right id, wrong family.
    stale = sorted(p.name for p in out_dir.iterdir()
                   if p.suffix in {".pdf", ".png"} and p.name not in written)
    if stale:
        print(f"\nWARNING: {len(stale)} sheet(s) in markers/ are NOT in the current "
              f"marker map and were not regenerated:")
        for name in stale:
            print(f"    {name}")
        print("  They may be a different family and will NOT be detected. Delete them,")
        print("  or add their ids to config/marker_map.yaml and re-run this tool.")

    print(f"\nwrote {out_dir}. Print PDFs at 100% scale and ruler-check one marker.")


if __name__ == "__main__":
    main()
