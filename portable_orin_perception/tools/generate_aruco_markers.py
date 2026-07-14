"""Generate the printable ArUco marker sheets in markers/ from
config/marker_map.yaml. The bundle ships with the output pre-generated —
re-run this only if you change ids/sizes in the marker map.

Output per marker id:
  markers/marker_<id>_<role>.pdf  -- print THIS at 100% scale ("actual size",
                                     no fit-to-page); the marker comes out at
                                     exactly the physical size in its caption.
  markers/marker_<id>_<role>.png  -- raw pixels (300 DPI metadata) if you'd
                                     rather print via an image editor.

After printing, ALWAYS ruler-check one marker's black square before trusting
poses — a scaled printout shifts every measurement in the arena.

Run from the bundle root:  python tools/generate_aruco_markers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "hive_perception"))

from hive_perception.core.frame_builder import load_marker_map  # noqa: E402

VEHICLE_SIDE_M = 0.07   # fits a 8.9 cm-wide car roof with quiet zone on a small plate
CORNER_SIDE_M = 0.10    # calibration markers sit on the floor/jig, can be bigger
RENDER_PX = 840         # crisp binary render; physical size is set at print time
DPI = 300


def role_of(marker_id: int, marker_map) -> tuple[str, float]:
    if marker_id == marker_map.evader_id:
        return "evader", VEHICLE_SIDE_M
    if marker_id in marker_map.pursuer_ids:
        return f"pursuer{marker_map.pursuer_ids.index(marker_id) + 1}", VEHICLE_SIDE_M
    return "corner", CORNER_SIDE_M


def main() -> None:
    import cv2

    marker_map = load_marker_map(ROOT / "config" / "marker_map.yaml")
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, marker_map.dictionary))
    out_dir = ROOT / "markers"
    out_dir.mkdir(exist_ok=True)

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

    all_ids = marker_map.pursuer_ids + [marker_map.evader_id] + marker_map.calibration_corner_ids
    for marker_id in sorted(all_ids):
        role, side_m = role_of(marker_id, marker_map)
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, RENDER_PX)
        stem = f"marker_{marker_id}_{role}"

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
            note = ("this edge = car NOSE" if role.startswith(("pursuer", "evader"))
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

    print(f"\nwrote {out_dir}. Print PDFs at 100% scale and ruler-check one marker.")


if __name__ == "__main__":
    main()
