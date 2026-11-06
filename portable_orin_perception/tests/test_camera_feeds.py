"""tools/camera_feeds.py: which cameras a server takes, on which port, and the X-Tags it sends.

The dashboard's corner-camera tag check trusts two things tested here: `serve --secondary`
never opens the vision service's camera (however that camera is named), and the tags in a
part are the ones found in that part's image.

    cd portable_orin_perception && python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import camera_feeds  # noqa: E402
from vision.capture import Frame, LatestFrameSlot  # noqa: E402  (puts hive_perception on the path)
from hive_perception.core.frame_builder import load_marker_map  # noqa: E402

MARKER_MAP = load_marker_map(ROOT / "config" / "marker_map.yaml")


def fake_by_id(tmp_path):
    """Two by-id links to two /dev/videoN stand-ins, like the vision Orin on 2026-11-06."""
    dev, by_id = tmp_path / "dev", tmp_path / "by-id"
    dev.mkdir()
    by_id.mkdir()
    for node in ("video0", "video2"):
        (dev / node).touch()
    c270 = by_id / "usb-046d_C270_HD_WEBCAM_520D0960-video-index0"
    x1 = by_id / "usb-Webcam_X1__Webcam_X1__Webcam_X1_-video-index0"
    c270.symlink_to(dev / "video0")
    x1.symlink_to(dev / "video2")
    return sorted([str(c270), str(x1)]), dev


def test_each_camera_keeps_its_port_whichever_others_are_served(tmp_path):
    cams, dev = fake_by_id(tmp_path)
    c270, x1 = cams
    assert camera_feeds.select_cameras(cams) == [(0, c270), (1, x1)]
    assert camera_feeds.select_cameras(cams, devices=[x1]) == [(1, x1)]
    # named by its /dev node instead of by-id: still the X1's port
    assert camera_feeds.select_cameras(cams, devices=[str(dev / "video2")]) == [(1, str(dev / "video2"))]


def test_secondary_skips_the_vision_services_camera_by_id_or_by_node(tmp_path):
    cams, dev = fake_by_id(tmp_path)
    c270, x1 = cams
    assert camera_feeds.select_cameras(cams, exclude=c270) == [(1, x1)]
    assert camera_feeds.select_cameras(cams, exclude=str(dev / "video0")) == [(1, x1)]
    assert camera_feeds.select_cameras([c270], exclude=c270) == []


def test_roles_use_the_vision_services_names():
    roles = camera_feeds.tag_roles(MARKER_MAP)
    assert roles[MARKER_MAP.evader_id] == "EVADER"
    assert [roles[p] for p in MARKER_MAP.pursuer_ids] == [f"P{i + 1}" for i in range(len(MARKER_MAP.pursuer_ids))]
    assert all(roles[c] == "corner" for c in MARKER_MAP.calibration_corner_ids)


def tag_frame(seq: int, x: int) -> Frame:
    """A 1280x720 MJPEG frame with corner tag 10 (60 px) at column ``x``."""
    img = np.full((720, 1280), 255, np.uint8)
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 10, 60)
    img[300:360, x:x + 60] = tag
    jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].reshape(-1)
    return Frame(seq, time.perf_counter(), jpg, None, "driver")


def test_each_part_carries_the_tags_found_in_its_own_image():
    src = types.SimpleNamespace(slot=LatestFrameSlot())
    info = {"index": 1, "name": "Webcam(X1)", "floor_px": 21.0}
    feed = camera_feeds.TaggedFeed(src, 0, info=info, marker_map=MARKER_MAP, max_fps=30, bind="127.0.0.1").start()
    base = f"http://127.0.0.1:{feed.httpd.server_address[1]}"
    alive = True

    def camera():            # the tag walks right 100 px per frame
        seq = 0
        while alive:
            seq += 1
            src.slot.put(tag_frame(seq, 100 + 100 * (seq % 8)))
            time.sleep(0.03)
    threading.Thread(target=camera, daemon=True).start()
    try:
        with urllib.request.urlopen(base + "/info.json", timeout=5) as r:
            assert json.loads(r.read()) == info
        with urllib.request.urlopen(base + "/stream.mjpg", timeout=5) as r:
            for _ in range(5):
                headers = {}
                while True:
                    line = r.readline().strip()
                    if not line and "content-length" in headers:
                        break
                    if b":" in line:
                        k, v = line.split(b":", 1)
                        headers[k.decode().lower()] = v.strip()
                img = cv2.imdecode(np.frombuffer(r.read(int(headers["content-length"])), np.uint8),
                                   cv2.IMREAD_GRAYSCALE)
                (tid, side, *corners), = json.loads(headers["x-tags"])["d"]
                assert tid == 10 and abs(side - 60) < 3
                # the outline sits on the tag in THIS image: the dark tag spans those columns
                x_tag = np.flatnonzero(img[330] < 128)
                assert abs(min(corners[0::2]) - x_tag.min()) < 3
                assert abs(time.time() * 1000 - int(headers["x-frame-time"])) < 2000
    finally:
        alive = False
        feed.stop()


def corner_scene():
    """Near corner tag 10 (60 px) and far corner tag 11 foreshortened to 30 x 13 px, softened."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = np.full((720, 1280), 200, np.uint8)
    img[300:360, 400:460] = cv2.aruco.generateImageMarker(d, 10, 60)
    img[100:113, 900:930] = cv2.resize(cv2.aruco.generateImageMarker(d, 11, 120), (30, 13),
                                       interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def full_frame(img):
    from hive_perception.core.localizer import Localizer

    return Localizer(MARKER_MAP, image_size=(1280, 720), roi_tracking=False, gating=False).audit(img)


# where corner_scene() put each tag's outer corners (pixel centres; an edge is half a pixel out)
TRUTH = {10: np.array([[399.5, 299.5], [459.5, 299.5], [459.5, 359.5], [399.5, 359.5]]),
         11: np.array([[899.5, 99.5], [929.5, 99.5], [929.5, 112.5], [899.5, 112.5]])}


def test_the_zoomed_look_finds_the_true_centre_even_of_a_foreshortened_tag():
    img = corner_scene()
    full = {d.marker_id: d for d in full_frame(img)}
    tr = camera_feeds.CornerTracker(MARKER_MAP)
    tr.update(img, list(full.values()), 0.0)
    zoomed = {d.marker_id: d for d in tr.update(img, [], 0.1)}       # full frame "missed" everything
    for tid, truth in TRUTH.items():
        # a 2x resize maps pixel centre x to 2x + 0.5, so mapping back by /2 alone is 0.25 px off
        assert np.linalg.norm(zoomed[tid].corners.mean(axis=0) - truth.mean(axis=0)) < 0.05
        assert np.linalg.norm(zoomed[tid].corners - truth, axis=1).max() < 0.5
    # the full frame's corners of the 13 px-tall tag are off by ~1.7 px, its centre by ~0.5 px
    assert np.linalg.norm(full[11].corners.mean(axis=0) - TRUTH[11].mean(axis=0)) > 0.2


def test_a_known_corner_tag_is_found_by_the_zoomed_look_when_the_full_frame_misses_it():
    img = corner_scene()
    tr = camera_feeds.CornerTracker(MARKER_MAP)
    tr.update(img, full_frame(img), 0.0)
    assert set(tr.known) == {10, 11}
    found = {d.marker_id: d for d in tr.update(img, [], 0.1)}
    assert set(found) == {10, 11}
    assert np.allclose(found[11].corners.mean(axis=0), (914.5, 106.0), atol=0.6)


def test_only_corner_tags_are_tracked_and_a_lost_one_is_forgotten():
    from hive_perception.core.localizer import Detection

    img = corner_scene()
    tr = camera_feeds.CornerTracker(MARKER_MAP)
    car = Detection(1, np.array([[600.0, 500], [640, 500], [640, 540], [600, 540]]), 40.0)
    tr.update(img, full_frame(img) + [car], 0.0)
    assert 1 not in tr.known and {10, 11} <= set(tr.known)
    blank = np.full_like(img, 200)
    tr.update(blank, [], 1.0)
    assert set(tr.known) == {10, 11}                                  # still remembered
    tr.update(blank, [], 1.0 + camera_feeds.CornerTracker.FORGET_S + 0.1)
    assert tr.known == {}
