import argparse
import csv
import json
import math
import os
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import psutil
import requests

# Cloud URL
CLOUD_HOST = "http://10.0.0.2:8000"

# Content shift detection parameters
DIFF_HISTORY = 90        # frames of frame-diff history to keep for a rolling mean
DIFF_WARMUP = 30         # frames to wait before checking for content shift
DIFF_K = 3.0             # how many standard deviations away from mean single a content shift

# Accumulated-change budget parameters ('budget' gate).
# The adaptive gate above triggers on frame_diff being statistically unusual, so
# in a scene with constant motion nothing is ever unusual and new objects are
# missed. The budget gate instead sums frame_diff since the last inference and
# fires when that total crosses BUDGET - it responds to how much the scene has
# changed, not to how surprising that change is. No baseline, no rolling stats,
# nothing to recalibrate when the scene changes.
DEFAULT_DIFF_BUDGET = 30.0   # summed frame_diff that must accrue before inferring
                             # (set per-video from offline profiling of frame_diff)
DEFAULT_MAX_SKIP = 90        # safety floor: never skip more than this many
                             # consecutive frames, whatever the budget says.
                             # Was 30, which at 25fps caps any skip at 1.2s - short
                             # enough that on quiet footage the floor, not the
                             # budget, decided almost every inference. Measured on
                             # clip4_5_cropped_25.mp4 over a 10s near-static window:
                             # every motion config skipped exactly 31 frames at a
                             # time, i.e. the gate was inert and this line was the
                             # whole policy. 90 is 3.6s at 25fps.

PUSH_EVERY = 15          # metric push cadence

# Real-time pacing: a file decodes as fast as the CPU allows, so set a sleep to pace to the videos fps.
FALLBACK_FPS = 30.0      # used when the container reports no usable fps

# For comparison of methods - `none` infer on every frame, `fixed` infer every N
# frames, `adaptive` on content-shift changes, `budget` once accumulated
# change since the last inference exceeds a budget, and `motion` once tracked
# objects are predicted to have drifted too far
GATE_MODES = ("none", "fixed", "adaptive", "budget", "motion", "flow")
DEFAULT_FRAME_GAP = 5

# --- 'motion' gate ---
# Same shape as 'budget', but what accumulates is predicted object displacement
# instead of pixel change. frame_diff answers "did the image change", which
# lighting, compression noise and wind all trip; track displacement answers "did
# the things we care about move", and its threshold is a staleness bound that
# can be stated: infer before any box has drifted more than this fraction of the
# frame width. Requires box centres, so it only works against a cloud that
# returns them.
# Which per-frame drift statistic the budget spends.
#
# 'max' is the principled choice: association is per object, so a track is lost
# once its OWN predicted box stops overlapping its detection. One fast vehicle
# breaks while the rest of the scene tracks fine, and the mean hides exactly the
# object that fails first.
#
# It does not survive contact with quiet footage. The max is whichever single
# box jitters most, and detector jitter does not fall when the scene stops
# moving. Measured on clip4_5_cropped_25.mp4, comparing a near-static 10s window
# against a 3.2x busier one:
#
#     signal                busy      still    ratio
#     disp_rate    (mean)   0.00141   0.00076   1.86
#     disp_rate_max (max)   0.00253   0.00178   1.42
#     frame_diff            0.51      0.22      2.32
#
# The peak was identical in both windows (0.0365 vs 0.0363) - a noise floor, not
# a measurement. Since disp_accum integrates it every frame, the static window
# still accrued 0.86 of budget per 10s from jitter alone.
#
# That predicts the mean should gate better. It does not. Scored against an
# ungated run on the same 640 frames, at matched inference counts:
#
#     signal  budget  infers  recall  vehicles seen
#     mean    0.010      68    0.848      5/13
#     max     0.030      24    0.802      7/13
#     (fixed gap 9)      72    0.934      9/13
#
# So the better-discriminating statistic loses vehicles faster, and BOTH lose to
# a fixed gap. The statistic is not the bottleneck: disp_rate_* can only be
# measured from tracks that SURVIVED association, and gating is what breaks
# association for the fastest objects - so the gate censors the signal it steers
# by (the same flaw written up under the 'flow' gate below, which exists to
# escape it). Choosing between mean and max is rearranging deck chairs on a
# censored measurement. 'max' stays the default because it empirically keeps
# more vehicles; the flag exists so the comparison can be rerun, not because
# either value is good.
MOTION_SIGNALS = ("mean", "max")
DEFAULT_MOTION_SIGNAL = "max"
# Scale: at IoU 0.5 two same-size boxes can be offset by about a third of a box
# width, and vehicles here are roughly 0.09 wide normalised - hence 0.03.
# NOTE: budgets do not transfer between signals - the mean runs roughly 0.55x
# the max on this footage - nor between videos. Re-profile per clip: on
# clip4_5_cropped_25.mp4 this 0.03 default sits far past the useful range.
DEFAULT_MOTION_BUDGET = 0.03   # normalised frame widths of drift to allow
MOTION_ALPHA = 0.2             # EWMA weight on the measured displacement rate

# --- 'flow' gate: optical flow, a motion signal the gate cannot censor ---
# The 'motion' gate above has a structural flaw. disp_rate_* are measured only
# from tracks that SURVIVED association, and a wider gap is exactly what breaks
# association for the fastest objects - so widening the gap removes the fast
# objects from the measurement used to choose the gap. Measured on this footage:
# the true per-frame drift is 0.0158 frame widths, but a gap of 6 reports 0.0063
# and a gap of 12 reports 0.0015, a 10x underestimate. A controller fed that
# number widens the gap because the gap made the scene look slow.
#
# It also has to EXTRAPOLATE: a gated frame brings no boxes, so the gate spends a
# stale rate (disp_rate_max_ewma) rather than anything observed.
#
# Sparse Lucas-Kanade flow fixes both. It runs on the greyscale frame already
# computed for frame_diff, on EVERY decoded frame, and knows nothing about
# detections or track IDs - so no gate setting can bias it, and every frame in a
# skip run contributes a real measurement instead of a forecast.
#
# Units: magnitudes are divided by frame WIDTH on both axes, giving "fraction of
# frame width travelled per frame" - the unit --motion-budget is already stated
# in. Not identical to disp_rate_*, which inherits ultralytics' xywhn convention
# (cx by width but cy by height, so its vertical component is inflated by the
# aspect ratio). Profile the two columns against each other on an ungated run to
# get the scale factor before spending a flow budget.
# Measured on traffic.mp4 (1080p, 400 frames) across scale x refresh. Cost is
# per decoded frame; ratio is mean flow_moving_p95 over mean disp_rate_max on the
# ungated run, and corr is against that same column:
#   scale 1.00 refresh 30 -> mean 6.9ms  p95  9.3ms  max 46.3ms  ratio 0.75  corr 0.750
#   scale 0.50 refresh 30 -> mean 3.4ms  p95  6.0ms  max 12.8ms  ratio 0.79  corr 0.678
#   scale 0.50 refresh 15 -> mean 4.1ms  p95 11.5ms  max 16.7ms  ratio 0.82  corr 0.686
#   scale 0.25 refresh 30 -> mean 1.7ms  p95  3.0ms  max  6.4ms  ratio 0.83  corr 0.648
# Full resolution correlates best but its re-detect frames spike to 46ms, past the
# 33ms budget of a 30fps feed - it would cause the dropped frames it exists to
# prevent. 0.5 keeps the worst frame at 13ms for 0.07 of correlation.
FLOW_SCALE = 0.5           # downscale before flow. Magnitudes are normalised, so
                           # this buys ~4x the speed without changing the unit
# Feature density. 240 was right while the field only had to produce one scalar
# per frame for the gate - a scalar is well estimated from a sparse set. Once the
# field also WARPS boxes (see below) the requirement changes: each box needs
# FLOW_WARP_MIN_FEATURES of its own, and a small distant vehicle at 240 corners
# over 1080p usually has none. That is a per-box failure a frame-level p95 cannot
# show. Measured over 400 frames at scale 0.5, gap 20, warp coverage = fraction
# of carried boxes with enough features to move:
#   maxCorners 240  minDistance 8 -> 2.5 ms/frame  coverage 49%
#   maxCorners 480  minDistance 6 -> 3.0 ms/frame  coverage 68%
#   maxCorners 800  minDistance 4 -> 3.2 ms/frame  coverage 80%
#   maxCorners 1200 minDistance 3 -> 3.8 ms/frame  coverage 84%
# and what that coverage buys, as carried-frame recall against an ungated run:
#   gap 12: freeze 0.631 | warp@240 0.785 | warp@800 0.883
#   gap 20: freeze 0.465 | warp@240 0.668 | warp@800 0.824
# 800 costs 0.7ms over 240 and converts half the remaining loss. 1200 buys 4
# points of coverage for another 0.6ms and no recall, so the knee is at 800.
FLOW_MAX_FEATURES = 800
# Re-detect corners every N frames: features riding on vehicles leave the frame
# and are permanently lost. goodFeaturesToTrack costs ~4x a tracking step, so
# this interval sets the latency spike RATE - at 15 the p95 nearly doubles (11.5ms
# vs 6.0ms) with no gain in feature count or correlation, because the
# FLOW_MIN_FEATURES floor below is what actually handles attrition. 30 frames is
# one second at 30fps, and only guards against slow drift in what survives.
FLOW_REFRESH = 30
FLOW_MIN_FEATURES = 300    # re-detect early once this few survive. Scaled with
                           # FLOW_MAX_FEATURES so it keeps its meaning ("the set
                           # has lost most of itself"); measured free on this
                           # footage, where attrition never reaches it inside one
                           # refresh interval, and it is there for the scene where
                           # it does
FLOW_MOVING_MIN = 0.0005   # per-frame magnitude above which a feature counts as
                           # moving. The camera is static, so most features sit
                           # on road and buildings; without this split the
                           # quantiles describe the background, not the traffic
FLOW_ERR_MAX = 20.0        # LK tracking error above which a match is discarded
FLOW_ALPHA = 0.2           # EWMA weight, for comparison with disp_rate_max_ewma

# --- global-motion rejection ---
# With a static camera most features sit on road, kerb and buildings, so the
# MEDIAN displacement vector over all features is the motion of the scene as a
# whole: shake, a pan, or the residual jitter of a "fixed" pole mount. Subtract
# it and what is left is object motion. Without it a two-pixel shake reads as
# every feature in the frame moving at once - the exact signature the gate is
# built to fire on - so the gate spends its budget on wind. Applied to the GATE
# statistics only: box warping below deliberately uses the RAW field, because
# after the camera moves a box really is somewhere else in the image.
FLOW_GLOBAL_REJECT = True

# --- carried-box warping ---
# A gated frame serves the previous inference's boxes. Freezing them in place is
# what makes gating expensive: a box the vehicle has left is wrong twice over -
# it misses the vehicle (recall) and reports one where there is none (precision)
# - so the only way to stay accurate is to keep the boxes fresh, i.e. to never
# really skip. That is why an adaptive budget of 0.03 against a true drift of
# 0.0158/frame converges on firing every ~1.9 frames: the machinery is adaptive
# but the thing it is protecting cannot survive a gap.
# Translating each carried box by the median flow of the features inside it
# breaks that link. The gate then only has to fire when the flow field stops
# describing the box - a turn, an occlusion, a new arrival - rather than every
# time anything moves at all. The field is already computed for the gate, so
# this costs one median per box per frame.
# Median, not mean: a box straddling a vehicle and the road behind it holds
# features from two populations, and their mean is a speed nothing in the frame
# is actually travelling at.
FLOW_WARP_MIN_FEATURES = 3   # below this a box has no reliable local field of
                             # its own, so it is left where it is rather than
                             # dragged by two corners of noise

# What it is worth, and what it does to the budget. Simulated over the 935-frame
# ungated reference (its own detections replayed through the gate, so
# inferred_recall is 1.0 by construction and only the CARRIED columns are the
# measurement), scored with track_eval.py:
#   flow budget   inferred    carried recall        carried centre err
#                 /935      freeze -> warp        freeze -> warp
#   0.0246 (dflt)   343      0.935 -> 0.946        0.0051 -> 0.0014
#   0.05            192      0.846 -> 0.930        0.0070 -> 0.0019
#   0.10            104              0.888                  0.0027
#   0.20             56              0.811                  0.0032
# Read it as: warping roughly doubles the budget that holds a given accuracy.
# Warped at 0.10 serves 104 inferences at 0.888 carried recall; frozen needs 192
# to reach 0.846. Note also that the DEFAULT budget of 0.03 (0.0246 in flow
# units) fires 343 times in 935 frames - a gap of 2.7 - which is the "adaptive
# machinery converging on every other frame" that made the whole gate pointless.
# The budget is what to sweep now; 0.05-0.20 is where the interesting trade is,
# not 0.03.
# The centre error is flat from 0.10 onwards, which says the residual is the ~18%
# of boxes with too few features to warp, not accumulated warp drift.
#
# Confirmed live on clip4_5_cropped_25.mp4 (640 frames, scored against an ungated
# run, not replayed). This is the first gate here that beats a fixed gap:
#
#   config                infers  recall  vehicles seen  lag max
#   fixed gap 6              107   0.949      10/13        240ms
#   flow moving_p95 0.05      38   0.946      10/13        390ms
#   flow p95 0.02             66   0.948      11/13       1021ms
#   motion max 0.03           24   0.802       7/13        287ms
#
# flow at 0.05 matches a fixed gap of 6 on both recall and vehicles seen for a
# THIRD of the inference. The reason is visible in the signal itself: over a 10s
# near-static window vs a 3.2x busier one, flow_p95 separated them 13.0x and
# flow_moving_p95 3.7x, where disp_rate_max_ewma managed 1.50x. Track drift
# cannot see stillness because it is measured off surviving tracks; flow can.
# Budgets do NOT carry over from the numbers above - those are traffic.mp4 at
# 1920x1080. On this clip the useful range is 0.02-0.05, and 0.10 already
# collapses coverage to 54%. Re-profile per clip.

# flow_moving_p95 and disp_rate_max ask the same question in the same unit but
# do not agree numerically: disp_rate_max is the single fastest BOX, flow is the
# 95th percentile of moving FEATURES, and flow runs on a downscaled frame. The
# ratio has to be measured, and --flow-budget defaults to it x --motion-budget so
# the two gates allow the same real staleness. Spending one --motion-budget on
# both units made every 'flow' run ~20% tighter than the 'motion' run it was
# compared against - a fifth of the effect the sweep was trying to measure.
#
# 0.82, not the 0.79 in the scale table above: that figure came from the first
# 400 frames, this one from all 935 frames of the ungated reference run
# (ref.csv), mean flow_moving_p95 0.01190 over mean disp_rate_max 0.01446.
# Neither global-motion rejection nor the denser feature set moves it (0.819 ->
# 0.823), which is the expected result on a static camera: the median vector is
# already ~0, so there is nothing for the rejection to remove, and a percentile
# is insensitive to how many features it is taken over.
# Re-measure it for other footage - it is a property of the scene, not of the
# code. A panning camera is exactly where it will change.
FLOW_TO_MOTION_RATIO = 0.82

FEATURE_PARAMS = dict(maxCorners=FLOW_MAX_FEATURES, qualityLevel=0.05,
                      # 4, not 8: minDistance is what actually decides whether a
                      # small box can hold three features. Raising maxCorners
                      # without lowering this just spreads the same count wider.
                      minDistance=4, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Candidate flow statistics, same pattern as DENSITY_METRICS: log them all, pick
# the one that actually tracks true object drift offline, then gate on it.
FLOW_SIGNALS = {
    "moving_p95": "flow_moving_p95",
    "p95": "flow_p95",
    "p90": "flow_p90",
    "max": "flow_max",
    "mean": "flow_mean",
}
DEFAULT_FLOW_SIGNAL = "moving_p95"

# --- density (spatial) signals ---
# Derived on the edge from the boxes the cloud returns. Nothing consumes these
# yet - they are logged so the trigger and thresholds can be chosen offline.
CLOUD_CONF_FLOOR = 0.3    # must match CONF in server_side.py
LOW_CONF_MARGIN = 0.1     # a detection within this of the floor counts as unsure
SMALL_BOX_AREA = 0.002    # normalised area below which a box is "small" - roughly
                          # where 640px inference starts losing distant vehicles
OVERLAP_IOU = 0.1         # IoU above which two boxes count as a crowded pair

DENSITY_ALPHA = 0.1              # EWMA weight, updated per inference not per frame
DENSITY_HYSTERESIS_FRAC = 0.15   # boundary widening, as a fraction of the lo-hi band
DENSITY_DWELL = 30               # frames a state must hold before it may change

# Candidate triggers for "this scene needs more pixels". All are oriented so
# higher = harder; which one actually predicts it is what the sweep answers.
DENSITY_METRICS = {
    "small_boxes": "small_boxes",
    "box_count": "box_count",
    "low_conf": "low_conf_frac",
    "overlap": "overlap_pairs",
}
DEFAULT_DENSITY_METRIC = "small_boxes"
DEFAULT_DENSITY_LO = 2.0
DEFAULT_DENSITY_HI = 6.0

# Wire format for frames sent to the cloud. `png` is lossless - the golden run's
# upper bound on both accuracy and payload. `jpeg` is the lossy baseline, swept
# across --width to trade bandwidth against detection quality.
ENCODINGS = ("png", "jpeg")
DEFAULT_JPEG_QUALITY = 80

# Save results for post-run analysis
CSV_HEADER = [
    "stream_id", "frame", "ts", "storage_io_ms", "preprocess_ms", "round_trip_ms",
    "decode_ms", "inference_ms", "queue_wait_ms", "network_ms", "end_to_end_ms",
    "throughput_fps", "objects_in_frame",
    # Bookkeeping only - NOT an accuracy metric, and not comparable between gate
    # settings. It moves in both directions at once: an ID switch inflates it, a
    # missed vehicle deflates it, so a gate that fragments every track can report
    # the same total as a perfect one. Accuracy comes from track_eval.py scoring
    # a --tracks-out dump against an UNGATED dump of the same frames.
    "unique_total",
    # Vehicles counted across the line, cumulative, as reported by the cloud. THIS
    # is the accuracy-facing count: a crossing needs identity to hold only over
    # the two frames either side of the line, so it survives the ID fanning that
    # makes unique_total above uninterpretable. Compare a gated run's final total
    # against the ungated run's on the same clip.
    "count_in", "count_out", "count_total", "count_unique",
    "edge_cpu", "edge_mem", "payload_kb", "bandwidth_mbps",
    "frame_diff", "content_shift_detected",
    # summed frame_diff since the last inference, as it stood when this frame was
    # gated (so on an inferred row it is the change that paid for the inference).
    # Profile this column offline to pick --budget.
    "change_since_infer",
    "ttff_ms",
    # how far behind the frame's scheduled arrival time we finished it - the
    # real "can the edge keep up?" signal once the loop is paced
    "pacing_lag_ms",
    # --- frame gating ---
    # request_failed marks a frame that WAS selected for inference but got no
    # answer. It is then served carried-forward exactly like a gated frame, so
    # inference_ran is False for it; without this column a dropped request is
    # indistinguishable from a skip the gate chose to make.
    "gate_mode", "inference_ran", "request_failed", "filter_rate",
    # cloud runtime that served this run, so backend comparisons are self-labelling
    "backend",
    # --- cloud concurrency config, reported per response ---
    # which model instance served this stream, and how many CPU threads it had.
    # Lets a stream-count sweep be reconstructed from the edge CSV alone, and
    # confirms each stream really did land on its own worker.
    "worker_id", "infer_threads",
    # --- what actually served this frame ---
    # The full detection config, echoed by the cloud per response, so a results
    # file can be proven comparable to another without trusting its filename.
    # This is not bookkeeping: six earlier runs in this project were silently
    # produced at a different effective resolution and read as a gating effect
    # for weeks. served_imgsz is what the loaded artefact RUNS at, not what was
    # requested of it.
    "served_imgsz", "served_weights", "served_conf", "served_int8",
    # --- density (spatial) signals: how hard this frame is to resolve ---
    # Recomputed on inference and carried forward on gated frames, so every row
    # describes the detections it actually reports.
    "box_count", "small_boxes", "mean_box_area", "min_box_area",
    "mean_conf", "low_conf_frac", "overlap_pairs",
    "density_metric", "density_ewma", "density_state",
    # --- motion (temporal) signals: how fast the scene is moving ---
    # disp_* are normalised frame widths per frame. The 'motion' gate spends
    # disp_rate_max_ewma; the mean is logged alongside for comparison.
    # disp_accum is the predicted drift since the last inference, as it stood
    # when this frame was gated.
    "disp_rate", "disp_rate_max", "disp_rate_ewma", "disp_rate_max_ewma",
    "tracks_matched", "disp_accum",
    # --- optical flow: the same question as disp_*, asked without the model ---
    # Measured on every decoded frame from pixels alone, so unlike disp_* these
    # are unaffected by how heavily the stream is gated. Same unit (normalised
    # frame widths per frame). Compare flow_* against disp_rate_max on an UNGATED
    # run to calibrate, then against a heavily gated run to see the censoring:
    # disp_rate_max collapses, flow_* should not move.
    "flow_features", "flow_mean", "flow_p50", "flow_p90", "flow_p95", "flow_max",
    "flow_moving_frac", "flow_moving_p95",
    # which flow statistic the 'flow' gate is spending, its EWMA, and the total
    # accrued since the last inference as it stood when this frame was gated
    "flow_signal", "flow_ewma", "flow_accum",
    # --- carried-box warping ---
    # How many carried boxes this frame's flow field was dense enough to move,
    # and the mean distance it moved them (frame widths). warp_boxes well below
    # box_count means the field is too sparse for the small boxes - the first
    # thing to check when carried_recall does not improve.
    "warp_boxes", "warp_shift",
]

# Instead of sampling CPU metrics per instance, sampling is done periodically
# psutil.cpu_percent() measures difference since last call, if called too frequently will return 0
host_usage = {"cpu": 0.0, "mem": 0.0}

def sample_host_usage():
    while True:
        host_usage["cpu"] = psutil.cpu_percent()
        host_usage["mem"] = psutil.virtual_memory().percent
        time.sleep(0.5)

# lock terminal print
print_lock = threading.Lock()


def rnd(value, digits):
    """round(), but None passes through - gated frames report null latencies."""
    return None if value is None else round(value, digits)


def box_iou(a, b):
    """IoU of two normalised centre-form boxes."""
    iw = min(a["cx"] + a["w"] / 2, b["cx"] + b["w"] / 2) - max(a["cx"] - a["w"] / 2, b["cx"] - b["w"] / 2)
    ih = min(a["cy"] + a["h"] / 2, b["cy"] + b["h"] / 2) - max(a["cy"] - a["h"] / 2, b["cy"] - b["h"] / 2)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def density_signals(dets):
    """Spatial-difficulty components for one set of detections.

    Areas are normalised, so they compare across resolutions and videos. The
    overlap count is O(n^2) but n is vehicles in one frame.
    """
    boxed = [d for d in dets if "w" in d]
    if not boxed:
        return {"box_count": len(dets), "small_boxes": 0, "mean_box_area": None,
                "min_box_area": None, "mean_conf": None, "low_conf_frac": None,
                "overlap_pairs": 0}

    areas = [d["w"] * d["h"] for d in boxed]
    confs = [d["conf"] for d in boxed]
    overlaps = sum(1 for i in range(len(boxed)) for j in range(i + 1, len(boxed))
                   if box_iou(boxed[i], boxed[j]) > OVERLAP_IOU)
    return {
        "box_count": len(dets),
        "small_boxes": sum(1 for a in areas if a < SMALL_BOX_AREA),
        "mean_box_area": sum(areas) / len(areas),
        "min_box_area": min(areas),
        "mean_conf": sum(confs) / len(confs),
        # detections sitting on the confidence floor: evidence the model is
        # unsure, which box count alone never shows
        "low_conf_frac": sum(1 for c in confs if c <= CLOUD_CONF_FLOOR + LOW_CONF_MARGIN) / len(confs),
        "overlap_pairs": overlaps,
    }


def track_displacement(prev_dets, dets, frames_elapsed):
    """Per-frame centre displacement of tracks present in both frames.

    Matched by track ID, so an object that entered or left contributes nothing -
    there is no displacement to measure for it. Returns
    (mean, max, matched_count) in normalised frame widths per frame, or
    (None, None, 0) if nothing could be matched.
    """
    if not prev_dets or not dets or frames_elapsed <= 0:
        return None, None, 0

    prev_by_id = {d["id"]: d for d in prev_dets if "cx" in d}
    moves = []
    for d in dets:
        prev = prev_by_id.get(d["id"]) if "cx" in d else None
        if prev is not None:
            moves.append(math.hypot(d["cx"] - prev["cx"], d["cy"] - prev["cy"]))
    if not moves:
        return None, None, 0
    return (sum(moves) / len(moves) / frames_elapsed,
            max(moves) / frames_elapsed,
            len(moves))


class FlowMeter:
    """Per-frame scene motion from sparse optical flow.

    Stateful across frames: holds the previous (downscaled) greyscale frame and
    the feature set currently being tracked. `update` is called once per decoded
    frame, BEFORE the gate decides anything, so the signal is identical whatever
    the gate does - which is the entire point of it existing alongside
    track_displacement().

    Returns the distribution of per-feature displacement in normalised frame
    widths per frame, or None on any frame where no measurement was possible
    (the first frame, or after the feature set was lost and re-seeded).
    """

    def __init__(self, scale=FLOW_SCALE, refresh=FLOW_REFRESH):
        self.scale = scale
        self.refresh = refresh
        self.prev = None
        self.points = None
        self.since_detect = 0
        # This frame's RAW displacement field in box coordinates: (positions,
        # deltas), both normalised x/width and y/height to match the xywhn
        # convention cx/cy already use, so warp() can add deltas straight onto a
        # box centre. Cleared at the top of every update, so a frame where flow
        # failed warps nothing instead of re-applying the last field.
        self.field = None

    def _detect(self, small):
        """Re-seed the feature set. Returns nothing; points may end up None."""
        self.points = cv2.goodFeaturesToTrack(small, mask=None, **FEATURE_PARAMS)
        self.since_detect = 0

    def update(self, gray):
        self.field = None
        small = (cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                            interpolation=cv2.INTER_AREA)
                 if self.scale != 1.0 else gray)

        # Nothing to flow from: seed and wait for the next frame. Also the
        # recovery path when a previous frame lost every feature.
        if self.prev is None or self.points is None or len(self.points) < 2:
            self.prev = small
            self._detect(small)
            return None

        nxt, status, err = cv2.calcOpticalFlowPyrLK(self.prev, small, self.points,
                                                    None, **LK_PARAMS)
        stats = None
        if nxt is not None and status is not None:
            keep = status.reshape(-1) == 1
            if err is not None:
                # LK reports a match for features it lost track of; the error
                # term is what separates those from real ones.
                keep &= err.reshape(-1) <= FLOW_ERR_MAX
            new_pts = nxt.reshape(-1, 2)[keep]
            old_pts = self.points.reshape(-1, 2)[keep]
            if len(new_pts):
                delta = new_pts - old_pts
                # Field for warping, in box coordinates: x by width, y by height,
                # the convention cx/cy are in. Kept RAW - a pan moves every box
                # in the image and warp() has to follow it.
                scale_xy = np.array([small.shape[1], small.shape[0]], dtype=float)
                self.field = (old_pts / scale_xy, delta / scale_xy)
                # Gate statistics, in motion coordinates. Subtract the median
                # vector first: with a truly static camera it is ~0 and this
                # changes nothing, and when the camera moves it IS that movement.
                if FLOW_GLOBAL_REJECT:
                    delta = delta - np.median(delta, axis=0)
                # Both axes over WIDTH: mixing width and height normalisation
                # would make the unit depend on the aspect ratio.
                mag = np.hypot(delta[:, 0], delta[:, 1]) / small.shape[1]
                stats = self._summarise(mag)
                self.points = new_pts.reshape(-1, 1, 2)
            else:
                self.points = None

        self.prev = small
        self.since_detect += 1
        # Top up on a schedule and on attrition. Without the schedule the feature
        # set slowly becomes whatever is static enough to survive, which would
        # bias the signal towards zero over a long run.
        if (self.points is None or len(self.points) < FLOW_MIN_FEATURES
                or self.since_detect >= self.refresh):
            self._detect(small)
        return stats

    def warp(self, dets):
        """Advance carried-forward boxes by one frame of measured flow.

        Each box moves by the MEDIAN displacement of the features that were
        inside it, so a box holding features from both a vehicle and the road
        behind it follows the majority rather than their average. A box with
        fewer than FLOW_WARP_MIN_FEATURES of its own is left where it is - the
        old freeze behaviour, but now only for the boxes that have no evidence.

        Returns (boxes, warped_count, mean_shift). Boxes are fresh dicts: the
        caller's last inference set is the baseline the NEXT inference measures
        displacement against, and must stay exactly as it was detected.

        A box whose centre leaves the frame is dropped rather than clamped. The
        vehicle has gone, and holding it against the edge would report a phantom
        for as long as the gate coasts - a precision loss with no upside.
        """
        if self.field is None or not dets:
            return [dict(d) for d in (dets or [])], 0, None

        pts, delta = self.field
        out, warped, shift_sum = [], 0, 0.0
        for d in dets:
            box = dict(d)
            if "cx" not in d or "w" not in d:
                out.append(box)
                continue
            inside = ((np.abs(pts[:, 0] - d["cx"]) <= d["w"] / 2)
                      & (np.abs(pts[:, 1] - d["cy"]) <= d["h"] / 2))
            if int(inside.sum()) >= FLOW_WARP_MIN_FEATURES:
                dx, dy = np.median(delta[inside], axis=0)
                box["cx"] = float(d["cx"] + dx)
                box["cy"] = float(d["cy"] + dy)
                warped += 1
                shift_sum += float(math.hypot(dx, dy))
            if 0.0 <= box["cx"] <= 1.0 and 0.0 <= box["cy"] <= 1.0:
                out.append(box)
        return out, warped, (shift_sum / warped if warped else None)

    @staticmethod
    def _summarise(mag):
        """Candidate motion statistics for one frame's feature displacements."""
        moving = mag[mag >= FLOW_MOVING_MIN]
        return {
            "flow_features": int(mag.size),
            "flow_mean": float(mag.mean()),
            "flow_p50": float(np.percentile(mag, 50)),
            "flow_p90": float(np.percentile(mag, 90)),
            "flow_p95": float(np.percentile(mag, 95)),
            "flow_max": float(mag.max()),
            # How much of the frame is in motion at all - the density-ish
            # companion to the speed statistics, and free from the same pass.
            "flow_moving_frac": float(moving.size) / float(mag.size),
            # p95 restricted to features that are actually moving. With a static
            # camera the plain quantiles are dominated by background; this one
            # describes the vehicles, so it is the default gate signal.
            "flow_moving_p95": float(np.percentile(moving, 95)) if moving.size else 0.0,
        }


def classify_density(value, state, lo, hi, margin):
    """Band `value` into low/med/high, biased towards staying put.

    Boundaries widen by `margin` in whichever direction would leave the current
    state, so a value hovering on one doesn't oscillate.
    """
    if state == "low":
        return "med" if value > lo + margin else "low"
    if state == "high":
        return "med" if value < hi - margin else "high"
    if value < lo - margin:
        return "low"
    if value > hi + margin:
        return "high"
    return "med"


def run_stream(stream_id, video_path, host, gate_mode="none", frame_gap=DEFAULT_FRAME_GAP,
               realtime=True, encoding="png", width=None, jpeg_quality=DEFAULT_JPEG_QUALITY,
               diff_budget=DEFAULT_DIFF_BUDGET, max_skip=DEFAULT_MAX_SKIP,
               motion_budget=DEFAULT_MOTION_BUDGET,
               motion_signal=DEFAULT_MOTION_SIGNAL,
               density_metric=DEFAULT_DENSITY_METRIC,
               density_lo=DEFAULT_DENSITY_LO, density_hi=DEFAULT_DENSITY_HI,
               flow_signal=DEFAULT_FLOW_SIGNAL, flow_scale=FLOW_SCALE, use_flow=True,
               flow_budget=None, warp_carried=True,
               tracks_out=None, max_frames=None):
    """Process a single video stream, sending frames to the cloud for inference, and logging metrics."""

    detect_url = f"{host}/detect"
    metrics_url = f"{host}/metrics"

    # Flow is measured in a different unit from track displacement, so it gets
    # its own budget rather than borrowing --motion-budget (see
    # FLOW_TO_MOTION_RATIO). Default converts, so an unspecified --flow-budget
    # still means "the same staleness --motion-budget asks for".
    if flow_budget is None:
        flow_budget = motion_budget * FLOW_TO_MOTION_RATIO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[stream {stream_id}] ERROR: could not open {video_path}")
        return

    # Pace to the video's own fps, so the loop behaves like a camera feed
    # instead of racing through the file at decode speed.
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (src_fps and src_fps > 0):   # also catches the NaN some containers report
        src_fps = FALLBACK_FPS
        if realtime:
            with print_lock:
                print(f"[stream {stream_id}] no fps in container, pacing at {FALLBACK_FPS:g} fps")
    # 0 disables pacing - the loop then runs at whatever speed decode allows
    frame_interval = (1.0 / src_fps) if realtime else 0.0

    # Resolve --width against the source once, so the resize target is fixed for
    # the run. Height follows the source aspect ratio, and a --width at or above
    # the native width is treated as "send native" - upscaling on the edge would
    # cost bandwidth without adding detail.
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize_to = None
    if width and src_w and src_h and width < src_w:
        resize_to = (width, round(src_h * width / src_w))
    encode_desc = f"{encoding}" + (f"q{jpeg_quality}" if encoding == "jpeg" else "")
    res_desc = f"{resize_to[0]}x{resize_to[1]}" if resize_to else f"native {src_w}x{src_h}"
    with print_lock:
        print(f"[stream {stream_id}] encoding {encode_desc} at {res_desc}")

    # one connection pool per stream, so streams don't queue behind each other
    session = requests.Session()

    seen_ids = set()
    frame_num = 0
    wall_start = time.time()   # for throughput: frames per real second, and TTFF
    ttff_ms = None             # set once, when the first frame's result comes back

    prev_gray = None
    diff_history = deque(maxlen=DIFF_HISTORY)

    # 'budget' gate state: change accrued since the last inference, and how many
    # frames in a row have been skipped (for the MAX_SKIP safety floor).
    diff_accum = 0.0
    skipped_since_infer = 0
    budget_floor_hits = 0      # inferences forced by MAX_SKIP rather than the budget

    # Cache last detections so they're reused for frames not sent to the cloud.
    # last_dets holds them exactly as DETECTED - it is the baseline the next
    # inference measures displacement against, so it must not be warped.
    # carried_dets is the working copy served on gated frames, advanced one frame
    # of flow at a time, and reset to None by every successful inference.
    last_dets = None
    carried_dets = None
    frames_inferred = 0
    frames_skipped = 0
    frames_failed = 0          # selected for inference but the request dropped
    failure_logged = False     # only print the exception text once per stream
    frames_late = 0            # frames that finished after their scheduled slot
    max_lag_ms = 0.0
    backend = None            # reported by the cloud on each /detect response
    worker_id = None          # cloud model instance serving this stream
    infer_threads = None      # CPU threads that instance was given
    served_imgsz = None       # resolution the cloud ACTUALLY ran at
    served_weights = None     # model that produced these detections
    served_conf = None        # confidence floor the cloud applied
    served_int8 = None        # whether it was the quantised build
    conf_floor_checked = False   # only warn about a floor mismatch once
    # Line-crossing totals as last reported by the cloud. None until the first
    # response, and None for the whole run if no line is configured for this video.
    count_in = count_out = count_total = count_unique = None
    video_name = os.path.basename(video_path)   # registry key for this stream's line

    # Density state. Updated per inference, not per frame - a gated frame brings
    # no new detections, so folding it in would just weight the EWMA by how
    # heavily the stream happens to be gated.
    density = density_signals([])
    density_ewma = None
    density_state = "low"
    density_state_since = 0
    density_margin = DENSITY_HYSTERESIS_FRAC * (density_hi - density_lo)

    # Motion state for the 'motion' gate.
    last_infer_frame = None
    disp_rate = None
    disp_rate_max = None
    disp_rate_ewma = None       # mean drift - logged, not spent
    disp_rate_max_ewma = None   # what the gate spends. None until two inferred
                                # frames share a track ID
    tracks_matched = 0
    disp_accum = 0.0

    # Optical flow state for the 'flow' gate. The meter runs regardless of gate
    # mode so every run carries the uncensored signal for offline comparison -
    # that comparison is the whole reason it exists, and it needs the columns
    # present in runs that were not gated on flow.
    flow_meter = FlowMeter(scale=flow_scale) if use_flow else None
    flow = None                 # this frame's statistics, or None
    flow_value = 0.0            # the chosen signal, 0 when unmeasurable
    flow_ewma = None
    flow_accum = 0.0            # measured, not extrapolated: every frame adds a
                                # real observation, which is what makes this gate
                                # different from 'motion'
    flow_key = FLOW_SIGNALS[flow_signal]

    csv_file = open(f"edge_metrics_stream{stream_id}.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)
    csv_file.flush()

    # Per-frame track dump, for scoring a gated run against an ungated one
    # offline (see track_eval.py). Written for EVERY decoded frame, carrying the
    # detections the consumer actually saw - on a gated frame that is the
    # carried-forward set, which is what a downstream analytics consumer is
    # really served and therefore what should be scored.
    tracks_file = None
    if tracks_out:
        tracks_file = open(f"{tracks_out}_stream{stream_id}.jsonl", "w")

    # --- off-loop I/O ---
    # The CSV flush, the JSONL flush, the metrics POST and the terminal print all
    # used to happen between finishing a frame and sleeping until the next one -
    # i.e. inside the pacing window, adding to the very pacing_lag_ms used to
    # judge whether the edge can service the feed in real time. A flush is a
    # syscall, a POST is a network round trip and a Windows console write is
    # neither cheap nor bounded; measuring the loop with them inside it measures
    # the logger.
    #
    # Measured, 300 frames paced at 30fps against a local stub cloud:
    #   flushes + local POST   inline 23.0ms mean lag -> off-loop 22.1ms
    #   with a 60ms /metrics   inline 31.0ms mean, p95 87.5 -> off-loop 24.5, p95 56.8
    #                          and on the frame after a push: 85.9ms -> 30.8ms
    # So the flushes alone are worth about 1ms a frame, and the honest headline is
    # the second row: the dashboard POST is remote in the real deployment
    # (CLOUD_HOST is another host), and inline it was charging one frame in
    # PUSH_EVERY the entire round trip. That is a spike in p95 and max, not in the
    # mean, which is exactly the shape of number a real-time claim rests on.
    #
    # They move to one consumer thread per stream, which keeps
    # per-row flushing (these threads are daemons, so Ctrl-C loses anything
    # buffered) at no cost to the loop. The queue is unbounded on purpose:
    # dropping a metrics row to protect the loop would corrupt the results the
    # loop exists to produce, and the writer only ever trails by its own latency.
    sink_q = queue.Queue()

    def sink():
        # Its own session: requests.Session is not documented thread-safe, and
        # the loop is using the other one for /detect.
        push_session = requests.Session()
        while True:
            item = sink_q.get()
            if item is None:
                break
            row, track_line, push, status = item
            if status is not None:
                with print_lock:
                    print(status)
            if track_line is not None and tracks_file is not None:
                tracks_file.write(track_line)
                tracks_file.flush()
            writer.writerow(row)
            csv_file.flush()
            if push is not None:
                try:
                    push_session.post(metrics_url, json=push, timeout=2)
                except requests.RequestException:
                    pass

    sink_thread = threading.Thread(target=sink, name=f"sink-{stream_id}", daemon=True)
    sink_thread.start()

    while cap.isOpened():
        # Checked before the read, so the count is exact and no frame is decoded
        # only to be thrown away. At the top of an iteration frame_num is the
        # number of frames already processed.
        if max_frames and frame_num >= max_frames:
            with print_lock:
                print(f"[stream {stream_id}] Reached --max-frames {max_frames}.")
            break

        # read frame from disk and time it
        io0 = time.time()
        success, frame = cap.read()
        storage_io_ms = (time.time() - io0) * 1000  
        if not success:
            with print_lock:
                print(f"[stream {stream_id}] Playback complete.")
            break
        frame_num += 1

        # Edge-side analysis of the raw frame: content shift, then optical flow.
        # Both run on every decoded frame and both are timed together, because
        # together they are what a gated frame costs - end_to_end_ms for a skipped
        # frame is this plus the disk read, so leaving flow out would make the
        # real-time headroom look better than it is.
        shift0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = None
        content_shift_detected = False
        if prev_gray is not None:
            frame_diff = float(cv2.absdiff(gray, prev_gray).mean())
            if len(diff_history) >= DIFF_WARMUP:
                baseline_mean = sum(diff_history) / len(diff_history)
                variance = sum((d - baseline_mean) ** 2 for d in diff_history) / len(diff_history)
                baseline_std = variance ** 0.5
                if abs(frame_diff - baseline_mean) > DIFF_K * baseline_std:
                    content_shift_detected = True
            diff_history.append(frame_diff)
        prev_gray = gray

        # Optical flow, before the gate and independent of it. flow is None on
        # frames where no measurement was possible (first frame, or a re-seed),
        # and those contribute nothing rather than a misleading zero.
        if flow_meter is not None:
            flow = flow_meter.update(gray)
            if flow is not None:
                flow_value = flow[flow_key]
                flow_ewma = (flow_value if flow_ewma is None else
                             FLOW_ALPHA * flow_value + (1 - FLOW_ALPHA) * flow_ewma)
                flow_accum += flow_value
        shift_ms = (time.time() - shift0) * 1000

        # Accumulated change since the last inference. The first frame has no
        # predecessor, so frame_diff is None and contributes nothing.
        diff_accum += frame_diff or 0.0
        # Snapshot before the gate resets it, so the CSV row for an inferred frame
        # carries the change that actually triggered it.
        change_since_infer = diff_accum

        # Predicted drift since the last inference, from whichever statistic the
        # gate is set to spend (see MOTION_SIGNALS). Unlike diff_accum this is a
        # forecast, not a measurement - a gated frame has no fresh boxes to
        # measure from, so the last known rate is extrapolated. Both EWMAs are
        # always maintained and logged; only the spent one is selected here.
        motion_rate_ewma = (disp_rate_max_ewma if motion_signal == "max"
                            else disp_rate_ewma)
        disp_accum += motion_rate_ewma or 0.0
        motion_since_infer = disp_accum

        # Flow accrued since the last inference. Snapshotted before the gate can
        # reset it, so an inferred row carries the drift that actually paid for
        # it. Unlike disp_accum this is a sum of measurements, not of forecasts.
        flow_since_infer = flow_accum

        # Decide whether to run inference on this frame based on gating mode defined
        forced_by_floor = False
        request_failed = False
        warp_boxes = 0
        warp_shift = None
        if last_dets is None or gate_mode == "none":
            # last_dets is None covers the first frame of the stream (and any frame
            # after a failed request): a stream always starts with an inference.
            run_inference = True
        elif gate_mode == "fixed":
            run_inference = (frame_num % frame_gap == 0)
        elif gate_mode == "budget":
            # Fire when enough change has accrued, or when the safety floor says
            # we've coasted on stale detections for long enough. The floor bounds
            # the worst-case miss independently of how good the diff signal is.
            forced_by_floor = skipped_since_infer >= max_skip
            run_inference = diff_accum >= diff_budget or forced_by_floor
        elif gate_mode == "motion":
            forced_by_floor = skipped_since_infer >= max_skip
            if motion_rate_ewma is None:
                # No rate yet: it takes two inferred frames sharing a track for
                # one to exist, so infer back to back until they do - but only
                # while there is a track to wait for. With nothing detected there
                # is nothing to bootstrap from, so coast on the floor instead of
                # inferring every frame at an empty camera. Costs one floor
                # interval at startup, while the tracker confirms its first
                # tracks and reports nothing.
                run_inference = bool(last_dets) or forced_by_floor
            else:
                run_inference = disp_accum >= motion_budget or forced_by_floor
        elif gate_mode == "flow":
            # Same staleness contract as 'motion' - infer before the scene has
            # moved further than the budget - but spending observed flow instead
            # of an extrapolated track rate. No bootstrap case: flow exists from
            # the second frame and needs neither a detection nor a track, so
            # there is nothing to wait for and nothing to warm up.
            forced_by_floor = skipped_since_infer >= max_skip
            # flow_accum already includes this frame's motion, and this frame is
            # about to be inferred, so it is served fresh - the drift that was
            # actually SERVED stale is the previous frame's total, which was below
            # budget. The contract therefore already holds exactly, and the
            # flow_accum logged on an inferred row necessarily exceeds the budget
            # by construction (it is the value that tripped it). Do not "fix"
            # that gap by firing a frame early: it just tightens the budget by
            # one frame's flow, which --flow-budget expresses more clearly.
            # Spends flow_budget, not motion_budget: the units differ by ~0.79
            # (see FLOW_TO_MOTION_RATIO), so sharing one number silently gave
            # this gate a tighter budget than the 'motion' gate it is compared to.
            run_inference = flow_accum >= flow_budget or forced_by_floor
        else:  # adaptive
            run_inference = content_shift_detected

        if run_inference:
            # Edge preprocessing - resize and encode, image already in gray scale
            prep0 = time.time()
            # Downscale first when asked, so the encoder only works on the pixels
            # that go on the wire.
            to_encode = cv2.resize(frame, resize_to) if resize_to else frame
            if encoding == "jpeg":
                ok, buf = cv2.imencode(".jpg", to_encode,
                                       [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                ok, buf = cv2.imencode(".png", to_encode)   # lossless, no quality knob
            preprocess_ms = (time.time() - prep0) * 1000  # preprocessing duration
            payload_kb = len(buf) / 1024  # size of compressed image

            # Send to cloud for inference and measure round trip time
            try:
                rt0 = time.time()
                # Post - stream id keeps the cloud's tracker state separate per video
                # `frame` and `video` are what let the cloud count line crossings:
                # the video selects this stream's line from its registry, and the
                # frame number is the counter's clock. It must be the edge's own
                # count, not the cloud's tally of requests - under gating those
                # diverge, and the counter's cooldown is in video frames.
                resp = session.post(detect_url, data=buf.tobytes(),
                                    params={"stream": stream_id,
                                            "frame": frame_num,
                                            "video": video_name},
                                    headers={"Content-Type": "application/octet-stream"},
                                    timeout=15)
                round_trip_ms = (time.time() - rt0) * 1000
                data = resp.json()
                # Get inference times for metrics Post
                dets = data["detections"]
                decode_ms = data.get("decode_ms", 0)
                inference_ms = data.get("inference_ms", 0)
                queue_wait_ms = data.get("queue_wait_ms", 0)
                backend = data.get("backend")
                worker_id = data.get("worker_id")
                infer_threads = data.get("infer_threads")
                served_imgsz = data.get("served_imgsz")
                served_weights = data.get("served_weights")
                served_conf = data.get("served_conf")
                served_int8 = data.get("served_int8")
                # Cumulative line crossings, counted on the cloud because that is
                # where detections exist. Carried forward on gated frames rather
                # than nulled: the total has not changed, and a gap in the column
                # would read as "counting stopped".
                count_in = data.get("count_in", count_in)
                count_out = data.get("count_out", count_out)
                count_total = data.get("count_total", count_total)
                count_unique = data.get("count_unique", count_unique)
                # low_conf_frac is computed here against CLOUD_CONF_FLOOR, so a
                # cloud serving a different floor makes that column mean something
                # else. Warn once rather than silently logging incomparable values.
                if (served_conf is not None and not conf_floor_checked):
                    conf_floor_checked = True
                    if abs(served_conf - CLOUD_CONF_FLOOR) > 1e-6:
                        with print_lock:
                            print(f"[stream {stream_id}] WARNING: cloud serves "
                                  f"conf={served_conf}, this client computes "
                                  f"low_conf_frac against {CLOUD_CONF_FLOOR}. Update "
                                  f"CLOUD_CONF_FLOOR to compare against earlier runs.")
            except requests.RequestException as e:
                # Deliberately NOT a `continue`. That did two things:
                #  - skipped the CSV row, so the drop left no trace at all. The
                #    frame counter still advanced, so the run reported N frames
                #    processed when some of them were never served.
                #  - skipped the pacing sleep at the bottom of the loop. A fast
                #    failure (connection refused returns in milliseconds) then let
                #    the loop decode the next frame immediately, and a run of them
                #    consumed the video at decode speed while pacing_lag_ms sat
                #    near zero - the loop was no longer simulating a live feed at
                #    all, and the metric that should have said so read fine.
                # The frame is served like a gated one instead: carried boxes, a
                # row of its own, and the accumulators left intact so the next
                # frame retries immediately.
                request_failed = True
                frames_failed += 1
                # The wait really happened and is the reason this frame is late,
                # so it is kept rather than nulled. rt0 is set before the post.
                round_trip_ms = (time.time() - rt0) * 1000
                # Printed once. Every later drop shows as REQUEST FAILED in the
                # per-frame line and as request_failed in the CSV, and a console
                # write per failed frame is itself time inside the pacing window.
                if not failure_logged:
                    failure_logged = True
                    with print_lock:
                        print(f"[stream {stream_id}] Request failed ({e}). Further "
                              f"drops are reported per frame and in request_failed.")

        if run_inference and not request_failed:
            # Displacement needs the previous inferred frame, so it is measured
            # before last_dets is replaced.
            if last_infer_frame is not None:
                disp_rate, disp_rate_max, tracks_matched = track_displacement(
                    last_dets, dets, frame_num - last_infer_frame)
                if disp_rate is None and not dets and disp_rate_max_ewma is not None:
                    # Empty scene: nothing on screen can go stale, so the rate is
                    # genuinely zero rather than unmeasurable.
                    # Only once a real rate exists, though - the tracker returns
                    # no detections for its first few frames while it confirms
                    # tracks, and folding those zeros in would pin the EWMA at 0
                    # before it had ever measured anything.
                    disp_rate = disp_rate_max = 0.0
                if disp_rate is not None:
                    disp_rate_ewma = (disp_rate if disp_rate_ewma is None else
                                      MOTION_ALPHA * disp_rate + (1 - MOTION_ALPHA) * disp_rate_ewma)
                    disp_rate_max_ewma = (disp_rate_max if disp_rate_max_ewma is None else
                                          MOTION_ALPHA * disp_rate_max
                                          + (1 - MOTION_ALPHA) * disp_rate_max_ewma)

            density = density_signals(dets)
            metric_value = density[DENSITY_METRICS[density_metric]] or 0.0
            density_ewma = (metric_value if density_ewma is None else
                            DENSITY_ALPHA * metric_value + (1 - DENSITY_ALPHA) * density_ewma)
            # Hysteresis picks the candidate state, dwell decides whether it may
            # take effect yet. Both are needed: hysteresis stops a value hovering
            # on a boundary from oscillating, dwell stops a brief excursion well
            # past one from doing the same.
            candidate = classify_density(density_ewma, density_state,
                                         density_lo, density_hi, density_margin)
            if candidate != density_state and frame_num - density_state_since >= DENSITY_DWELL:
                density_state = candidate
                density_state_since = frame_num

            last_dets = dets
            # Drop the warped working copy: the next gated frame re-seeds it from
            # these fresh boxes, so warp error never accumulates across an
            # inference.
            carried_dets = None
            last_infer_frame = frame_num
            frames_inferred += 1
            # Spend the accumulated change only once the frame has actually been
            # served - a failed request 'continue's above with the total intact.
            diff_accum = 0.0
            disp_accum = 0.0
            flow_accum = 0.0
            skipped_since_infer = 0
            if forced_by_floor:
                budget_floor_hits += 1

            # network time = round trip minus cloud-side decode + inference + lock wait
            network_ms = round_trip_ms - decode_ms - inference_ms - queue_wait_ms

            # End-to-end latency: disk read, edge analysis, encode, cloud round
            # trip. shift_ms is included HERE as well as on gated frames because
            # it is paid on every decoded frame - frame diff and optical flow run
            # before the gate. Leaving it out reported the ungated baseline as
            # having no edge-analysis cost at all while every gated run paid it in
            # full, so the baseline was flattered by exactly the term that makes
            # gating look expensive.
            end_to_end_ms = shift_ms + storage_io_ms + preprocess_ms + round_trip_ms

            # Mbps - bits / time (s) / 1e6 for megabits
            bandwidth_mbps = (len(buf) * 8 / (network_ms / 1000) / 1e6) if network_ms > 0 else 0

            # time to first frame: only set once, on the first successful result
            if ttff_ms is None:
                ttff_ms = (time.time() - wall_start) * 1000
                with print_lock:
                    print(f">>> [stream {stream_id}] Time to first frame: {ttff_ms:.0f}ms")
        else:
            # No fresh detections for this frame - either the gate skipped it, or
            # the request for it failed. Both serve carried-forward boxes and both
            # are 'carried' frames to the scorer; the difference is that a failure
            # already paid for the encode and the wait, and those costs are kept
            # rather than nulled because they are why the frame was late.
            #
            # The boxes are not served frozen. Each gated frame advances them by
            # one frame of MEASURED flow, so a skip run degrades at the rate the
            # flow field is wrong by, not at the rate the traffic moves. This is
            # what lets a budget buy a real gap: a frozen box at 0.0158 drift per
            # frame exhausts a 0.03 budget in under two frames, which is how the
            # adaptive gate ended up behaving like 'fixed' with a gap of 2.
            if carried_dets is None:
                carried_dets = [dict(d) for d in (last_dets or [])]
            if warp_carried and flow_meter is not None:
                carried_dets, warp_boxes, warp_shift = flow_meter.warp(carried_dets)
            dets = carried_dets
            skipped_since_infer += 1
            if request_failed:
                # preprocess_ms, payload_kb and round_trip_ms hold this frame's
                # real measurements; leave them. Not counted as skipped either -
                # filter_rate answers "what did the gate keep from the model",
                # and a dropped request is not a gating decision.
                pass
            else:
                frames_skipped += 1
                preprocess_ms = None
                payload_kb = None
                round_trip_ms = None
            decode_ms = None
            inference_ms = None
            queue_wait_ms = None
            network_ms = None
            bandwidth_mbps = None
            # Disk read + edge analysis, plus the encode and the dropped wait on
            # a failed frame. Both terms are None on a genuinely gated frame.
            end_to_end_ms = (shift_ms + storage_io_ms
                             + (preprocess_ms or 0.0) + (round_trip_ms or 0.0))

        # frames completed per second so far
        throughput_fps = frame_num / (time.time() - wall_start)

        # How late this frame is against its slot in the stream's schedule.
        # Frame N is due at wall_start + (N-1) * frame_interval; a lag that
        # climbs means the edge can't service the feed in real time.
        if frame_interval:
            pacing_lag_ms = (time.time() - wall_start - (frame_num - 1) * frame_interval) * 1000
            if pacing_lag_ms > frame_interval * 1000:
                frames_late += 1
            max_lag_ms = max(max_lag_ms, pacing_lag_ms)
        else:
            pacing_lag_ms = None   # unpaced: there's no schedule to be late against

        # Count labels
        labels = [d["label"] for d in dets]
        object_counts = {l: labels.count(l) for l in set(labels)}
        frame_ids = sorted(d["id"] for d in dets)
        seen_ids.update(frame_ids)

        # edge resource metrics 
        edge_cpu = host_usage["cpu"]
        edge_mem = host_usage["mem"]

        # --- assemble the metrics record ---
        # Assembled for EVERY decoded frame, gated or not, so the push cadence
        # below is unaffected by the gate mode. rnd() keeps the null-on-reuse
        record = {
            "stream_id": stream_id,
            "video": video_path,
            "frame": frame_num,
            "ts": time.time(),
            "storage_io_ms": round(storage_io_ms, 1),
            "preprocess_ms": rnd(preprocess_ms, 1),
            "round_trip_ms": rnd(round_trip_ms, 1),
            "decode_ms": rnd(decode_ms, 1),
            "inference_ms": rnd(inference_ms, 1),
            "queue_wait_ms": rnd(queue_wait_ms, 1),
            "network_ms": rnd(network_ms, 1),
            "end_to_end_ms": rnd(end_to_end_ms, 1),
            "throughput_fps": round(throughput_fps, 1),
            "objects_in_frame": len(dets),
            "counts": object_counts,
            "unique_total": len(seen_ids),
            "count_in": count_in,
            "count_out": count_out,
            "count_total": count_total,
            "count_unique": count_unique,
            "edge_cpu": edge_cpu,
            "edge_mem": edge_mem,
            "payload_kb": rnd(payload_kb, 1),
            "bandwidth_mbps": rnd(bandwidth_mbps, 2),
            "frame_diff": rnd(frame_diff, 2),
            "content_shift_detected": content_shift_detected,
            # change accrued since the previous inference (pre-reset), so the
            # budget can be profiled offline from any run, whatever the gate mode
            "change_since_infer": round(change_since_infer, 2),
            # rnd(), not round(): frame 1's request can fail, and then there is
            # no first frame to have timed yet. Only reachable now that a failed
            # request writes its row instead of skipping the loop body.
            "ttff_ms": rnd(ttff_ms, 1) if frame_num == 1 else None,
            "pacing_lag_ms": rnd(pacing_lag_ms, 1),
            # fps the loop is pacing to, so the dashboard can show achieved
            # rate against the target instead of a bare throughput number
            "target_fps": round(src_fps, 2) if frame_interval else None,
            # frame gating. inference_ran is what the frame was actually SERVED
            # from, not what the gate wanted: a frame whose request dropped is
            # served carried-forward, so it reads False and request_failed True.
            "gate_mode": gate_mode,
            "inference_ran": run_inference and not request_failed,
            "request_failed": request_failed,
            # fraction of decoded frames that never reached the model so far
            "filter_rate": round(frames_skipped / frame_num, 3),
            "backend": backend,
            "worker_id": worker_id,
            "infer_threads": infer_threads,
            "served_imgsz": served_imgsz,
            "served_weights": served_weights,
            "served_conf": served_conf,
            "served_int8": served_int8,
            # density signals - logged, not yet consumed
            "box_count": density["box_count"],
            "small_boxes": density["small_boxes"],
            "mean_box_area": rnd(density["mean_box_area"], 5),
            "min_box_area": rnd(density["min_box_area"], 5),
            "mean_conf": rnd(density["mean_conf"], 3),
            "low_conf_frac": rnd(density["low_conf_frac"], 3),
            "overlap_pairs": density["overlap_pairs"],
            "density_metric": density_metric,
            "density_ewma": rnd(density_ewma, 3),
            "density_state": density_state,
            # motion signals - disp_rate* are null until two inferred frames share a track
            "disp_rate": rnd(disp_rate, 5),
            "disp_rate_max": rnd(disp_rate_max, 5),
            "disp_rate_ewma": rnd(disp_rate_ewma, 5),
            "disp_rate_max_ewma": rnd(disp_rate_max_ewma, 5),
            "tracks_matched": tracks_matched,
            "disp_accum": round(motion_since_infer, 5),
            # flow signals - null on frames where flow was not measurable, so a
            # re-seed reads as "no data" rather than as a stationary scene
            "flow_features": flow["flow_features"] if flow else None,
            "flow_mean": rnd(flow["flow_mean"], 6) if flow else None,
            "flow_p50": rnd(flow["flow_p50"], 6) if flow else None,
            "flow_p90": rnd(flow["flow_p90"], 6) if flow else None,
            "flow_p95": rnd(flow["flow_p95"], 6) if flow else None,
            "flow_max": rnd(flow["flow_max"], 6) if flow else None,
            "flow_moving_frac": rnd(flow["flow_moving_frac"], 4) if flow else None,
            "flow_moving_p95": rnd(flow["flow_moving_p95"], 6) if flow else None,
            "flow_signal": flow_signal if flow_meter is not None else None,
            "flow_ewma": rnd(flow_ewma, 6),
            "flow_accum": round(flow_since_infer, 6),
            # carried-box warping - 0/None on an inferred frame, since fresh
            # boxes have nothing to warp forward
            "warp_boxes": warp_boxes,
            "warp_shift": rnd(warp_shift, 6),
        }

        # Print to terminal. IDs in the frame are listed, not just counted, so a
        # gate that drops an object is visible as an ID that never appears.
        ids_desc = ",".join(str(i) for i in frame_ids) if frame_ids else "-"
        # The 'motion' gate spends drift, not pixel change, so show what it acted on
        if gate_mode == "motion":
            accrued = f"drift since infer: {motion_since_infer:.3f}"
        elif gate_mode == "flow":
            accrued = f"flow since infer: {flow_since_infer:.3f}"
        else:
            accrued = f"change since infer: {change_since_infer:.1f}"
        if request_failed:
            state = "REQUEST FAILED"
        elif run_inference:
            state = "**Inferred**" + (" [floor]" if forced_by_floor else "")
        else:
            state = "Not inferred" + (f" [warp {warp_boxes}]" if warp_boxes else "")
        # '-' rather than 0 when the cloud has no line for this video: no line and
        # nothing-crossed-yet are different states, and only one of them is a
        # configuration mistake worth noticing mid-run.
        crossed_desc = count_total if count_total is not None else "-"
        status_line = (f"Stream: {stream_id} Frame:{frame_num} | {state} | "
                       f"End-to-End: {end_to_end_ms:.0f}ms "
                       f"| {throughput_fps:.1f} FPS | {accrued} | density: {density_state} "
                       f"| IDs: [{ids_desc}] | ids seen: {len(seen_ids)} "
                       f"| crossed: {crossed_desc}")

        # Hand the frame's I/O to the sink thread and get straight to the pacing
        # sleep. Everything below this point used to run inside the pacing window.
        # The track dump's `inferred` flag is what lets the scorer separate the
        # accuracy of a fresh inference from the accuracy of carried-forward state
        # - two different failures that objects_in_frame blends into one - so it
        # follows inference_ran, not the gate's intent.
        sink_q.put((
            [record[k] for k in CSV_HEADER],
            (json.dumps({
                "frame": frame_num,
                "inferred": bool(run_inference and not request_failed),
                "dets": dets or [],
            }) + "\n") if tracks_file is not None else None,
            record if frame_num % PUSH_EVERY == 0 else None,
            status_line,
        ))

        # If frame processing finishes early, wait for the next frames slot rather than racing ahead infront of real time.
        # Sleeping against an absolute deadline (rather than a fixed sleep per
        # frame) means processing time is absorbed by the wait instead of
        # accumulating as drift.
        if frame_interval:
            next_due = wall_start + frame_num * frame_interval
            wait = next_due - time.time()
            if wait > 0:
                time.sleep(wait)

    cap.release()
    # Drain the sink before closing anything it writes to. The queue holds at
    # most a frame or two of backlog, but a pending metrics POST can be sitting
    # on its 2s timeout, hence the bounded join rather than a bare one.
    sink_q.put(None)
    sink_thread.join(timeout=15)
    csv_file.close()
    if tracks_file is not None:
        tracks_file.close()

    with print_lock:
        pacing_desc = (f"paced={src_fps:.3g}fps late={frames_late} max_lag={max_lag_ms:.0f}ms "
                       if frame_interval else "unpaced ")
        if gate_mode == "budget":
            gate_detail = f"budget={diff_budget:g} max_skip={max_skip} floor_hits={budget_floor_hits} "
        elif gate_mode == "motion":
            spent_rate = (disp_rate_max_ewma if motion_signal == "max"
                          else disp_rate_ewma)
            gate_detail = (f"motion_budget={motion_budget:g} max_skip={max_skip} "
                           f"floor_hits={budget_floor_hits} signal={motion_signal} "
                           f"final_rate={spent_rate or 0:.4f} ")
        elif gate_mode == "flow":
            gate_detail = (f"flow_budget={flow_budget:g} max_skip={max_skip} "
                           f"floor_hits={budget_floor_hits} signal={flow_signal} "
                           f"final_flow_rate={flow_ewma or 0:.4f} ")
        else:
            gate_detail = ""
        warp_desc = ("warp=on " if warp_carried and flow_meter is not None else "warp=off ")
        # ids_seen is bookkeeping, not accuracy - see the CSV_HEADER note on
        # unique_total. The accuracy of this run is whatever track_eval.py says
        # about its dump next to an ungated one, and there is no way to state it
        # from inside a single run.
        print(f"stream {stream_id} COMPLETE: gate={gate_mode} {gate_detail}{warp_desc}{pacing_desc}"
              f"frames={frame_num} inferred={frames_inferred} skipped={frames_skipped} "
              f"failed={frames_failed} ids_seen={len(seen_ids)}, "
              f"crossings={count_total if count_total is not None else 'n/a'}, "
              f"Total time={time.time() - wall_start:.1f}s")
        if count_total is None:
            print(f"stream {stream_id} NOTE: no crossings counted - the cloud has no "
                  f"line for {video_name}. Pick one with pick_line.py and put "
                  f"count_lines.json where the server runs.")
        if gate_mode != "none" and not tracks_out:
            print(f"stream {stream_id} NOTE: gated run with no --tracks-out, so it "
                  f"cannot be scored. Re-run with --tracks-out and compare against "
                  f"an ungated dump using track_eval.py.")


def main():
    parser = argparse.ArgumentParser(description="Edge client - processes N video streams concurrently.")
    parser.add_argument("--videos", nargs="+", default=["traffic.mp4"],
                        help="video files to process, one stream each")
    parser.add_argument("--streams", type=int, default=None,
                        help="run this many streams, cycling through --videos "
                             "(lets you load-test with copies of one file)")
    parser.add_argument("--host", default=CLOUD_HOST, help="cloud base URL")
    parser.add_argument("--gate", choices=GATE_MODES, default="none",
                        help="frame gating mode: none = infer on every frame "
                             "(baseline), fixed = every --frame-gap'th frame, "
                             "adaptive = only on content-shift frames (frame_diff "
                             "unusual vs a rolling baseline), budget = whenever the "
                             "change accumulated since the last inference exceeds "
                             "--budget, with a --max-skip safety floor, motion = "
                             "whenever tracked boxes are predicted to have drifted "
                             "past --motion-budget, same floor, flow = as motion but "
                             "spending MEASURED optical flow instead of an "
                             "extrapolated track rate, so the signal is not censored "
                             "by the gate's own skipping")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false",
                        help="decode as fast as the machine allows instead of pacing to "
                             "the video's fps - measures raw pipeline capacity rather "
                             "than live-stream latency")
    parser.add_argument("--frame-gap", type=int, default=DEFAULT_FRAME_GAP,
                        help=f"'fixed' gate only: run inference every Nth frame "
                             f"(default {DEFAULT_FRAME_GAP})")
    parser.add_argument("--budget", type=float, default=DEFAULT_DIFF_BUDGET,
                        help=f"'budget' gate only: infer once summed frame_diff since "
                             f"the last inference reaches this value (default "
                             f"{DEFAULT_DIFF_BUDGET:g}). Pick it offline from the "
                             f"frame_diff column of an ungated run - it is the change "
                             f"you are willing to let pass unseen, so e.g. the sum of "
                             f"frame_diff over the longest gap you can tolerate")
    parser.add_argument("--max-skip", type=int, default=DEFAULT_MAX_SKIP,
                        help=f"'budget', 'motion' and 'flow' gates: never skip more than "
                             f"this many consecutive frames, whatever the budget says - bounds "
                             f"the worst-case miss (default {DEFAULT_MAX_SKIP})")
    parser.add_argument("--motion-budget", type=float, default=DEFAULT_MOTION_BUDGET,
                        help=f"'motion' gate only: infer once tracked boxes are predicted "
                             f"to have drifted this far, as a fraction of frame "
                             f"width (default {DEFAULT_MOTION_BUDGET:g}); --motion-signal picks "
                             f"whether that is the fastest box or the mean. Unlike --budget this "
                             f"is directly interpretable - it is the staleness you allow, "
                             f"and association fails at roughly a third of "
                             f"a box width. Profile the matching disp_rate_*_ewma column on an "
                             f"ungated run and "
                             f"divide the budget by it to get the skip length")
    parser.add_argument("--motion-signal", choices=sorted(MOTION_SIGNALS),
                        default=DEFAULT_MOTION_SIGNAL,
                        help=f"which per-track drift statistic the motion budget "
                             f"spends (default {DEFAULT_MOTION_SIGNAL}). 'max' follows "
                             f"the fastest box, which is the object association "
                             f"loses first, but on quiet footage it tracks detector "
                             f"jitter rather than scene motion and the gate stops "
                             f"discriminating. 'mean' keeps more of the signal. Both "
                             f"are logged either way, so an ungated run lets you "
                             f"profile one against the other before choosing")
    parser.add_argument("--flow-budget", type=float, default=None,
                        help=f"'flow' gate only: infer once summed optical flow since "
                             f"the last inference reaches this value, in fractions of "
                             f"frame width. Defaults to --motion-budget x "
                             f"{FLOW_TO_MOTION_RATIO:g}, which is the measured ratio "
                             f"between flow_moving_p95 and disp_rate_max on this "
                             f"footage - flow and track displacement are the same unit "
                             f"but not the same number, so the two gates need separate "
                             f"budgets to allow the same real staleness. Spending one "
                             f"--motion-budget on both made every head-to-head unfair "
                             f"by about 20%%")
    parser.add_argument("--no-warp", dest="warp_carried", action="store_false",
                        help="serve carried-forward boxes frozen in place instead of "
                             "translating each one by the median optical flow of the "
                             "features inside it. Warping is on by default - freezing "
                             "is what forces a gate to fire every couple of frames to "
                             "stay accurate, so --no-warp is for measuring how much "
                             "the warp is worth, not for running")
    parser.add_argument("--density-metric", choices=sorted(DENSITY_METRICS),
                        default=DEFAULT_DENSITY_METRIC,
                        help=f"which spatial signal drives the logged density state "
                             f"(default {DEFAULT_DENSITY_METRIC}). Nothing consumes the "
                             f"state yet - this chooses what gets EWMA'd and classified "
                             f"so the trigger can be picked from the CSV")
    parser.add_argument("--density-lo", type=float, default=DEFAULT_DENSITY_LO,
                        help=f"below this the density state is 'low' (default "
                             f"{DEFAULT_DENSITY_LO:g}). Units follow --density-metric, so "
                             f"pick both together from a profiling run")
    parser.add_argument("--density-hi", type=float, default=DEFAULT_DENSITY_HI,
                        help=f"above this the density state is 'high' (default "
                             f"{DEFAULT_DENSITY_HI:g})")
    parser.add_argument("--flow-signal", choices=sorted(FLOW_SIGNALS),
                        default=DEFAULT_FLOW_SIGNAL,
                        help=f"which optical-flow statistic the 'flow' gate spends, "
                             f"and which one is EWMA'd into flow_ewma (default "
                             f"{DEFAULT_FLOW_SIGNAL}). All candidates are logged "
                             f"whatever this is set to, so the right one can be "
                             f"picked from an ungated run: the one whose value "
                             f"matches disp_rate_max when nothing is gated")
    parser.add_argument("--flow-scale", type=float, default=FLOW_SCALE,
                        help=f"downscale factor applied before optical flow (default "
                             f"{FLOW_SCALE:g}). Magnitudes are normalised by frame "
                             f"width, so this changes cost and precision but not units")
    parser.add_argument("--no-flow", dest="use_flow", action="store_false",
                        help="skip optical flow entirely. Saves a few ms per frame "
                             "and blanks the flow_* columns - only useful for "
                             "measuring what flow itself costs")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop each stream after this many processed frames. Use "
                             "it to pin every run in a sweep to identical footage - "
                             "traffic.mp4 is 9184 frames and its density varies "
                             "through the clip, so runs of different lengths are not "
                             "comparable (e.g. 1800 = the first minute at 30fps)")
    parser.add_argument("--tracks-out", default=None, metavar="PREFIX",
                        help="also dump per-frame detections to PREFIX_stream<id>.jsonl, "
                             "for offline scoring with track_eval.py. Dump an ungated "
                             "run once to use as the reference, then dump each gated "
                             "run and score against it")
    parser.add_argument("--encode", choices=ENCODINGS, default="png",
                        help="wire format: png = lossless golden run, "
                             "jpeg = lossy baseline")
    parser.add_argument("--width", type=int, default=None,
                        help="downscale to this width before encoding, keeping the "
                             "source aspect ratio (e.g. 1920 / 1280 / 640). Omit, or "
                             "pass the native width, to send at full resolution. Match "
                             "this to the cloud's --imgsz for a like-for-like sweep")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY,
                        help=f"jpeg only: encoder quality 1-100 (default {DEFAULT_JPEG_QUALITY})")
    args = parser.parse_args()

    if args.frame_gap < 1:
        parser.error("--frame-gap must be >= 1")

    if args.budget <= 0:
        parser.error("--budget must be > 0")

    if args.max_skip < 1:
        parser.error("--max-skip must be >= 1")

    if args.motion_budget <= 0:
        parser.error("--motion-budget must be > 0")

    if args.flow_budget is not None and args.flow_budget <= 0:
        parser.error("--flow-budget must be > 0")

    if args.gate == "flow" and not args.warp_carried:
        print("NOTE: --gate flow with --no-warp. The flow field is being spent on "
              "the gate decision but not used to move the boxes it gates, which is "
              "the ablation, not the intended configuration.")

    if args.density_lo >= args.density_hi:
        parser.error("--density-lo must be < --density-hi")

    if not 0 < args.flow_scale <= 1:
        parser.error("--flow-scale must be > 0 and <= 1")

    if args.gate == "flow" and not args.use_flow:
        parser.error("--gate flow needs optical flow; drop --no-flow")

    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be >= 1")

    if args.width is not None and args.width < 1:
        parser.error("--width must be >= 1")

    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    count = args.streams or len(args.videos)
    sources = [args.videos[i % len(args.videos)] for i in range(count)]

    threading.Thread(target=sample_host_usage, daemon=True).start()

    gate_desc = args.gate
    if args.gate == "fixed":
        gate_desc += f" (every {args.frame_gap} frames)"
    elif args.gate == "budget":
        gate_desc += f" (budget {args.budget:g}, max skip {args.max_skip})"
    elif args.gate == "motion":
        gate_desc += (f" (drift {args.motion_budget:g} on {args.motion_signal}, "
                      f"max skip {args.max_skip})")
    elif args.gate == "flow":
        eff_flow_budget = (args.flow_budget if args.flow_budget is not None
                           else args.motion_budget * FLOW_TO_MOTION_RATIO)
        derived = "" if args.flow_budget is not None else " derived"
        gate_desc += (f" (flow {eff_flow_budget:g}{derived} on {args.flow_signal}, "
                      f"max skip {args.max_skip})")
    pace_desc = "real-time (source fps)" if args.realtime else "unpaced (max decode speed)"
    flow_desc = f"{args.flow_signal} @ scale {args.flow_scale:g}" if args.use_flow else "off"
    if args.use_flow:
        flow_desc += f", warp {'on' if args.warp_carried else 'off'}"
    print(f"Starting {count} concurrent stream(s) -> {args.host} | gate: {gate_desc} | pacing: {pace_desc} "
          f"| density signal: {args.density_metric} ({args.density_lo:g}/{args.density_hi:g}) "
          f"| flow: {flow_desc}"
          + (f" | tracks -> {args.tracks_out}_stream*.jsonl" if args.tracks_out else ""))
    threads = []
    for stream_id, video_path in enumerate(sources):
        t = threading.Thread(target=run_stream,
                             kwargs=dict(stream_id=stream_id, video_path=video_path,
                                         host=args.host, gate_mode=args.gate,
                                         frame_gap=args.frame_gap, realtime=args.realtime,
                                         encoding=args.encode, width=args.width,
                                         jpeg_quality=args.jpeg_quality,
                                         diff_budget=args.budget, max_skip=args.max_skip,
                                         motion_budget=args.motion_budget,
                                         motion_signal=args.motion_signal,
                                         density_metric=args.density_metric,
                                         density_lo=args.density_lo, density_hi=args.density_hi,
                                         flow_signal=args.flow_signal,
                                         flow_scale=args.flow_scale, use_flow=args.use_flow,
                                         flow_budget=args.flow_budget,
                                         warp_carried=args.warp_carried,
                                         tracks_out=args.tracks_out,
                                         max_frames=args.max_frames),
                             name=f"stream-{stream_id}", daemon=True)
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted - stopping.")

    print("All streams finished.")


if __name__ == "__main__":
    main()
