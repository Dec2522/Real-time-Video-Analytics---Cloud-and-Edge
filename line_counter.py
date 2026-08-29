#!/usr/bin/env python3
"""Count vehicles across a virtual line, from live detections or a track dump.

Why a line and not unique track IDs. `ids_seen` counts every distinct ID a run
ever emitted, so it is the sum of three unrelated things: real vehicles, the same
vehicle re-acquired under a new ID after the tracker dropped it, and detector
flicker that lived for two frames and was never a vehicle. On the 640-frame
reference for clip4_5_cropped_25 that came to 13 IDs for roughly 7 vehicles -
five tracks alive 1-4 frames, and one car handed over from id 3824 (f500-569) to
id 3825 (f570-640) mid-clip.

A line crossing needs identity to hold only across the two frames either side of
the line, instead of across the vehicle's whole life in shot. A 600-frame track
that switches ID once is counted once here and twice by ids_seen. Combined with a
minimum age, which discards flicker before it can ever be counted, this is a
throughput number you can actually defend.

By default a track is counted AT MOST ONCE, however many times it crosses. The
number is a vehicle count - throughput past the line as a congestion measure -
so a box that drifts back over the line must not add to it. The cooldown this
replaced only suppressed a re-crossing within N frames, which left a slow drift
back across counting twice. Pass --allow-recross for the older behaviour, where
crossings are counted in both directions and the cooldown bounds the jitter.

Offline, over dumps you already have:

  python line_counter.py --line 0.0,0.55,1.0,0.55 \
      --runs crop25_results/tracks_none_stream0.jsonl \
             crop25_results/tracks_fixed_6_stream0.jsonl \
             crop25_results/tracks_flow_mp95_005_stream0.jsonl

Live, drawn on the video, via gate_viewer.py --count-line.
"""

import argparse
import json
import os
from collections import Counter

DEFAULT_MIN_AGE = 5     # frames a track must have been seen before it may count
DEFAULT_COOLDOWN = 10   # frames before the same track may be counted again
DEFAULT_LINE = (0.0, 0.55, 1.0, 0.55)

# The counting line for each video, as (x1, y1, x2, y2) in normalised coords.
# Pick them with pick_line.py - it prints these four numbers when you click the
# two endpoints - and paste them in here.
#
# None means "not set yet", and that video is NOT counted. Deliberately not
# defaulted to DEFAULT_LINE: a plausible line in the wrong place produces counts
# that look valid, which is worse than no counts at all.
VIDEO_LINES = {
    "clip1_cropped.mp4": (0.4, 0.4, 1, 0.387),
    "clip4_5_cropped.mp4": (0.4, 0.35, 1, 0.315),
    "clip5_3_cropped.mp4": (0.4, 0.38, 1, 0.34),
}


def line_for(video):
    """The configured line for a video, or None if it has no entry.

    Matched on filename alone, so the caller's path does not matter - the edge
    sends a basename and this file is edited by hand with the same.
    """
    if not video:
        return None
    return VIDEO_LINES.get(os.path.basename(video))


def parse_line(text):
    """'x1,y1,x2,y2' in normalised coords -> tuple of four floats."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--line needs four comma-separated numbers: x1,y1,x2,y2")
    vals = tuple(float(p) for p in parts)
    if not all(0.0 <= v <= 1.0 for v in vals):
        raise ValueError("--line coordinates are normalised, so all four must be 0..1")
    if vals[0] == vals[2] and vals[1] == vals[3]:
        raise ValueError("--line endpoints are identical, that is a point not a line")
    return vals


def _side(line, px, py):
    """Signed area of the triangle (A, B, P). Sign says which side of AB P is on.

    Zero means exactly on the line. Magnitude is proportional to distance, but
    only the sign is used - a point that lands exactly on the line keeps its
    previous side rather than being treated as a crossing, which is handled by
    the strict inequalities in crossed().
    """
    ax, ay, bx, by = line
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def crossed(line, prev, curr):
    """Did the segment prev->curr cross the line segment AB?

    Full segment-segment intersection, not just a side flip. A side flip alone
    counts a vehicle that passed the line's infinite extension far outside the
    junction - on a line spanning only part of the frame that is a real
    over-count, so both orientation tests are needed.

    Returns None, or 'in' / 'out' for the direction of travel. Which physical
    direction those names mean depends on the order the two endpoints were given:
    'in' is a crossing that ends on the left of A->B.
    """
    ax, ay, bx, by = line
    d1 = _side(line, prev[0], prev[1])
    d2 = _side(line, curr[0], curr[1])
    if (d1 > 0) == (d2 > 0) or d1 == 0 or d2 == 0:
        return None
    # Orientation of A and B about the movement segment. Without this, any
    # motion anywhere in the frame that flips sides against the INFINITE line
    # would register.
    mline = (prev[0], prev[1], curr[0], curr[1])
    d3 = _side(mline, ax, ay)
    d4 = _side(mline, bx, by)
    if (d3 > 0) == (d4 > 0):
        return None
    return "in" if d2 > 0 else "out"


class LineCounter:
    """Streaming counter. Feed it one frame of detections at a time.

    Holds the previous centre per track id, so it is order-independent within a
    frame but does depend on being called once per frame in order.
    """

    def __init__(self, line=DEFAULT_LINE, min_age=DEFAULT_MIN_AGE, labels=None,
                 cooldown=DEFAULT_COOLDOWN, once_per_track=True):
        self.line = line
        self.min_age = min_age
        self.cooldown = cooldown
        # One track, one vehicle, one count. `cooldown` is only consulted when
        # this is off - see the note in update().
        self.once_per_track = once_per_track
        # None means count every label. Anything else is a set of label strings.
        self.labels = set(labels) if labels else None
        self.age = Counter()          # track id -> frames it has been seen
        self.prev = {}                # track id -> last centre
        self.counts = Counter()       # 'in' / 'out' -> crossings
        self.events = []              # (frame, track id, direction)
        self.unique = set()           # track ids that crossed at least once
        self.last_cross = {}          # track id -> frame it last counted on
        self.suppressed = 0           # crossings dropped by the cooldown

    def update(self, frame_num, dets):
        """Advance one frame. Returns this frame's crossing events."""
        fired = []
        for d in dets or []:
            tid = d.get("id")
            if tid is None or "cx" not in d:
                continue
            if self.labels is not None and d.get("label") not in self.labels:
                continue
            centre = (d["cx"], d["cy"])
            self.age[tid] += 1
            prev = self.prev.get(tid)
            self.prev[tid] = centre
            # A track must have been around for min_age frames before it may be
            # counted. This is what discards detector flicker: on the reference
            # dump five of thirteen tracks never reach five frames.
            if prev is None or self.age[tid] < self.min_age:
                continue
            direction = crossed(self.line, prev, centre)
            if not direction:
                continue
            # A box that sits on the line jitters back and forth across it and
            # scores a crossing each way. Observed on fixed_3: id 3833 counted
            # 'in' at f207 and 'out' at f209, turning three real crossings into
            # five.
            #
            # Counting each track once removes that outright, and does not depend
            # on how quickly the box comes back - a slow drift across the line
            # over twenty frames beat the cooldown and was counted twice.
            if self.once_per_track:
                if tid in self.unique:
                    self.suppressed += 1
                    continue
            else:
                last = self.last_cross.get(tid)
                if last is not None and frame_num - last < self.cooldown:
                    self.suppressed += 1
                    continue
            self.last_cross[tid] = frame_num
            self.counts[direction] += 1
            self.events.append((frame_num, tid, direction))
            self.unique.add(tid)
            fired.append((tid, direction))
        return fired

    @property
    def total(self):
        return self.counts["in"] + self.counts["out"]

    def summary(self):
        return {
            "in": self.counts["in"],
            "out": self.counts["out"],
            "total": self.total,
            "unique_tracks": len(self.unique),
            "tracks_seen": len(self.age),
            "tracks_old_enough": sum(1 for n in self.age.values() if n >= self.min_age),
            "suppressed": self.suppressed,
        }


def count_dump(path, line, min_age, labels, cooldown=DEFAULT_COOLDOWN,
               once_per_track=True):
    """Replay a --tracks-out JSONL through a LineCounter."""
    counter = LineCounter(line, min_age, labels, cooldown, once_per_track)
    frames = 0
    with open(path) as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                # A run killed with Ctrl-C leaves a half-written final line.
                print(f"  [{os.path.basename(path)}] ignoring malformed line {line_no}")
                continue
            frames += 1
            counter.update(row["frame"], row.get("dets") or [])
    out = counter.summary()
    out["frames"] = frames
    out["events"] = counter.events
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="track dumps to count, from client_side --tracks-out")
    p.add_argument("--line", default=",".join(str(v) for v in DEFAULT_LINE),
                   help="counting line as x1,y1,x2,y2 in normalised coords "
                        f"(default {','.join(str(v) for v in DEFAULT_LINE)}, a "
                        f"horizontal line across the middle)")
    p.add_argument("--min-age", type=int, default=DEFAULT_MIN_AGE,
                   help=f"frames a track must exist before it may be counted "
                        f"(default {DEFAULT_MIN_AGE}) - this is what discards "
                        f"detector flicker")
    p.add_argument("--labels", nargs="+", default=None,
                   help="only count these labels, e.g. car truck bus. Default counts all")
    p.add_argument("--allow-recross", action="store_true",
                   help="count a track every time it crosses, not just once. Use "
                        "for a directional junction study; leave off for a vehicle "
                        "count, where one track is one vehicle")
    p.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN,
                   help=f"only with --allow-recross: frames before the same track "
                        f"may be counted again (default {DEFAULT_COOLDOWN}), which "
                        f"suppresses a box jittering on the line. 0 disables it")
    p.add_argument("--events", action="store_true",
                   help="also list every crossing, so a suspect count can be "
                        "checked against the video frame by frame")
    p.add_argument("--csv", default=None, help="also write the table here")
    args = p.parse_args()

    line = parse_line(args.line)
    mode = f"cooldown {args.cooldown}" if args.allow_recross else "once per track"
    print(f"line {line}  min_age {args.min_age}  {mode}  labels "
          f"{' '.join(args.labels) if args.labels else 'all'}\n")

    rows = []
    for path in args.runs:
        res = count_dump(path, line, args.min_age, args.labels, args.cooldown,
                         not args.allow_recross)
        res["run"] = os.path.basename(path)
        rows.append(res)

    width = max(len(r["run"]) for r in rows)
    print(f"{'run':<{width}}  frames    in   out  total  uniq  tracks  aged  supp")
    print("-" * (width + 52))
    for r in rows:
        print(f"{r['run']:<{width}}  {r['frames']:6}  {r['in']:4}  {r['out']:4}  "
              f"{r['total']:5}  {r['unique_tracks']:4}  {r['tracks_seen']:6}  "
              f"{r['tracks_old_enough']:4}  {r['suppressed']:4}")
    print("\n  in/out  crossings by direction; 'in' ends on the left of the line A->B")
    print("  uniq    distinct tracks that crossed - lower than total if one crossed back")
    print("  tracks  distinct ids seen at all; aged = those that reached --min-age.")
    print("          The gap between them is the flicker --min-age discarded.")
    if args.allow_recross:
        print("  supp    re-crossings dropped by --cooldown, i.e. a box jittering on the line.")
    else:
        print("  supp    repeat crossings by a track already counted. Non-zero means "
              "boxes\n          are re-crossing the line - usually jitter, and the "
              "count total is\n          unaffected either way.")

    if args.events:
        for r in rows:
            print(f"\n{r['run']} crossings:")
            for frame, tid, direction in r["events"]:
                print(f"  f{frame:<5} id {tid:<6} {direction}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "frames", "in", "out", "total", "unique_tracks",
                        "tracks_seen", "tracks_old_enough"])
            for r in rows:
                w.writerow([r["run"], r["frames"], r["in"], r["out"], r["total"],
                            r["unique_tracks"], r["tracks_seen"],
                            r["tracks_old_enough"]])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
