"""Count vehicles that cross a virtual line in a video.

Introduced because counting unique YOLO detection IDs is not accurate. It commonly 
jitters detections on the same object leading to double/triple etc. counting. 

Line counting asks if the same detection ID has crossed a line. 
"""

import argparse
import json
import os
from collections import Counter

DEFAULT_MIN_AGE = 5     # frames a track must have been seen before it may count
DEFAULT_COOLDOWN = 10   # frames before the same track may be counted again
DEFAULT_LINE = (0.0, 0.55, 1.0, 0.55)

# Pre-configured lines - a video won't be counted unless it has an entry here.
VIDEO_LINES = {
    "clip1_tight.mp4": (0.0,0.18,1,0.16),
    "clip4_5_tight.mp4": (0.0, 0.2, 1, 0.17),
    "clip5_3_tight.mp4": (0.0,0.225,1,0.175),
}

def line_for(video):
    """Return the line for a video."""
    if not video:
        return None
    return VIDEO_LINES.get(os.path.basename(video))

def _side(line, px, py):
    """Signed area of the triangle (A, B, P). Sign says which side of AB (the line) P (the vehicle) is on."""
    ax, ay, bx, by = line
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

def crossed(line, prev, curr):
    """Did a vehicle cross the line?

    compares `_side` of the current and previous, if one is positive and the other negative, it crossed.

    The line covers the full width of the frame, but if this was partial, a second orientation check
    on the cars movement lines would be needed. 
    """
    ax, ay, bx, by = line
    d1 = _side(line, prev[0], prev[1])
    d2 = _side(line, curr[0], curr[1])
    if (d1 > 0) == (d2 > 0) or d1 == 0 or d2 == 0:
        return None
    # Should not be bidirectional for the selected videos, but a catch.
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
        self.age = Counter()          
        self.prev = {}                
        self.counts = Counter()       
        self.unique = set()           

    def update(self, frame_num, dets):
        """Advance one frame. Returns this frame's crossing events."""
        for d in dets or []:
            tid = d.get("id")
            if tid is None or "cx" not in d:
                continue
            centre = (d["cx"], d["cy"])
            self.age[tid] += 1
            prev = self.prev.get(tid)
            self.prev[tid] = centre
            # A track must have be seen for n frames to avoid detection flicker
            if prev is None or self.age[tid] < self.min_age:
                continue
            # Already counted - skip
            if tid in self.unique:
                continue
            direction = crossed(self.line, prev, centre)
            if not direction:
                continue
            self.counts[direction] += 1
            self.unique.add(tid)

    @property
    def total(self):
        return self.counts["in"] + self.counts["out"]
    

