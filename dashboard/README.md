# Cloud Pipeline Monitor (S2 dashboard)

A single self-contained HTML file (`index.html`) that polls `GET /metrics/data`
once a second and renders it as a live-updating dashboard: resource
utilization, application performance, pipeline efficiency, and a few
current-value stat cards. No build step, no server of its own — it's just a
static file that calls `fetch()` against `server_side.py`.

## Expected `/metrics/data` response

This matches what `server_side.py` actually serves: two independent rolling
histories (`deque(maxlen=300)`), one per stream, each with its own cadence and
field set — there's no single flat "current snapshot" object.

```json
{
  "edge": [
    {
      "frame": 1432, "ts": 1690000000.0,
      "storage_io_ms": 2.1, "preprocess_ms": 6.4, "round_trip_ms": 55.0,
      "decode_ms": 4.0, "inference_ms": 38.0, "network_ms": 13.0,
      "end_to_end_ms": 63.5, "throughput_fps": 24.3,
      "objects_in_frame": 6, "counts": {"car": 5, "truck": 1},
      "unique_total": 812, "edge_cpu": 41.0, "edge_mem": 55.0,
      "payload_kb": 82.3, "bandwidth_mbps": 11.9,
      "frame_diff": 3.2, "content_shift_detected": false, "ttff_ms": null
    }
  ],
  "cloud": [
    {
      "ts": 1690000000.0, "cpu_percent": 42.1, "cpu_per_core": [40.0, 44.0],
      "mem_percent": 63.0, "mem_used_mb": 8123.4,
      "net_sent_mb": 512.3, "net_recv_mb": 900.1,
      "proc_cpu": 30.0, "proc_mem_mb": 1200.5, "load_avg": 1.8
    }
  ]
}
```

- **`edge`** is pushed from `client_side.py` every 15 frames of video, so its
  timestamps are irregular and tied to playback progress, not wall-clock
  seconds.
- **`cloud`** is sampled locally by `server_side.py` roughly once a second.
- `frame` and `unique_total` are cumulative counters (only ever go up); the
  dashboard reads them directly as "total processed" / "unique objects
  tracked" rather than deriving a rate, since `throughput_fps` is already
  computed server-side.
- There is no GPU metric, no "frames received/filtered" counter, and no
  latency percentile in this schema — the dashboard reflects that: the GPU
  line, filter-rate chart, and in-flight-requests stat from earlier drafts
  have been dropped/replaced (see "What's on screen" below).
- Because `edge` and `cloud` arrive on different cadences, each stream gets
  its own rolling-window state and its own x-axis (time) labels — they are
  never plotted on a shared label array.

## Running it

1. Open `dashboard/index.html` in a browser — either double-click it, or
   serve the folder locally so relative paths behave consistently:
   ```
   cd dashboard
   python -m http.server 8080
   ```
   then visit `http://localhost:8080`.

2. Point it at your service. Open `index.html` and edit the `CONFIG` block
   near the top of the `<script>` — it's the only place you should need to
   touch:
   ```js
   const CONFIG = {
     METRICS_URL: "http://localhost:8000/metrics/data",  // <-- server_side.py's GET endpoint
     POLL_INTERVAL_MS: 1000,
     WINDOW_SIZE: 90,            // rolling samples kept on screen per chart, per stream
     FETCH_TIMEOUT_MS: 3000,
     SHIFT_RATE_WINDOW: 10,      // trailing edge samples for the content-shift rate chart
     STATUS_LOOKBACK: 5,         // trailing edge samples for the content-shift status badge
   };
   ```

3. **CORS**: since this is a static page fetching a different origin,
   `server_side.py` now sends `Access-Control-Allow-Origin: *` on every
   response (see the `add_cors_headers` hook near the top of the file). If
   you swap in a different server, it needs the same header or the browser
   will block every request (the dashboard will just sit in
   "Reconnecting…" and the browser console will show a CORS error).

## What's on screen

- **Stat cards**: total frames processed (cumulative, from `edge[-1].frame`),
  unique objects tracked (`edge[-1].unique_total`), current processing rate
  (`edge[-1].throughput_fps`, as reported — not re-derived), and a
  content-shift status badge (Stable / Shift detected / Frequent shifts)
  computed from how many of the last `STATUS_LOOKBACK` edge samples had
  `content_shift_detected` set.
- **Resource utilization**: two charts, since cloud and edge samples are on
  different cadences — Cloud CPU/Memory/Process-CPU %, and Edge CPU/Memory %.
- **Application performance**: throughput (fps, as reported), objects
  detected (per-frame count vs. cumulative unique-tracked total, on a
  secondary axis), and a latency breakdown (end-to-end, round trip,
  inference, network — all in ms, from the edge's own timing instrumentation).
- **Pipeline efficiency**: content-shift rate (% of the trailing
  `SHIFT_RATE_WINDOW` edge samples flagged as a content shift — the
  Reducto-style frame-diff signal in `client_side.py`) and bandwidth (Mbps,
  as measured on the edge's upload).
- Every chart has a **"View as table"** toggle for the same rolling window of
  data, as an accessible/non-visual alternative to the plot.
- If `/metrics/data` stops responding, the header shows "Reconnecting…",
  charts and stat cards dim and freeze on last-known values, and polling
  keeps retrying at the same interval. Since the server already retains full
  history, a failed poll just means we stop refreshing what's on screen —
  there's no local gap-filling needed.

## Notes / things to revisit

- Not yet tested against a live edge+cloud run end-to-end — verify frame
  counters and timestamps behave once both `client_side.py` and
  `server_side.py` are running for real.
- Colors were run through the project's palette validator (colorblind-safe
  adjacent-pair contrast) for both light and dark mode — see the categorical
  slots at the top of the `<style>` block if you need to change them.
- No external JS dependency besides Chart.js, loaded from a CDN
  (`cdn.jsdelivr.net`) — if the grading environment has no internet access,
  download `chart.umd.min.js` and change the `<script src=...>` tag to a
  local path.
- `cpu_per_core`, `mem_used_mb`, `net_sent_mb`/`net_recv_mb`, `load_avg`,
  `counts` (per-label object counts), and `ttff_ms` are all present in the
  data but not currently wired into a chart — straightforward to add if
  they'd be useful.
