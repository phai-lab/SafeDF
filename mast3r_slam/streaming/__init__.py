"""
Streaming input/output utilities for MASt3R-SLAM.

This package is intentionally optional and designed for *real-time-first* online usage:
  - Input sources should be non-blocking for the SLAM main loop (latest-only behavior).
  - Heavy operations (e.g., segmentation inference, GUI rendering) run asynchronously so
    transient latency does not stall tracking.
"""

