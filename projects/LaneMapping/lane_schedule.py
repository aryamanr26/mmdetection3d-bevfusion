"""2 Hz MapTRv2 scheduling with a single in-flight inference.

A new frame is eligible every ``period_s``. If inference is already running,
only the latest eligible frame is kept. Completing an older request never
discards a newer result, and the queued frame starts only after the running
one finishes.
"""


class LatestFrameScheduler:
    def __init__(self, period_s=0.5):
        self.period_s = period_s
        self.pending = None
        self.inflight = None
        self.accepted_seq = -1
        self._next_due = None

    def submit(self, frame, now):
        """Consider ``frame`` at wall time ``now``.

        Returns ``(job, scheduled)``. ``job`` is the inference to start now,
        or None when one is already running. ``scheduled`` is true when this
        frame became the pending or running update.
        """
        if self._next_due is None:
            self._next_due = now
        if now + 1e-6 < self._next_due:
            return None, False
        self.pending = frame
        self._next_due += self.period_s
        # If the clock jumped, don't schedule a burst of catch-up frames.
        if self._next_due < now:
            self._next_due = now + self.period_s
        return self._launch(), True

    def complete(self, seq):
        """Inference of ``seq`` finished. Return the next job, or None."""
        if self.inflight is not None and self.inflight['seq'] == seq:
            self.inflight = None
        if seq > self.accepted_seq:
            self.accepted_seq = seq
        return self._launch()

    def accept_result(self, seq):
        """False when ``seq`` is older than a result already accepted."""
        if seq <= self.accepted_seq:
            return False
        self.accepted_seq = seq
        return True

    def _launch(self):
        if self.inflight is not None or self.pending is None:
            return None
        self.inflight = self.pending
        self.pending = None
        return self.inflight
