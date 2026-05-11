"""Telemetry session helpers for profiling runs."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, Optional

from ..core.types import TelemetryReading
from ..telemetry import EnergyMonitorCollector


@dataclass
class TelemetrySample:
    timestamp: float
    reading: TelemetryReading


class TelemetrySession(AbstractContextManager["TelemetrySession"]):
    """Capture telemetry readings in a background thread."""

    def __init__(
        self,
        collector: EnergyMonitorCollector,
        *,
        buffer_seconds: Optional[float] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self._collector = collector
        self._buffer_seconds = buffer_seconds
        self._max_samples = max_samples
        self._samples: Deque[TelemetrySample] = deque(maxlen=max_samples)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._collector_ctx = None
        self._gpu_device_id = self._resolve_gpu_device_id()

    def __enter__(self) -> "TelemetrySession":
        self._collector_ctx = self._collector.start()
        self._collector_ctx.__enter__()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._collector_ctx is not None:
            self._collector_ctx.__exit__(None, None, None)

    def _run(self) -> None:
        try:
            for reading in self._collector.stream_readings():
                if not self._include_reading(reading):
                    continue
                timestamp = (
                    float(reading.timestamp_nanos) / 1_000_000_000.0
                    if reading.timestamp_nanos is not None
                    else time.time()
                )
                self._samples.append(
                    TelemetrySample(timestamp=timestamp, reading=reading)
                )
                self._trim(timestamp)
                if self._stop_event.is_set():
                    break
        except Exception:  # pragma: no cover - surface to caller on access
            self._stop_event.set()
            raise

    def _resolve_gpu_device_id(self) -> Optional[int]:
        device = os.getenv("IPW_GPU_DEVICE_ID")
        if device is None:
            visible = [part.strip() for part in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
            device = visible[0] if len(visible) == 1 else None
        try:
            return int(device) if device is not None else None
        except (TypeError, ValueError):
            return None

    def _include_reading(self, reading: TelemetryReading) -> bool:
        gpu_info = reading.gpu_info
        return self._gpu_device_id is None or gpu_info is None or int(gpu_info.device_id) == self._gpu_device_id

    def _drop_before(self, timestamp: float) -> None:
        while self._samples and self._samples[0].timestamp < timestamp:
            self._samples.popleft()

    def _trim(self, current_time: float) -> None:
        if self._buffer_seconds is not None:
            self._drop_before(current_time - self._buffer_seconds)

    def prune_before(self, timestamp: float) -> None:
        """Discard samples older than ``timestamp`` after they are consumed."""
        self._drop_before(timestamp)

    def readings(self) -> Iterable[TelemetrySample]:
        return list(self._samples)

    def window(self, start_time: float, end_time: float) -> Iterator[TelemetrySample]:
        for sample in list(self._samples):
            if start_time <= sample.timestamp <= end_time:
                yield sample
