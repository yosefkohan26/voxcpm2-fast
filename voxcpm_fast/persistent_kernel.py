"""Python wrapper for the P2.1 persistent-kernel PoC.

Design
------
The persistent kernel has two roles:

* **Dispatcher** (block 0) — polls a pinned-mapped doorbell *ring* via
  ``ld.acquire.sys`` loads, drains every newly-published item into an
  HBM-resident work queue, and mirrors the pinned terminate flag into HBM
  so workers never need to touch PCIe for the shutdown check.
* **Workers** (blocks 1..N) — claim slots from the HBM queue via a
  round-robin token handoff (no CAS contention on the single-in-flight
  path), execute the trivial echo op, and write results + done flags into
  a pinned-mapped `done[]` array.

Submit path on the host (zero CUDA API calls):

1. Write the 16-byte payload to ``doorbell.slots[tail % CAPACITY]``.
2. Issue an SFENCE (via the C extension) and release-store the new tail.
3. Return a ``_DoneEvent`` keyed on ``done[seq % CAPACITY].flag``.

Because the doorbell is a *ring* (not a single slot), the host can enqueue
many items before the dispatcher has drained any of them. Throughput is not
gated by single-doorbell round-trip.

The wrapper exposes:

    kernel = PersistentKernel(num_sms=32)
    kernel.start()
    done_evt, out = kernel.submit(a=1, b=2, c=3)
    done_evt.wait()
    assert int(out[0]) == 7
    kernel.stop()
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch  # must precede the C extension so its shared libs load

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import persistent_poc_ext as _ext  # noqa: E402  (after sys.path)


# WorkItem layout: (i32 a, i32 b, i32 c, i32 idx) — 16 B
_WORK_ITEM_FMT = "<iiii"
_WORK_ITEM_SIZE = struct.calcsize(_WORK_ITEM_FMT)
assert _WORK_ITEM_SIZE == 16


class _DoneEvent:
    """Completion handle for a single submitted work item."""

    __slots__ = ("_flag_addr", "_expected_seq", "_out_addr", "_done")

    def __init__(self, flag_addr: int, expected_seq: int, out_addr: int):
        self._flag_addr = flag_addr
        self._expected_seq = expected_seq & 0xFFFFFFFF
        self._out_addr = out_addr
        self._done = False

    def poll(self) -> bool:
        if self._done:
            return True
        v = ctypes.c_uint32.from_address(self._flag_addr).value
        if v == self._expected_seq:
            self._done = True
            return True
        return False

    def wait(self, timeout: float | None = None) -> None:
        if self._done:
            return
        deadline = None if timeout is None else (time.perf_counter() + timeout)
        while not self.poll():
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError("done flag never fired")

    def result(self) -> int:
        return ctypes.c_int32.from_address(self._out_addr).value


class _OutputView:
    __slots__ = ("_addr",)

    def __init__(self, addr: int):
        self._addr = addr

    def __getitem__(self, idx: int) -> int:
        if idx != 0:
            raise IndexError("output view is size 1")
        return ctypes.c_int32.from_address(self._addr).value


class PersistentKernel:
    """Persistent dispatcher + worker kernel with host-side producer API."""

    def __init__(self, num_sms: int = 32):
        if num_sms <= 0:
            raise ValueError("num_sms must be > 0")
        self._num_sms = int(num_sms)

        info = _ext.queue_info()
        (self._doorbell_slots_off, self._doorbell_tail_off,
         self._doorbell_head_off, self._doorbell_slot_stride,
         self._done_flag_off, self._done_out_off, self._done_stride,
         self._ctrl_terminate_off,
         self._queue_capacity, self._doorbell_ring_capacity) = info

        self._queue_capacity_mask = self._queue_capacity - 1
        self._ring_capacity_mask = self._doorbell_ring_capacity - 1
        assert (self._queue_capacity & self._queue_capacity_mask) == 0
        assert (self._doorbell_ring_capacity & self._ring_capacity_mask) == 0

        self._doorbell_addr = 0
        self._ctrl_addr = 0
        self._done_addr = 0
        self._q_dev_addr = 0
        self._stream_addr = 0

        # Host-side producer state. `_next_tail` is the next tail value we
        # will write into doorbell.tail. It starts at 1 (first submit writes
        # slot 0 and publishes tail=1). `_last_dispatcher_head_seen` caches
        # the most recent head we've observed from the dispatcher, used for
        # backpressure when the ring would otherwise wrap.
        self._next_tail = 1
        self._last_dispatcher_head_seen = 0

        self._submit_lock = threading.Lock()

        self._started = False
        self._stopped = False

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._started:
            return
        ptrs = _ext.launch_persistent(self._num_sms).tolist()
        self._doorbell_addr = int(ptrs[0])
        self._ctrl_addr = int(ptrs[1])
        self._done_addr = int(ptrs[2])
        self._q_dev_addr = int(ptrs[3])
        self._stream_addr = int(ptrs[4])

        self._doorbell_slots_addr = self._doorbell_addr + self._doorbell_slots_off
        self._doorbell_tail_addr = self._doorbell_addr + self._doorbell_tail_off
        self._doorbell_head_addr = self._doorbell_addr + self._doorbell_head_off

        self._started = True

    def stop(self, shutdown_timeout: float = 0.1) -> None:
        if not self._started or self._stopped:
            return
        terminate_addr = self._ctrl_addr + self._ctrl_terminate_off
        ctypes.c_uint32.from_address(terminate_addr).value = 1

        cuda = _libcudart()
        stream = ctypes.c_void_p(self._stream_addr)
        deadline = time.perf_counter() + shutdown_timeout
        while True:
            rc = cuda.cudaStreamQuery(stream)
            if rc == 0:
                break
            if time.perf_counter() > deadline:
                _ext.shutdown_persistent()
                self._stopped = True
                self._started = False
                raise TimeoutError(
                    f"persistent kernel did not exit within "
                    f"{shutdown_timeout*1000:.1f} ms (stream rc={rc})"
                )
            time.sleep(0.0005)

        _ext.shutdown_persistent()
        self._stopped = True
        self._started = False

    # ---------- hot path ----------

    def submit(self, a: int, b: int, c: int) -> Tuple[_DoneEvent, _OutputView]:
        """Publish one work item. Returns ``(done_event, output_view)``.

        ``output_view[0]`` is valid only after ``done_event.wait()`` returns.
        """
        with self._submit_lock:
            tail = self._next_tail  # value we'll publish as doorbell.tail
            self._next_tail = (self._next_tail + 1) & 0xFFFFFFFF
        idx = tail - 1  # 0-based sequence number
        ring_pos = idx & self._ring_capacity_mask
        slot_pos = idx & self._queue_capacity_mask

        # Backpressure: if we'd outrun the dispatcher's head by more than
        # DOORBELL_RING_CAPACITY items, block until the dispatcher catches
        # up. In steady state this almost never fires because the
        # dispatcher drains as fast as the host enqueues.
        need_head = idx - self._doorbell_ring_capacity + 1
        if need_head > 0:
            while self._last_dispatcher_head_seen < need_head:
                h = ctypes.c_uint32.from_address(self._doorbell_head_addr).value
                # Handle 32-bit wrap on `last_seen` only (we're well below
                # 2^32 items in any session here).
                self._last_dispatcher_head_seen = h
                if h >= need_head:
                    break
                # tiny pause so we don't flood PCIe with head reads
                pass

        # Reset the done flag for this lap.
        flag_addr = self._done_addr + slot_pos * self._done_stride + self._done_flag_off
        out_addr = self._done_addr + slot_pos * self._done_stride + self._done_out_off
        ctypes.c_uint32.from_address(flag_addr).value = 0

        # Pack payload directly into the doorbell ring.
        struct.pack_into(_WORK_ITEM_FMT, _PAYLOAD_SCRATCH, 0,
                         a, b, c, idx & 0x7FFFFFFF)
        slot_addr = (self._doorbell_slots_addr
                     + ring_pos * self._doorbell_slot_stride)
        ctypes.memmove(slot_addr, _PAYLOAD_SCRATCH_ADDR, _WORK_ITEM_SIZE)

        # Release-ordered store of tail (SFENCE + store + SFENCE in the C
        # extension). The SFENCE guarantees the GPU observes the payload
        # bytes before the tail update, so the dispatcher never drains a
        # slot that doesn't yet contain the host's intended payload.
        _ext.store_release_seq(self._doorbell_tail_addr, tail)

        # seq used as the expected done flag value is `tail` (== idx + 1).
        evt = _DoneEvent(flag_addr, tail, out_addr)
        return evt, _OutputView(out_addr)

    def burst_submit(self, a_arr, b_arr, c_arr) -> int:
        """Push ``N`` items via a C-side tight loop.

        ``a_arr``/``b_arr``/``c_arr`` must be contiguous int32 CPU tensors
        of equal length. Returns the base sequence number of the first
        item; callers can compute per-item done-flag addresses themselves.

        The C loop amortises the per-item Python/ctypes overhead that caps
        the per-item ``submit()`` path at ~0.5 M items/s, exposing the
        underlying GPU throughput for benchmarking.
        """
        n = int(a_arr.numel())
        assert b_arr.numel() == n and c_arr.numel() == n
        with self._submit_lock:
            starting_tail = self._next_tail
            self._next_tail = (self._next_tail + n) & 0xFFFFFFFF

        _ext.burst_submit(
            self._doorbell_slots_addr,
            self._doorbell_tail_addr,
            self._doorbell_head_addr,
            self._done_addr,
            self._doorbell_slot_stride,
            self._done_stride,
            self._done_flag_off,
            self._doorbell_ring_capacity,
            starting_tail,
            n,
            a_arr, b_arr, c_arr,
        )
        # Base sequence number of the first item in this burst.
        return int(starting_tail)

    def __enter__(self) -> "PersistentKernel":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop()
        except TimeoutError:
            pass


# ---------- low-level helpers ----------

_PAYLOAD_SCRATCH = bytearray(_WORK_ITEM_SIZE)
_PAYLOAD_SCRATCH_ADDR = ctypes.addressof(
    (ctypes.c_char * _WORK_ITEM_SIZE).from_buffer(_PAYLOAD_SCRATCH)
)

_libcudart_handle = None


def _libcudart():
    global _libcudart_handle
    if _libcudart_handle is None:
        for name in ("libcudart.so.12", "libcudart.so", "libcudart.so.12.8"):
            try:
                _libcudart_handle = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _libcudart_handle is None:
            raise RuntimeError("could not locate libcudart.so")
        _libcudart_handle.cudaStreamQuery.argtypes = [ctypes.c_void_p]
        _libcudart_handle.cudaStreamQuery.restype = ctypes.c_int
    return _libcudart_handle
