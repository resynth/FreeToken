# Plan: graph-capturable staged decode for the mmap expert source (H)

Follow-up plan for the last item of [mmap-tiering-performance.md](mmap-tiering-performance.md)
("H. Make staged decode CUDA-graph-capturable"). Not started; this is the design.

## Problem

With `--expert-source mmap` and a staged GPU decode (`--moe-strategy offload`, and the
fetched share of `hybrid`), `_finish_mmap_source` sets `cuda_graph_max_bs = 0`
(`engine/engine.py`), so **every decode token runs the whole model step eagerly**: one
Python-driven launch per kernel, plus one full-stream D2H sync per MoE layer
(`copy_missing` -> `_read_pending_staged`). For a 48-layer model that is ~48 host
syncs and hundreds of launches per token on top of the actual work. C cut the syncs
from two per layer to one, but the eager cost is why the mmap tier logs a warning when
it disables graphs.

The uncapturable part is exactly one call: `copy_missing` (`moe/offload_cache.py`). Its
inputs are produced by `ensure_experts` — already device-side, fixed-shape Triton, and
capture-safe (the docstring of `_decode_routed`, `layers/moe.py`, says so, and the
hybrid `ensure_experts_hybrid` is captured today). Its body is host code by
construction: read the pinned `num_indices`/`src_indices` mirrors after a stream sync,
`preadv` the missing experts out of the page cache into the pinned ring, and issue the
H2D. A captured graph replays kernels; it does not re-run Python, so a whole-step
capture would only stage at capture time.

Two things in-tree already prove the pieces exist:

* The **CPU executor flag-sync** (`moe/cpu_executor.py`): a persistent coordinator
  thread + mapped-pinned `ready`/`done`/`err` int64 arrays + a device-side WAIT whose
  immediate is constant, explicitly documented as CUDA-graph-replay-safe, with a
  watchdog that poisons `done` when a flag goes unanswered. This is the in-repo
  precedent for host work *outside* a graph feeding a captured consumer.
* The **hybrid ensure kernel** (`moe/offload_kernels.py`) writes the whole miss set
  device-side (fixed shapes, capped fetch count), so the selection half of the staged
  path is capturable today.

## Measurement gate (M0) - do this first

I and J both measured as washes on this host's NVMe because multi-MiB preads are already
at drive bandwidth; the remaining candidate cost of staged decode is (a) the per-layer
D2H syncs and (b) eager launch overhead. Quantify both before writing any capture code:

1. Server-level A/B on the real model, one fixed prompt, bs=1 decode:
   `--expert-source mmap --moe-strategy offload` (eager, graphs off — today) vs
   `--expert-source pinned --moe-strategy offload` (graphs on, where banks fit in
   pinned RAM — use a reduced `--moe-cache-size` if needed) vs
   `--moe-strategy cpu` (graphs on, D). Record tok/s and TTFT.
2. Profile the eager staged path once with the torch profiler (or Nsight) and count
   kernel launches and stream syncs per token.

Decision rule: if the staged path's deficit vs the pinned path is explained by PCIe
bytes (misses x expert_bytes / pcie_bw) rather than by launch/sync overhead, H will
not fix it - stop here and keep graphs off (the current warning is then correct and
honest). If sync+launch overhead is a majority of the deficit, take H1, then H2 only if
H1's numbers still leave launch overhead dominant. On this 48 x 512-expert model the
PCIe term is expected to dominate; H is polish, not the fix for "unusably slow".

## Option H1 - segment capture (cheap, low risk)

Keep graphs for all non-MoE work; run each MoE layer eagerly (the doc's first option).
The step alternates attention/router segments with MoE layers, so this is one small
captured graph per layer group (attention + norms + router) and an eager MoE segment
between replays. What it saves: the launch overhead of the non-MoE kernels only. What
it keeps: 48 replays/token, 48 eager MoE segments/token with their syncs and staging.

* Cost estimate: the non-MoE kernels are the cheap half of a decode step; expect
  fractions of a millisecond per token. Only worth it if M0 shows launch overhead is
  the dominant term and H2 is rejected.
* Implementation sketch: a `GraphRunner` mode that captures per-segment graphs instead
  of whole-step ones (`engine/graph.py`), with the MoE layers run eagerly between
  replays; `pad_batch`/replay bookkeeping per segment group. New flag, e.g.
  `--moe-staged-graphs segment`.
* Risk: the capture machinery assumes whole-step capture today (`_capture_graphs`
  resets the offload cache around each capture; segment capture changes those
  invariants). Effort M.

## Option H2 - staging coordinator (full capture; effort L, risk high)

Generalize the CPU executor's flag handshake so the ring fill runs on a persistent host
thread outside the graph while the graph waits for it device-side. Target end state:
`_finish_mmap_source` keeps graphs for staged decode too.

1. **Selection (captured).** `ensure_experts` keeps its current work and additionally
   (a) async-copies `num_indices` + `src_indices` into mapped-pinned mirrors (an
   in-graph memcpy node; the mirrors replace the stream-synced `_staging_count_host`
   read) and (b) sets a device-visible `ready[layer, bs]` flag (memops store, as the
   CPU executor's producers do).
2. **Coordinator (host, outside the graph).** A persistent daemon thread (modeled on
   the CPU executor's, including its watchdog and core pinning) polls the `ready`
   mirrors, then per MoE layer: reads the miss list from the pinned mirrors (no stream
   sync), runs the ring fill exactly as `_copy_missing_staged` does today
   (`read_rows_into` preads + the same-layer `MADV_WILLNEED` from I), issues the
   pinned->device bounce copy on a private side stream, waits its own event, and sets
   the device-visible `done[layer, bs, chunk]` flag with the chunk's valid row count.
3. **Consumption (captured).** Between the ensure node and the grouped GEMM, the graph
   (a) WAITs on `done` (constant immediate, replay-safe - same primitive as the CPU
   executor) and (b) scatters the bounce into the slot cache with a fixed-grid kernel
   masked by the chunk count read from device memory (fixed shapes; variable miss
   counts are data, not control flow). Misses larger than one ring chunk occupy
   successive chunk slots; the ring depth (buffers x chunks per layer) bounds the
   pipeline and the WAIT provides back-pressure when the coordinator lags.
4. **Hybrid** reuses the same coordinator for its capped fetch share
   (`_decode_hybrid`'s `copy_missing`), and the CPU executor stays as-is (it is already
   captured).

Hard parts, in order of danger:

* **Memory visibility.** The WAIT orders the flag, not the payload. The H2D's data
  must be visible to the consumer kernel at replay time; the CPU executor solved the
  same problem for its input buffers - reuse its mechanism (side stream + event +
  mapped-pinned flag write) verbatim rather than inventing a second protocol.
* **Deadlock classes.** Coordinator lag stalls decode (that is the design - the WAIT
  back-pressures), but a dead coordinator must degrade loudly: port the watchdog
  (poison `done`, surface `raise_if_unhealthy` as a step error). Teardown order matters:
  `destroy_cuda_graphs` / runtime rebuild / engine close must stop the coordinator
  after the graphs stop referencing its flags (`cpu_executor` already has the
  weakref-based teardown pattern to copy).
* **Capture-time cold pass.** `_capture_graphs` deliberately resets the offload cache
  so capture exercises cold-cache expert copies; with H2, capture itself drives the
  coordinator protocol (the dummy pass must complete the handshake or capture hangs).
  The capture loop needs the coordinator started before capture and the reset
  semantics re-checked per bs.
* **Batch-size fan-out.** Flag slots per (layer, bs) like `_flag_slots`; the mirrors
  are bs-sized; today's per-bs capture loop makes this natural but multiplies the
  flag footprint (48 layers x bs list).
* **Replay determinism.** Replay re-executes the ensure kernel with the step's real
  routing, so the coordinator sees a fresh miss list every replay - the same property
  the LRU kernel already relies on. The grouped GEMM then reads whatever the scatter
  landed; no new hazard beyond the visibility item above.

Effort L (3+ days): coordinator lifecycle, protocol, capture integration, teardown,
hybrid share, tests. A fallback to today's eager path must remain one flag away
(`--moe-staged-graphs off`), and capture failure should degrade to eager, not abort
serving.

## Files (H2)

* `moe/offload_kernels.py` - `lru_ensure`/`ensure_experts_hybrid`: mirror writes + the
  ready flag.
* `moe/offload_cache.py` - pinned mirrors, chunk flags, the scatter kernel, ring
  re-shaping from "fill+copy loop" to "coordinator-owned chunks"; `_copy_missing_staged`
  remains as the eager fallback.
* `moe/cpu_executor.py` - extract the flag-sync coordinator + watchdog into a shared
  helper (both consumers), or template H2's coordinator on it directly.
* `engine/graph.py` - capture protocol (WAIT + scatter nodes), coordinator start/stop,
  the cold-pass handshake.
* `engine/engine.py` - `_finish_mmap_source` keeps graphs when staged capture is on.
* `server/args.py` + `engine/config.py` - `--moe-staged-graphs {off,segment,coordinator}`
  (default off until H2's numbers land).

## Test strategy

* **Protocol unit**: ready/done handshake; watchdog poisons a stuck flag; teardown
  stops the thread; a poisoned `err` unblocks the stream and raises.
* **Equivalence**: a captured two-layer staged decode replayed with a recorded
  (`layer, miss-list) script must produce bit-identical slot-cache contents and outputs
  to the eager `_copy_missing_staged` on the same script (extend
  `tests/moe/test_expert_store.py`, which already has the mmap staged fixtures).
* **Chunk edge cases**: 0 misses, 1 chunk, multi-chunk spanning both ring buffers,
  misses > ring depth (back-pressure), reset() between replays, `destroy_cuda_graphs`
  with a live coordinator, capture-time cold pass per bs.
* **Perf gate**: M0's A/B re-run after H2; the merge criterion is a measured tok/s win
  on the real model, not a microbenchmark (the store-side benches cannot see this).

## Milestones

1. **M0 - measure and decide** (S): the A/B + profile above. H proceeds only if the
   eager overhead term is worth its effort; otherwise this document's verdict is
   "stay eager, the warning is correct".
2. **M1 - H1 segment capture** (M): only if M0 shows launch overhead dominant and H2
   is rejected (e.g. capture machinery risk is not acceptable).
3. **M2 - H2 coordinator for offload decode** (L, high risk): flag, fallback, tests.
4. **M3 - hybrid fetch share** (M): `_decode_hybrid` moves onto the coordinator.

## Risks

* H2's deadlock/visibility classes are the same ones the CPU executor already handles;
  the risk is importing that machinery into the graph-capture path, which has more
  lifecycle events (capture, replay, reset, rebuild, teardown).
* A coordinator thread adds CPU jitter next to the GIL and torch's pool; pin it and
  budget a core like the CPU executor does.
* Multi-GPU (TP) capture is untouched by this plan; H2 assumes single-GPU decode
  capture as the current code does.
* If J-style hot banks ever become load-relevant (network stores), H2's coordinator is
  where their sequential preads would be issued; the two designs compose (the
  coordinator just calls `read_rows_into`).
* Out of scope: prefill (never captured), K (`cudaHostRegister`, which would remove the
  ring entirely and obsolete H2 for the pinned subset - if K ever lands, revisit).