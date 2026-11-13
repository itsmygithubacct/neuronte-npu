# Using the Rockchip NPU (RK3588) for Neuronte

What this project found out about running small dense networks on the
RK3588 NPU, and how the backend uses it. Measurements are from an Orange Pi
5 Plus (RK3588, 16 GB) with RKNPU driver 0.9.3 on a vendor 6.1 kernel,
runtime and toolkit 2.3.2, NPU at 1 GHz, while other services ran on the
board. Other kernels and driver versions behave differently in places;
`tools/npu-discover.sh` shows what a given board has.

## The stack

- **Kernel driver (RKNPU).** Exposes the NPU as a DRM device
  (`/dev/dri/renderD*`, `/dev/dri/card*`), or `/dev/rknpu` on older
  kernels. Its debugfs (`/sys/kernel/debug/rknpu/`) has the version, load per
  core, power, frequency and voltage.
- **Runtime (`librknnrt.so`).** Loads a compiled `.rknn` model and runs it.
  The backend loads it with `dlopen`, so the same binary runs (on the CPU)
  where there is no NPU or no runtime.
- **RKNN-Toolkit2 (PC).** Converts ONNX into `.rknn` for a target SoC, in
  fp16 or quantized, and has a simulator that runs the compiled model on the
  PC.

Use a toolkit and runtime of the same version. With driver 0.9.3, runtime
2.3.2 worked; 2.5.0, 2.0.0b0 and 1.6.0 crashed in `rknn_run` on simple
convolutions.

## Before using a board

Run `tools/npu-discover.sh [path/to/librknnrt.so]`. It changes nothing; with
passwordless `sudo` it also reads debugfs and the kernel log. Check:

- **An RKNPU device is bound.** RK3399 boards (no NPU) can still load an
  `rknpu` module; with no device the backend reports `cpu-fallback`.
- **How many cores can complete jobs** (next section).
- **Device access.** The DRM nodes belong to the `render` and `video` groups.
  `tools/npu-run.sh CMD` runs a command with those two groups added (via
  `sudo setpriv`, no capabilities), without changing group membership.
- **SRAM.** `RKNN_FLAG_ENABLE_SRAM` needs `CONFIG_ROCKCHIP_RKNPU_SRAM` and a
  `rockchip,sram` property on the NPU node. The board tested has neither, and
  `RKNN_QUERY_MEM_SIZE` reports 0 bytes of SRAM.

## Cores

The RK3588 NPU has three cores. Each signals job completion with its own
interrupt (`npu0_irq`, `npu1_irq`, `npu2_irq` in the device tree). On the
kernel tested, the driver registered only core 0's interrupt; the other two
lines carry only the IOMMU handler. In `/proc/interrupts`:

    GICv3 142 Level  fdab9000.iommu, fdab0000.npu
    GICv3 143 Level  fdab9000.iommu
    GICv3 144 Level  fdab9000.iommu

Consequences, all measured:

- A job on core 1 or 2 finishes in hardware (the core's status register
  shows completion), but the driver never sees it. After 6 s it logs `job
  timeout` and soft-resets the NPU.
- The unhandled interrupts eventually trip the kernel's spurious-interrupt
  check (`irq 42: nobody cared`, `Disabling IRQ #42`), which disables those
  lines, and with them those cores' IOMMU fault interrupts, until reboot.
- The runtime's default core mask (`RKNN_NPU_CORE_AUTO`) puts a second
  concurrent context on another core: two processes each running a model
  with the default mask gave one 6 s timeout and an NPU reset.

So on such a kernel, use core 0 only. The backend does this by itself. With
`core_mask` 0 it counts the `/proc/interrupts` lines whose handler list
names the RKNPU device, and when fewer than three cores are registered it
sets `RKNN_NPU_CORE_0`. The count and mask are reported in
`neuronte_npu_info` (`npu_cores`, `core_mask`) and in the benchmark JSON.

Why, from the driver source: rknpu 0.9.3 decides whether the device tree
names its interrupts with `platform_get_resource_byname(pdev,
IORESOURCE_IRQ, "npu0_irq")`. Since Linux 5.19 the device-tree core no
longer adds interrupt resources to platform devices (upstream commit
`a1a2b7125e10`), so on a 6.1 kernel that lookup always fails and the driver
falls back to requesting interrupt 0 alone, with the core-0 handler. No
device-tree change helps. rknpu 0.9.4 fixed it (on 6.1 and later it looks
the interrupts up with `platform_get_irq_byname`, falling back to
`platform_get_irq`), and later versions keep the fix. The driver is usually
built into the vendor kernel, so using all three cores means a kernel with
rknpu 0.9.4 or later (check `/sys/kernel/debug/rknpu/version`, and that all
three interrupt lines name `fdab0000.npu`). Rockchip's guide expects little
from a multi-core mask on small networks; one context per core is the way to
use the extra cores.

## Getting data in and out

- **Native layout, zero copy.** Query `RKNN_QUERY_NATIVE_INPUT_ATTR` and
  `RKNN_QUERY_NATIVE_OUTPUT_ATTR`, allocate with `rknn_create_mem`, bind
  with `rknn_set_io_mem` (`pass_through = 1`), and write the NPU's own
  layout: `NC1HWC2`, with C2 = 8 for fp16 and 16 for int8. This avoids the
  runtime's conversion. With `rknn_inputs_set`, 2.3.2 accepts only NHWC
  source data for 4-D float inputs: it logs an error for NCHW but returns 0.
- **Cache maintenance.** Buffers from `rknn_create_mem` are cacheable. Init
  with `RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE |
  RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE` and call `rknn_mem_sync`
  yourself (`TO_DEVICE` after writing an input, `FROM_DEVICE` before reading
  an output). Cleaning a 12 MB input takes about 0.7 ms; do it on the thread
  that wrote the buffer, so it overlaps the previous launch.
- **Two input buffers.** Rebinding the input with `rknn_set_io_mem` is cheap.
  With two buffers, a second thread prepares and syncs batch k+1 while the
  NPU runs batch k. The first batch of each call cannot overlap anything.
- **Reading the output.** The native output is `NC1HWC2` as well, so the
  pixels of one glyph are scattered: neighbours are C2 elements apart, and
  blocks sit in different C1 planes. For every native layout the offset of
  an element is linear in the batch index (the glyph), so the backend
  computes, once at open, where each pixel of the cell is for glyph 0, and
  assembles every glyph with one gather over the cell. A loop over blocks
  and rows was as fast for wide blocks but up to 1.3× slower overall for
  models with many narrow blocks, where the per-row overhead dominates.
- **Non-cacheable memory.** `rknn_create_mem2` with
  `RKNN_FLAG_MEMORY_NON_CACHEABLE` gave a buffer on which `memset` raised
  SIGBUS (likely libc's cache-line zeroing, which non-cacheable memory does
  not allow); the backend does not use it.

## What runs well

The NPU is memory-bound on these networks. In the per-operator profile, the
48→48 1×1 convolutions use 9–11% of the MAC units, and their DDR cycles are
about five times their NPU cycles. Time follows bytes moved:

- Every layer is a full pass over its activation tensor, so removing layers
  (here, computing layer 1 on the host) gains more than anything else.
- A 48→1 convolution costs nearly as much as a 48→48 one, because reading its
  input dominates.
- **int8 halves activation traffic.** The layer-2/3 convolutions run 1.8×
  faster (per-operator profile). Over whole benchmark runs, host work
  included, int8 was 0.83–1.44× as fast as fp16 (median 1.27×) and faster
  in 38 of 45 configurations. The other seven are small or narrow models
  whose host work per glyph (filling the input, assembling the output) is
  about as long as the NPU's run time or longer, so a faster launch does
  not help.
- **With one host core the host can be the limit.** Quantizing the input on
  the host costs more than converting it to fp16. When the zero point is
  −128 (the usual case for a ReLU output), rounding without the ReLU,
  saturating to [0, 255] unsigned and flipping the top bit gives the same
  bytes with fewer operations; the backend does this.
- `AveragePool` is relatively expensive for the little data it reads.
  Measured on whole 16×32 launches of three models (median of three
  rounds): leaving the averaging to the host takes 9.7–11.2% off an fp16
  launch and 4.2–5.7% off an int8 one. A strided depthwise convolution
  doing the same averaging takes 3.9–8.7% off fp16 and changes int8 by
  −4.2% to +2.1%. At 8×16 (one model) both forms are within 4% of
  `AveragePool`, except the convolution in int8, 11.7% slower. The backend
  uses `AveragePool`; `convert.py --pool conv` builds the convolution
  form.
- The DDR clock follows load (see Measuring), so a short workload can run
  entirely at the lowest memory clock.

Operators and patterns to avoid on driver 0.9.3 / runtime 2.3.2:

- `Add` that broadcasts along the batch (height) axis computes wrong values,
  whether the broadcast operand is a constant or an input.
- A fused `AddRelu` either fails to submit (timeout) or is wrong.
- Grouped convolutions with few large groups (two groups of 24 channels)
  are lowered to Split, per-group Conv, Concat and an unfused ReLU, 3.2–3.5×
  slower than the block-diagonal dense form. With 4–64 groups the two forms
  are within 5% in most cases.
- A `ConvTranspose` with a long 1×M kernel compiles into huge buffers (b64
  asked for a 3.3 GB NPU buffer; b16 ran at 58 ms per glyph).
- A graph whose output is constant does not compile.

## Precision

- **fp16** (no quantization) matches the CPU to at most 4.4e-2 in coverage.
  That is enough to flip pixels that lie just past the 0.5 threshold, which
  happens in cells with one sample per pixel (48×96). The backend's
  `refine_margin` recomputes, on the CPU, every tile with a pixel within the
  margin of 0.5. A model that needs it carries the margin in its `.meta`
  (`refine 0.05`), so the backend refines by default; a caller's positive
  margin replaces it and a negative one turns it off.
- **int8 (w8a8)** quality depends on the model. `convert.py --dtype w8a8`
  calibrates on the h1 inputs of all 256 Braille scalars, which is the whole
  input domain of these models. `--quant-algorithm mmse` was more accurate
  than `normal` for most models, but not all.
- **The toolkit simulator predicts the board's int8 error closely**
  (largest error 0.0146 simulated vs 0.0146 measured, 0.171 vs 0.171,
  0.030 vs 0.027; over 45 models, within 0.004 for 37 and within 0.014 for
  all). `convert.py --sim-check` records the simulated error and
  flips over every Braille scalar, so a quantized model can be screened
  before it reaches a board.
- With a domain this small (256 inputs), an evaluation over all of it is
  exhaustive: zero flips measured means identical binary output, not an
  estimate.

## Measuring

- `build/rknn_probe -z -m 1 LIB MODEL` times `rknn_run` with zero-copy
  native buffers, as the backend uses them. It also prints every tensor's
  attributes and the model's memory use. `-p` adds the runtime's
  per-operator profile (MAC use, DDR/NPU cycles, bytes moved). Perf mode
  runs operators one at a time, so its times are larger than real runs; use
  it for proportions.
- `tools/run-set.sh SPEC OUT` runs CPU and NPU benchmarks back to back on one
  pinned core, keeping every JSON and coverage dump. `tools/summarize.py`
  gives medians per glyph, and `tools/compare.py` compares dumps with the
  CPU and with teacher masks. `tools/choose-forms.py OUT` picks, per
  configuration, the fastest NPU run with zero binary flips against that
  configuration's CPU run. Where two forms are within a few per cent, the
  choice can change between campaigns (see run-to-run variance below); both
  are exact, so either is correct.
- A batch-1 launch costs at least about 31 µs, so single-glyph latency gains
  less than throughput.
- **The DDR clock moves.** The memory controller has its own on-demand
  governor (`/sys/class/devfreq/dmc`: 528 MHz idle, up to 2112 MHz on the
  board tested). A benchmark run of a small 8×16 model lasts well under a
  second; some ran at a mean of 528 MHz throughout, others at about 1 GHz,
  while 48×96 runs averaged 1.5–2.0 GHz. These networks are memory-bound,
  so compare times only from paired, interleaved runs, and record the
  clock: `run-set.sh` keeps each run's devfreq time-in-state (`.freq`) and
  `summarize.py` reports the mean DDR clock per run. The measurements here
  leave the governor alone, because it is shared with everything else on
  the board.
- **Short launches vary from run to run.** The same fp16 code, measured in
  two campaigns 18 minutes apart, gave per-configuration ratios of 0.83 to
  1.39 (5 of 45 outside ±10%, all 8×16 or 16×32 models, mostly in the
  NPU run time itself), while the CPU library stayed within 0.93–1.02.
  Judge a change only by runs interleaved with the previous build (ABBA).
- Small changes to host code show up on small models first: there the host
  takes most of a launch's time. Check a change on the smallest and largest
  cells.
