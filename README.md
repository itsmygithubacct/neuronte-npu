# neuronte-npu

An NPU implementation of [Neuronte](https://github.com/itsmygithubacct/neuronte),
with a benchmark that compares it against Neuronte's standard CPU
implementation.

Neuronte is a small neural network whose weights are a font. Its C library
evaluates the network on the CPU to draw glyphs into a pixel buffer. This
project runs the same network on a Rockchip NPU (RK3588 class, through the
RKNN runtime) and measures how the two compare.

Development happens on the `work/rc9-npu` branch, for the RC9 release.

## How it works

A Neuronte distance network is `90 -> H -> H -> H -> 1` with three ReLUs. The
CPU library splits layer 1 into a character part, computed once per glyph,
and position parts, computed once per axis; only layers 2 to 4 run per
sample. This project keeps that split:

- **Host (CPU), per batch:** the character features and layer 1. For each
  glyph, the host adds the character part to a precomputed position term
  and applies the ReLU, writing `h1` straight into the NPU's native fp16
  input buffer (`NC1HWC2`). This is about 1% of the multiply-adds.
- **NPU, one launch per batch:** layers 2 to 4 as 1 x 1 convolutions over a
  `[1, channels, glyphs, samples]` plane, with the coverage slope folded into
  layer 4, then `Clip(0, 1)` and an average over each pixel's samples.
- **Host, after the launch:** each glyph's coverage is gathered from the
  native output through an offset table built when the model is opened.

Spatial model sets (a cell split into tiles, each with its own network) are
evaluated in the same launch: tiles are cut into equal pixel blocks, and
every block is a group of channels (`--dense` uses block-diagonal convs
instead of grouped convs).

With two input buffers, the host fills batch k+1 on a second thread, and
cleans it from the CPU caches, while the NPU runs batch k.

Models compile in fp16, or in int8 (`--dtype w8a8`), which halves the
NPU's memory traffic. The NPU is memory-bound on these networks, so int8
convolutions run about 1.8× faster, and whole runs were 1.27× faster than
fp16 (median over 45 configurations). Its error depends on the model:
screen each quantized model with `--sim-check` and score it against the CPU
before use. `refine_margin` makes the backend recompute, on the CPU, every
tile with a pixel near the 0.5 threshold. A model can carry its own margin
(a `refine` line in its `.meta`, written by `convert.py --refine`), which the
backend applies unless the caller sets one.

## Results

On an Orange Pi 5 Plus (RK3588, NPU core 0, host work on one Cortex-A76),
against the reference CPU library on the same core, over 45 model
configurations (15 model sets, each at 8×16, 16×32 and 48×96) and all 256
Braille patterns: fp16 was 7.8–21.7× faster (median 11.4×) and int8 9.6–21.3×
(median 15.1×). Every configuration had a form whose binary output matched
the CPU library exactly (fp16, int8, or fp16 with refinement), at 2.3–21.7×
(median 13.9×). `tools/choose-forms.py` picks that form for each
configuration from a benchmark run and assembles the set: models, their
networks, `forms.json` with the measurements, and `SHA256SUMS`. On boards
without an NPU the backend falls back to the CPU library and says so.

[docs/RKNPU.md](docs/RKNPU.md) collects what was learned about the RK3588
NPU: driver and core behaviour, data layouts and cache handling, which
operators run well or wrongly, precision, and how to measure.

Other layer-1 forms (`graph.py`, `--l1`) keep layer 1 on the NPU. They are
kept for comparison and to document runtime behaviour; see the docstring of
`graph.build`.

## Layout

- `python/neuronte_npu/`: exporter. `nrt.py` reads `.nrt` weights (and routes
  a bundle's range table), `layout.py` cuts a cell into tiles and blocks,
  `graph.py` builds the ONNX graph, `cpu.py` calls the reference library,
  `convert.py` writes the ONNX, `.meta`, sidecar and compiled `.rknn` files.
- `src/neuronte_npu.[ch]`: the C backend. It loads the RKNN runtime with
  `dlopen`, checks that an RKNPU device exists, and otherwise falls back to
  the reference CPU library, reporting that it did so (`cpu-fallback` and a
  reason) or failing if fallback is not allowed.
- `bench/neuronte_npu_bench.c`: benchmark. Open time, first call, warm
  rounds with per-stage times, single-glyph latency, hardware snapshots, and
  a coverage dump for quality checks. One JSON object per run.
- `bench/rknn_probe.c`: loads any `.rknn` model, prints its tensors and
  memory use, and times it, through `rknn_inputs_set` or with zero-copy
  native buffers (`-z`); `-p` prints the runtime's per-operator profile.
- `tools/`: `npu-discover.sh` (read-only report on a board's NPU: device,
  driver, usable cores, SRAM, access), `npu-run.sh` (run with the NPU
  device groups), `run-set.sh` (a set of CPU and NPU benchmarks back to back
  on one core, optionally under a lock shared with other benchmark users,
  recording each run's DDR/NPU clock time-in-state), `summarize.py` (medians
  from benchmark JSON, and the mean DDR clock per run), `compare.py`
  (coverage dumps against the CPU and teacher masks), `choose-forms.py`
  (per configuration, the fastest NPU form whose binary output is identical
  to the CPU library's; `--assemble` copies the chosen forms into a set).

## Build and run

Export on a PC with RKNN-Toolkit2 (fp16 by default; `--dtype w8a8
--quant-algorithm mmse --sim-check` for int8 with a simulator quality
record):

    PYTHONPATH=python python -m neuronte_npu.convert --manifest MANIFEST \
        --case braille48 --cell 16x32 --batch 64 --out models/

Build on the board, against the reference library and the RKNN header:

    make NEURONTE_DIR=/path/to/neuronte RKNN_INCLUDE=/path/to/rknn/include

Run:

    build/neuronte_npu_bench --meta models/X.meta --rknn models/X.rknn \
        --models DIR_WITH_NRT_FILES --runtime /path/to/librknnrt.so --cpu 4

`--mode cpu` runs the reference CPU path for comparison. `--no-fallback`
makes the run fail when the NPU cannot be used. `--refine MARGIN` turns on
CPU refinement near the threshold (0, the default, uses the model's own
margin; a negative margin turns refinement off).

Check a board first with `tools/npu-discover.sh`. Python tests:
`make check PYTHON=... NEURONTE_MANIFEST=... NEURONTE_LIB=...` (tests that
need the manifest or library skip without them).

## Compatibility notes

Found on an RK3588 board with RKNPU driver 0.9.3:

- Runtime 2.3.2 (with toolkit 2.3.2 models) works. Runtimes 2.5.0, 2.0.0b0
  and 1.6.0 crash in `rknn_run` on simple convolutions with this driver.
- The driver registered only core 0's completion interrupt, so jobs on
  cores 1 and 2 time out after 6 s and the driver resets the NPU. The
  runtime's default core mask sends a second concurrent context to another
  core. With `core_mask` 0 the backend counts the registered interrupts and
  pins core 0 when fewer than three are registered.
- No SRAM is configured for the NPU (`CONFIG_ROCKCHIP_RKNPU_SRAM` unset).
- An `Add` that broadcasts along the batch axis computes wrong values, and
  a fused `AddRelu` either times out or is wrong. The host layer-1 form
  avoids both.
- Runtime 2.3.2 accepts only NHWC source data for 4-D float inputs through
  `rknn_inputs_set`; the backend writes native buffers directly.

## Licence

MIT. See [LICENSE](LICENSE).
