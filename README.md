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
- **Host, after the launch:** block coverage is assembled into glyph masks.

Spatial model sets (a cell split into tiles, each with its own network) are
evaluated in the same launch: tiles are cut into equal pixel blocks, and
every block is a group of channels (`--dense` uses block-diagonal convs
instead of grouped convs).

With two input buffers, the host fills batch k+1 on a second thread while
the NPU runs batch k.

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
- `bench/rknn_probe.c`: loads any `.rknn` model, times it and prints the
  runtime's per-operator profile (`PERF=1`).

## Build and run

Export on a PC with RKNN-Toolkit2 (fp16, no quantization):

    PYTHONPATH=python python -m neuronte_npu.convert --manifest MANIFEST \
        --case braille48 --cell 16x32 --batch 64 --out models/

Build on the board, against the reference library and the RKNN header:

    make NEURONTE_DIR=/path/to/neuronte RKNN_INCLUDE=/path/to/rknn/include

Run:

    build/neuronte_npu_bench --meta models/X.meta --rknn models/X.rknn \
        --models DIR_WITH_NRT_FILES --runtime /path/to/librknnrt.so --cpu 4

`--mode cpu` runs the reference CPU path for comparison. `--no-fallback`
makes the run fail when the NPU cannot be used.

## Compatibility notes

Found on an RK3588 board with RKNPU driver 0.9.3:

- Runtime 2.3.2 (with toolkit 2.3.2 models) works. Runtimes 2.5.0, 2.0.0b0
  and 1.6.0 crash in `rknn_run` on simple convolutions with this driver.
- Only NPU core 0 accepts jobs; core masks that include cores 1 or 2 time
  out (`failed to submit`) and the driver resets the NPU.
- An `Add` that broadcasts along the batch axis computes wrong values, and
  a fused `AddRelu` either times out or is wrong. The host layer-1 form
  avoids both.
- Runtime 2.3.2 accepts only NHWC source data for 4-D float inputs through
  `rknn_inputs_set`; the backend writes native buffers directly.

## Licence

MIT. See [LICENSE](LICENSE).
