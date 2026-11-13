# neuronte-npu

An NPU implementation of [Neuronte](https://github.com/itsmygithubacct/neuronte),
with a benchmark that compares it against Neuronte's standard CPU
implementation.

Neuronte is a small neural network whose weights are a font. Its C library
evaluates the network on the CPU to draw glyphs into a pixel buffer. This
project runs the same network on an NPU and measures how the two compare.

## Status

New. There is no code yet. The work targets the RC9 release. Development happens
on the `work/rc9-npu` branch.

## Plan

- Run Neuronte's glyph network on an NPU, using the same weights as the CPU
  library.
- Check that the NPU output matches the CPU implementation for the same glyphs
  and cell sizes.
- Benchmark the NPU and CPU paths under paired conditions, and record the exact
  revisions, weights and hardware behind each result.

## Licence

MIT. See [LICENSE](LICENSE).
