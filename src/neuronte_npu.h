/* neuronte-npu: evaluate Neuronte distance networks on a Rockchip NPU.
 *
 * A compiled model covers one model set (one network, or a set of spatial
 * tiles), one cell size and one static batch. It is described by a .meta
 * file written by the exporter and compiled into an .rknn file.
 *
 * When the NPU cannot be used (no RKNPU device, no runtime library, a model
 * built for another platform, or a runtime error at open), the handle can
 * fall back to the reference CPU library with the same tiles. The active
 * backend is always reported; fallback results are CPU results.
 */
#ifndef NEURONTE_NPU_H
#define NEURONTE_NPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct neuronte_npu neuronte_npu;

enum neuronte_npu_backend {
    NEURONTE_NPU_RKNN = 1,         /* inference on the NPU */
    NEURONTE_NPU_CPU_FALLBACK = 2, /* the NPU was unavailable: reference CPU library */
    NEURONTE_NPU_CPU = 3,          /* the CPU path was requested explicitly */
};

enum neuronte_npu_flags {
    NEURONTE_NPU_ALLOW_FALLBACK = 1, /* fall back to the CPU instead of failing open */
    NEURONTE_NPU_FORCE_CPU = 2,      /* do not try the NPU */
    NEURONTE_NPU_ZERO_COPY = 4,      /* native-layout buffers, no runtime conversion */
    NEURONTE_NPU_NO_PIPELINE = 8,    /* host layer 1: do not overlap the next batch's fill with a run */
};

struct neuronte_npu_options {
    const char *meta_path;   /* required */
    const char *model_dir;   /* directory of the tile .nrt files named in the meta */
    const char *rknn_path;   /* compiled model; may be NULL with FORCE_CPU */
    const char *runtime_lib; /* librknnrt.so path; NULL searches the loader path */
    int flags;
    int core_mask;           /* RKNN core mask; 0 = runtime default */
    unsigned init_flags;     /* extra rknn_init flags (RKNN_FLAG_*), for experiments; normally 0.
                              * Zero-copy handles always add the two DISABLE_FLUSH flags: the
                              * backend syncs its own buffers. */
    float refine_margin;     /* NPU path: recompute on the CPU every tile that has a pixel whose
                              * NPU coverage is within this distance of 0.5 (fp16 can flip
                              * pixels the CPU puts just past the threshold). 0 = off. */
};

/* How an open handle evaluates. */
struct neuronte_npu_info {
    int layer1;         /* 0 CPU path, 1 layer 1 on the NPU, 2 layer 1 on the host (h1 input) */
    int zero_copy;      /* native-layout buffers bound with rknn_set_io_mem */
    int pipeline;       /* host fill of batch k+1 overlaps the NPU run of batch k */
    unsigned init_flags; /* flags given to rknn_init */
    float refine_margin; /* see neuronte_npu_options */
};

/* Per-call stage times in nanoseconds. NPU stages are zero on the CPU path.
 * With the pipeline, `features` (the host fill) overlaps `run`, so the stage
 * sum can exceed `total`. */
struct neuronte_npu_timing {
    uint64_t features;  /* character features, or host layer 1, into the input buffer */
    uint64_t input;     /* rknn_inputs_set or cache sync to device */
    uint64_t run;       /* rknn_run: the NPU executes the batch */
    uint64_t output;    /* rknn_outputs_get or cache sync from device */
    uint64_t assemble;  /* block coverage -> glyph masks (includes fp16 conversion) */
    uint64_t cpu_eval;  /* reference CPU evaluation: the CPU paths, or refined tiles */
    uint64_t total;
    int launches;       /* NPU runs for this call */
    int refined;        /* tiles recomputed on the CPU (refine_margin) */
};

int neuronte_npu_open(const struct neuronte_npu_options *opt, neuronte_npu **out, char *reason,
                      size_t reason_len);
void neuronte_npu_close(neuronte_npu *n);

int neuronte_npu_backend(const neuronte_npu *n);
const char *neuronte_npu_backend_name(int backend);
const char *neuronte_npu_fallback_reason(const neuronte_npu *n);
int neuronte_npu_batch(const neuronte_npu *n);
int neuronte_npu_cell(const neuronte_npu *n, int *w, int *h);
/* "api;driver" versions when the NPU is active, otherwise "". */
const char *neuronte_npu_versions(const neuronte_npu *n);
int neuronte_npu_info(const neuronte_npu *n, struct neuronte_npu_info *info);

/* Coverage for `count` scalars into out[count * cell_w * cell_h], row-major
 * per glyph, the same values neuronte_coverage gives for each tile. On the
 * NPU, counts above the batch run in several launches and a short final
 * batch is padded: a launch always computes a full batch. */
int neuronte_npu_coverage(neuronte_npu *n, const uint32_t *scalars, int count, float *out,
                          struct neuronte_npu_timing *timing);

#ifdef __cplusplus
}
#endif

#endif
