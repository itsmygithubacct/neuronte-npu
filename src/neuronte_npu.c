/* neuronte-npu backend. See neuronte_npu.h. */
#define _GNU_SOURCE
#include "neuronte_npu.h"

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "neuronte.h"
#include "rknn_api.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define HAVE_NEON_F16 1
#endif

#define CODE 64
#define NMASK 43
#define MAX_TILES 64
#define MAX_BLOCKS 4096
#define BRAILLE_FIRST 0x2800u

#if defined(__aarch64__) || defined(__arm__)
typedef __fp16 half;
#else
typedef _Float16 half;
#endif

typedef int (*fn_init)(rknn_context *, void *, uint32_t, uint32_t, rknn_init_extend *);
typedef int (*fn_destroy)(rknn_context);
typedef int (*fn_query)(rknn_context, rknn_query_cmd, void *, uint32_t);
typedef int (*fn_inputs_set)(rknn_context, uint32_t, rknn_input *);
typedef int (*fn_run)(rknn_context, rknn_run_extend *);
typedef int (*fn_outputs_get)(rknn_context, uint32_t, rknn_output *, rknn_output_extend *);
typedef int (*fn_outputs_release)(rknn_context, uint32_t, rknn_output *);
typedef int (*fn_set_core_mask)(rknn_context, rknn_core_mask);
typedef rknn_tensor_mem *(*fn_create_mem)(rknn_context, uint32_t);
typedef int (*fn_destroy_mem)(rknn_context, rknn_tensor_mem *);
typedef int (*fn_set_io_mem)(rknn_context, rknn_tensor_mem *, rknn_tensor_attr *);
typedef int (*fn_mem_sync)(rknn_context, rknn_tensor_mem *, rknn_mem_sync_mode);

struct api {
    void *dl;
    fn_init init;
    fn_destroy destroy;
    fn_query query;
    fn_inputs_set inputs_set;
    fn_run run;
    fn_outputs_get outputs_get;
    fn_outputs_release outputs_release;
    fn_set_core_mask set_core_mask;
    fn_create_mem create_mem;
    fn_destroy_mem destroy_mem;
    fn_set_io_mem set_io_mem;
    fn_mem_sync mem_sync;
};

struct tile {
    int mask; /* -1 passes the scalar through */
    int x, y, w, h;
    char file[256];
    uint32_t masks[NMASK];
    neuronte *cpu;
};

struct block {
    int tile, x, y;
};

struct neuronte_npu {
    int backend;
    char reason[512];
    char versions[520];
    int cell_w, cell_h, samples, batch, hidden;
    int ntiles, nblocks, bw, bh;
    struct tile tiles[MAX_TILES];
    struct block *blocks;
    int in_c, out_c, out_p; /* input channels; output channels (blocks); pixels per block */
    /* NPU */
    struct api api;
    rknn_context ctx;
    int have_ctx;
    int zero_copy;
    float *in_f32;  /* NCHW float input for rknn_inputs_set */
    float *out_f32; /* NCHW float output from rknn_outputs_get */
    rknn_tensor_attr in_attr, out_attr; /* native attributes, zero-copy mode */
    rknn_tensor_mem *in_mem, *out_mem;
    int code_index;
    /* position term as a second input, filled once at open (graph "input" form) */
    char pos_file[256];
    int pos_rows, pos_m;
    int pos_index;
    rknn_tensor_attr pos_attr;
    rknn_tensor_mem *pos_mem;
    /* layer 1 on the host (graph "host" form): h1 = ReLU(code_w . features +
     * code_b + P) written into the native input buffer each launch */
    char l1_file[256];
    int l1_rows, l1_k, l1_m;
    float *l1_w;   /* [rows][k] */
    float *l1_b;   /* [rows] */
    float *l1_pn;  /* P in native order: [C1][W][C2], zero beyond rows */
    float *l1_c;   /* [batch][C1 * C2] character part of each glyph */
    float *l1_fv;  /* [ntiles][CODE] features of one glyph */
    rknn_tensor_mem *in_mem2; /* second input buffer for the pipeline */
    rknn_tensor_mem *bound;   /* input buffer bound to the context */
    unsigned init_flags;      /* flags given to rknn_init */
    float refine_margin;      /* recompute ambiguous tiles on the CPU */
    float meta_refine;        /* the model's own margin ("refine" in the meta), 0 = none */
    int core_mask;            /* applied core mask, 0 = runtime default */
    int npu_cores;            /* cores with a registered completion interrupt, -1 unknown */
    uint32_t *asm_off;        /* zero-copy: native output offset of each cell pixel of glyph 0 */
    size_t asm_glyph;         /* zero-copy: native output elements from one glyph to the next */
    /* CPU */
    float *cpu_buf;
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void say(char *buf, size_t len, const char *fmt, ...)
{
    va_list ap;
    if (!buf || !len)
        return;
    va_start(ap, fmt);
    vsnprintf(buf, len, fmt, ap);
    va_end(ap);
}

const char *neuronte_npu_backend_name(int backend)
{
    switch (backend) {
    case NEURONTE_NPU_RKNN: return "npu-rknn";
    case NEURONTE_NPU_CPU_FALLBACK: return "cpu-fallback";
    case NEURONTE_NPU_CPU: return "cpu";
    default: return "none";
    }
}

/* ---- meta file ---------------------------------------------------------- */

static int parse_meta(neuronte_npu *n, const char *path, char *err, size_t err_len)
{
    FILE *f = fopen(path, "r");
    char line[8192];
    int version = 0, seen_blocks = 0, nb = 0, nt = 0;
    if (!f) {
        say(err, err_len, "cannot open meta %s: %s", path, strerror(errno));
        return -1;
    }
    while (fgets(line, sizeof line, f)) {
        char key[32];
        if (sscanf(line, "%31s", key) != 1)
            continue;
        if (!strcmp(key, "neuronte-npu-meta"))
            sscanf(line, "%*s %d", &version);
        else if (!strcmp(key, "cell"))
            sscanf(line, "%*s %d %d", &n->cell_w, &n->cell_h);
        else if (!strcmp(key, "samples"))
            sscanf(line, "%*s %d", &n->samples);
        else if (!strcmp(key, "batch"))
            sscanf(line, "%*s %d", &n->batch);
        else if (!strcmp(key, "hidden"))
            sscanf(line, "%*s %d", &n->hidden);
        else if (!strcmp(key, "host-l1")) {
            if (sscanf(line, "%*s %255s %d %d %d", n->l1_file, &n->l1_rows, &n->l1_k, &n->l1_m) != 4 ||
                n->l1_rows < 1 || n->l1_k < 1 || n->l1_m < 1)
                goto bad;
        }
        else if (!strcmp(key, "pos")) {
            if (sscanf(line, "%*s %255s %d %d", n->pos_file, &n->pos_rows, &n->pos_m) != 3 || n->pos_rows < 1 ||
                n->pos_m < 1)
                goto bad;
        }
        else if (!strcmp(key, "refine")) {
            /* the margin this model needs for output identical to the CPU's */
            if (sscanf(line, "%*s %f", &n->meta_refine) != 1 || !(n->meta_refine >= 0.0f && n->meta_refine < 0.5f))
                goto bad;
        } else if (!strcmp(key, "tiles")) {
            sscanf(line, "%*s %d", &n->ntiles);
            if (n->ntiles < 1 || n->ntiles > MAX_TILES)
                goto bad;
        } else if (!strcmp(key, "tile")) {
            struct tile *t;
            int i, off = 0, index;
            char *p = line;
            if (nt >= n->ntiles)
                goto bad;
            t = &n->tiles[nt];
            if (sscanf(p, "%*s %d %d %d %d %d %d %255s %*s%n", &index, &t->mask, &t->x, &t->y, &t->w, &t->h,
                       t->file, &off) != 7 || index != nt)
                goto bad;
            p += off;
            for (i = 0; i < NMASK; i++) {
                unsigned long v;
                char *end;
                errno = 0;
                v = strtoul(p, &end, 10);
                if (end == p || errno || v == 0 || v >= 1ul << 21)
                    goto bad;
                t->masks[i] = (uint32_t)v;
                p = end;
            }
            nt++;
        } else if (!strcmp(key, "blocks")) {
            sscanf(line, "%*s %d %d %d", &n->nblocks, &n->bw, &n->bh);
            if (n->nblocks < 1 || n->nblocks > MAX_BLOCKS || n->bw < 1 || n->bh < 1)
                goto bad;
            n->blocks = calloc((size_t)n->nblocks, sizeof *n->blocks);
            if (!n->blocks)
                goto bad;
            seen_blocks = 1;
        } else if (!strcmp(key, "block")) {
            int g;
            if (!seen_blocks || nb >= n->nblocks)
                goto bad;
            if (sscanf(line, "%*s %d %d %d %d", &g, &n->blocks[nb].tile, &n->blocks[nb].x, &n->blocks[nb].y) != 4 ||
                g != nb || n->blocks[nb].tile < 0 || n->blocks[nb].tile >= n->ntiles)
                goto bad;
            nb++;
        }
    }
    fclose(f);
    if (version != 1 || n->cell_w < 1 || n->cell_h < 1 || n->samples < 1 || n->batch < 1 ||
        nt != n->ntiles || nb != n->nblocks || n->nblocks * n->bw * n->bh != n->cell_w * n->cell_h) {
        say(err, err_len, "meta %s is incomplete or inconsistent", path);
        return -1;
    }
    for (nb = 0; nb < n->nblocks; nb++) {
        const struct block *b = &n->blocks[nb];
        const struct tile *t = &n->tiles[b->tile];
        if (b->x < t->x || b->y < t->y || b->x + n->bw > t->x + t->w || b->y + n->bh > t->y + t->h)
            goto badclosed;
    }
    n->in_c = n->ntiles * CODE;
    n->out_c = n->nblocks;
    n->out_p = n->bw * n->bh;
    if (n->pos_file[0] && (n->pos_rows != n->nblocks * n->hidden ||
                           n->pos_m != n->out_p * n->samples * n->samples)) {
        say(err, err_len, "meta %s: position input shape does not match the layout", path);
        return -1;
    }
    if (n->l1_file[0] && (n->pos_file[0] || n->l1_rows != n->nblocks * n->hidden || n->l1_k != n->in_c ||
                          n->l1_m != n->out_p * n->samples * n->samples)) {
        say(err, err_len, "meta %s: host layer-1 shape does not match the layout", path);
        return -1;
    }
    return 0;
bad:
    fclose(f);
badclosed:
    say(err, err_len, "malformed meta %s", path);
    return -1;
}

/* ---- capability --------------------------------------------------------- */

/* Is an RKNPU device bound on this machine? Says what was found either way. */
static int rknpu_device(char *why, size_t why_len)
{
    DIR *d = opendir("/sys/class/drm");
    struct dirent *e;
    int found = 0;
    if (access("/dev/rknpu", F_OK) == 0) {
        say(why, why_len, "/dev/rknpu");
        return 1;
    }
    if (!d) {
        say(why, why_len, "no /sys/class/drm and no /dev/rknpu");
        return 0;
    }
    while ((e = readdir(d))) {
        char link[PATH_MAX], target[PATH_MAX];
        ssize_t len;
        if (strncmp(e->d_name, "renderD", 7) && strncmp(e->d_name, "card", 4))
            continue;
        snprintf(link, sizeof link, "/sys/class/drm/%s/device/driver", e->d_name);
        len = readlink(link, target, sizeof target - 1);
        if (len <= 0)
            continue;
        target[len] = 0;
        if (strstr(target, "RKNPU")) {
            say(why, why_len, "/dev/dri/%s bound to RKNPU", e->d_name);
            found = 1;
            break;
        }
    }
    closedir(d);
    if (!found)
        say(why, why_len, "no DRM device bound to the RKNPU driver and no /dev/rknpu");
    return found;
}

/* How many NPU cores can complete jobs: the number of /proc/interrupts lines
 * whose handler list names the RKNPU platform device (one completion
 * interrupt per core). Some RK3588 kernels register only core 0's, although
 * the device tree lists three; jobs sent to the other cores then finish in
 * hardware but are never seen as done, and the driver times out after
 * seconds and resets the NPU. -1 when it cannot be told. */
static int rknpu_irq_cores(void)
{
    DIR *d = opendir("/sys/class/drm");
    struct dirent *e;
    char name[256] = "";
    if (!d)
        return -1;
    while ((e = readdir(d))) {
        char link[PATH_MAX], target[PATH_MAX];
        ssize_t len;
        if (strncmp(e->d_name, "card", 4))
            continue;
        snprintf(link, sizeof link, "/sys/class/drm/%s/device/driver", e->d_name);
        len = readlink(link, target, sizeof target - 1);
        if (len <= 0)
            continue;
        target[len] = 0;
        if (!strstr(target, "RKNPU"))
            continue;
        snprintf(link, sizeof link, "/sys/class/drm/%s/device", e->d_name);
        len = readlink(link, target, sizeof target - 1);
        if (len > 0) {
            const char *base;
            target[len] = 0;
            base = strrchr(target, '/');
            snprintf(name, sizeof name, "%.200s", base ? base + 1 : target);
        }
        break;
    }
    closedir(d);
    if (!name[0])
        return -1;
    FILE *f = fopen("/proc/interrupts", "r");
    if (!f)
        return -1;
    char line[1024];
    size_t nl = strlen(name);
    int count = 0;
    while (fgets(line, sizeof line, f)) {
        const char *p = line;
        while ((p = strstr(p, name))) {
            char after = p[nl];
            if (p > line && p[-1] == ' ' && (after == '\n' || after == ',' || after == ' ' || after == 0)) {
                count++;
                break;
            }
            p += nl;
        }
    }
    fclose(f);
    return count;
}

static int load_api(struct api *a, const char *lib, char *err, size_t err_len)
{
    memset(a, 0, sizeof *a);
    a->dl = dlopen(lib ? lib : "librknnrt.so", RTLD_NOW | RTLD_LOCAL);
    if (!a->dl) {
        say(err, err_len, "cannot load the RKNN runtime: %s", dlerror());
        return -1;
    }
#define SYM(field, name)                                                         \
    do {                                                                         \
        *(void **)&a->field = dlsym(a->dl, name);                                \
        if (!a->field) {                                                         \
            say(err, err_len, "RKNN runtime lacks %s", name);                    \
            dlclose(a->dl);                                                      \
            a->dl = NULL;                                                        \
            return -1;                                                           \
        }                                                                        \
    } while (0)
    SYM(init, "rknn_init");
    SYM(destroy, "rknn_destroy");
    SYM(query, "rknn_query");
    SYM(inputs_set, "rknn_inputs_set");
    SYM(run, "rknn_run");
    SYM(outputs_get, "rknn_outputs_get");
    SYM(outputs_release, "rknn_outputs_release");
    SYM(set_core_mask, "rknn_set_core_mask");
    SYM(create_mem, "rknn_create_mem");
    SYM(destroy_mem, "rknn_destroy_mem");
    SYM(set_io_mem, "rknn_set_io_mem");
    SYM(mem_sync, "rknn_mem_sync");
#undef SYM
    return 0;
}

static void *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    void *buf = NULL;
    long size;
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) == 0 && (size = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0 &&
        (buf = malloc((size_t)size)) && fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    if (buf)
        *len = (size_t)size;
    return buf;
}

static size_t native_index(const rknn_tensor_attr *a, int c, int h, int w)
{
    if (a->fmt == RKNN_TENSOR_NC1HWC2) {
        size_t c2 = a->dims[4], hh = a->dims[2], ws = a->w_stride ? a->w_stride : a->dims[3];
        return (((size_t)c / c2 * hh + (size_t)h) * ws + (size_t)w) * c2 + (size_t)c % c2;
    }
    if (a->fmt == RKNN_TENSOR_NHWC) {
        size_t ww = a->w_stride ? a->w_stride : a->dims[2], cc = a->dims[3];
        return ((size_t)h * ww + (size_t)w) * cc + (size_t)c;
    }
    {
        size_t hh = a->dims[2], ww = a->w_stride ? a->w_stride : a->dims[3];
        return ((size_t)c * hh + (size_t)h) * ww + (size_t)w;
    }
}

static int native_supported(const rknn_tensor_attr *a)
{
    if (a->type == RKNN_TENSOR_INT8) /* quantized models: affine, native NC1HWC2 only */
        return a->fmt == RKNN_TENSOR_NC1HWC2 && a->n_dims == 5 && a->qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
               a->scale > 0.0f;
    return (a->type == RKNN_TENSOR_FLOAT16 || a->type == RKNN_TENSOR_FLOAT32) &&
           ((a->fmt == RKNN_TENSOR_NC1HWC2 && a->n_dims == 5) ||
            ((a->fmt == RKNN_TENSOR_NHWC || a->fmt == RKNN_TENSOR_NCHW) && a->n_dims == 4));
}

/* `file` from the meta, relative to the compiled model's directory. */
static void sidecar_path(char *path, size_t len, const struct neuronte_npu_options *o, const char *file)
{
    const char *slash = o->rknn_path ? strrchr(o->rknn_path, '/') : NULL;
    if (file[0] != '/' && slash)
        snprintf(path, len, "%.*s/%s", (int)(slash - o->rknn_path), o->rknn_path, file);
    else
        snprintf(path, len, "%s", file);
}

/* Load the host layer-1 data and lay P out in the input's native order. */
static int load_host_l1(neuronte_npu *n, const struct neuronte_npu_options *o, char *err, size_t err_len)
{
    char path[PATH_MAX];
    const rknn_tensor_attr *a = &n->in_attr;
    size_t len = 0, rows = (size_t)n->l1_rows, k = (size_t)n->l1_k, m = (size_t)n->l1_m;
    size_t want = (rows * k + rows + rows * m) * sizeof(float), c1n, c2n, w, j, c1;
    float *data;
    if (a->fmt != RKNN_TENSOR_NC1HWC2 || a->n_dims != 5 || a->dims[2] != (uint32_t)n->batch ||
        a->dims[3] != (uint32_t)n->l1_m || a->dims[1] * a->dims[4] < (uint32_t)n->l1_rows ||
        (a->type != RKNN_TENSOR_FLOAT16 && a->type != RKNN_TENSOR_FLOAT32 && a->type != RKNN_TENSOR_INT8)) {
        say(err, err_len, "host layer 1 needs an NC1HWC2 input of %d x %d x %d (got fmt %d dims %u,%u,%u,%u,%u)",
            n->l1_rows, n->batch, n->l1_m, a->fmt, a->dims[0], a->dims[1], a->dims[2], a->dims[3], a->dims[4]);
        return -1;
    }
    sidecar_path(path, sizeof path, o, n->l1_file);
    data = slurp(path, &len);
    if (!data || len != want) {
        free(data);
        say(err, err_len, "cannot read %zu bytes of host layer-1 data from %s", want, path);
        return -1;
    }
    c1n = a->dims[1];
    c2n = a->dims[4];
    n->l1_w = malloc(rows * k * sizeof(float));
    n->l1_b = malloc(rows * sizeof(float));
    n->l1_pn = calloc(c1n * m * c2n, sizeof(float));
    n->l1_c = calloc((size_t)n->batch * c1n * c2n, sizeof(float));
    n->l1_fv = malloc((size_t)n->ntiles * CODE * sizeof(float));
    if (!n->l1_w || !n->l1_b || !n->l1_pn || !n->l1_c || !n->l1_fv) {
        free(data);
        say(err, err_len, "out of memory");
        return -1;
    }
    memcpy(n->l1_w, data, rows * k * sizeof(float));
    memcpy(n->l1_b, data + rows * k, rows * sizeof(float));
    for (c1 = 0; c1 < c1n; c1++)
        for (w = 0; w < m; w++)
            for (j = 0; j < c2n; j++) {
                size_t ch = c1 * c2n + j;
                if (ch < rows)
                    n->l1_pn[(c1 * m + w) * c2n + j] = data[rows * k + rows + ch * m + w];
            }
    free(data);
    return 0;
}

/* Fill the position input once: P [rows, m] tiled over the batch, in the
 * tensor's native layout and type, then bind it for every later run. */
static int bind_positions(neuronte_npu *n, const struct neuronte_npu_options *o, char *err, size_t err_len)
{
    char path[PATH_MAX];
    size_t len = 0, want = (size_t)n->pos_rows * (size_t)n->pos_m * sizeof(float);
    float *p;
    int c, b, m, rc, pos_batch;
    sidecar_path(path, sizeof path, o, n->pos_file);
    p = slurp(path, &len);
    if (!p || len != want) {
        free(p);
        say(err, err_len, "cannot read %zu bytes of positions from %s", want, path);
        return -1;
    }
    memset(&n->pos_attr, 0, sizeof n->pos_attr);
    n->pos_attr.index = (uint32_t)n->pos_index;
    if (n->api.query(n->ctx, RKNN_QUERY_INPUT_ATTR, &n->pos_attr, sizeof n->pos_attr)) {
        free(p);
        say(err, err_len, "cannot query the position input");
        return -1;
    }
    /* the position tensor is tiled over the batch, or shared (one copy) */
    if (n->pos_attr.n_elems == (uint32_t)(n->pos_rows * n->pos_m))
        pos_batch = 1;
    else if (n->pos_attr.n_elems == (uint32_t)(n->pos_rows * n->pos_m * n->batch))
        pos_batch = n->batch;
    else {
        free(p);
        say(err, err_len, "position input has %u elements, expected %d x %d x (1 or %d)", n->pos_attr.n_elems,
            n->pos_rows, n->pos_m, n->batch);
        return -1;
    }
    memset(&n->pos_attr, 0, sizeof n->pos_attr);
    n->pos_attr.index = (uint32_t)n->pos_index;
    if (n->api.query(n->ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &n->pos_attr, sizeof n->pos_attr) ||
        !native_supported(&n->pos_attr)) {
        free(p);
        say(err, err_len, "position input layout not supported (fmt %d type %d)", n->pos_attr.fmt, n->pos_attr.type);
        return -1;
    }
    n->pos_mem = n->api.create_mem(n->ctx, n->pos_attr.size_with_stride);
    if (!n->pos_mem) {
        free(p);
        say(err, err_len, "rknn_create_mem failed for positions");
        return -1;
    }
    memset(n->pos_mem->virt_addr, 0, n->pos_attr.size_with_stride);
    for (c = 0; c < n->pos_rows; c++)
        for (b = 0; b < pos_batch; b++)
            for (m = 0; m < n->pos_m; m++) {
                size_t k = native_index(&n->pos_attr, c, b, m);
                float v = p[(size_t)c * (size_t)n->pos_m + (size_t)m];
                if (n->pos_attr.type == RKNN_TENSOR_FLOAT16)
                    ((half *)n->pos_mem->virt_addr)[k] = (half)v;
                else
                    ((float *)n->pos_mem->virt_addr)[k] = v;
            }
    free(p);
    n->pos_attr.pass_through = 1;
    if ((rc = n->api.mem_sync(n->ctx, n->pos_mem, RKNN_MEMORY_SYNC_TO_DEVICE)) ||
        (rc = n->api.set_io_mem(n->ctx, n->pos_mem, &n->pos_attr))) {
        say(err, err_len, "binding positions failed (%d)", rc);
        return -1;
    }
    return 0;
}

static rknn_tensor_mem *new_input_mem(neuronte_npu *n)
{
    rknn_tensor_mem *m = n->api.create_mem(n->ctx, n->in_attr.size_with_stride);
    if (m)
        memset(m->virt_addr, 0, n->in_attr.size_with_stride);
    return m;
}

/* Assembly table for the native output: the offset of every cell pixel for
 * glyph 0 and the distance from one glyph (batch row) to the next. Each
 * layout native_index accepts is linear in the batch index, so a glyph is
 * then one gather over the cell, whatever the block size. Leaves asm_off
 * NULL (the per-block loop is used) unless the blocks cover every pixel
 * exactly once. */
static void build_asm_table(neuronte_npu *n)
{
    const rknn_tensor_attr *a = &n->out_attr;
    size_t plane = (size_t)n->cell_w * (size_t)n->cell_h, q;
    uint32_t *off = malloc(plane * sizeof *off);
    unsigned char *seen = calloc(plane, 1);
    int g, r, i, ok = off && seen;
    for (g = 0; ok && g < n->nblocks; g++)
        for (r = 0; ok && r < n->bh; r++)
            for (i = 0; ok && i < n->bw; i++) {
                size_t k = native_index(a, g, 0, r * n->bw + i);
                q = (size_t)(n->blocks[g].y + r) * (size_t)n->cell_w + (size_t)(n->blocks[g].x + i);
                ok = q < plane && !seen[q] && k <= UINT32_MAX;
                if (ok) {
                    seen[q] = 1;
                    off[q] = (uint32_t)k;
                }
            }
    for (q = 0; ok && q < plane; q++)
        ok = seen[q];
    free(seen);
    if (!ok) {
        free(off);
        return;
    }
    n->asm_off = off;
    n->asm_glyph = native_index(a, 0, 1, 0) - native_index(a, 0, 0, 0);
}

static int open_rknn(neuronte_npu *n, const struct neuronte_npu_options *o, char *err, size_t err_len)
{
    char dev[256];
    rknn_input_output_num io;
    rknn_tensor_attr in, out;
    rknn_sdk_version ver;
    size_t len = 0;
    uint32_t want_in;
    void *model;
    int rc, zc;

    if (!o->rknn_path) {
        say(err, err_len, "no compiled NPU model given");
        return -1;
    }
    if (!rknpu_device(dev, sizeof dev)) {
        say(err, err_len, "%s", dev);
        return -1;
    }
    if (load_api(&n->api, o->runtime_lib, err, err_len))
        return -1;
    model = slurp(o->rknn_path, &len);
    if (!model) {
        say(err, err_len, "cannot read %s", o->rknn_path);
        return -1;
    }
    /* zero-copy buffers are synced explicitly (mem_sync), so the runtime's
     * own flush and invalidate on every run are redundant */
    zc = (o->flags & NEURONTE_NPU_ZERO_COPY) || n->pos_file[0] || n->l1_file[0];
    n->init_flags = o->init_flags | (zc ? RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE | RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE : 0);
    rc = n->api.init(&n->ctx, model, (uint32_t)len, n->init_flags, NULL);
    free(model);
    if (rc) {
        say(err, err_len, "rknn_init failed (%d) with %s", rc, dev);
        return -1;
    }
    n->have_ctx = 1;
    n->npu_cores = rknpu_irq_cores();
    n->core_mask = o->core_mask > 0 ? o->core_mask : 0;
    if (o->core_mask == 0 && n->npu_cores >= 0 && n->npu_cores < 3)
        n->core_mask = RKNN_NPU_CORE_0;
    if (n->core_mask && (rc = n->api.set_core_mask(n->ctx, (rknn_core_mask)n->core_mask))) {
        say(err, err_len, "rknn_set_core_mask(%d) failed (%d)", n->core_mask, rc);
        return -1;
    }
    memset(&ver, 0, sizeof ver);
    if (n->api.query(n->ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof ver) == 0)
        snprintf(n->versions, sizeof n->versions, "%s;%s", ver.api_version, ver.drv_version);
    if ((rc = n->api.query(n->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof io)) ||
        io.n_input != (n->pos_file[0] ? 2u : 1u) || io.n_output != 1) {
        say(err, err_len, "unexpected model inputs/outputs (%d)", rc);
        return -1;
    }
    n->code_index = 0;
    n->pos_index = -1;
    if (n->pos_file[0]) {
        uint32_t i;
        for (i = 0; i < io.n_input; i++) {
            rknn_tensor_attr a;
            memset(&a, 0, sizeof a);
            a.index = i;
            if (n->api.query(n->ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof a)) {
                say(err, err_len, "cannot query input %u", i);
                return -1;
            }
            if (!strcmp(a.name, "pos"))
                n->pos_index = (int)i;
            else
                n->code_index = (int)i;
        }
        if (n->pos_index < 0 || n->pos_index == n->code_index) {
            say(err, err_len, "compiled model has no distinct pos input");
            return -1;
        }
    }
    memset(&in, 0, sizeof in);
    memset(&out, 0, sizeof out);
    in.index = (uint32_t)n->code_index;
    /* the input is the character features, or h1 when layer 1 runs on the host */
    want_in = n->l1_file[0] ? (uint32_t)(n->l1_rows * n->batch * n->l1_m) : (uint32_t)(n->in_c * n->batch);
    if (n->api.query(n->ctx, RKNN_QUERY_INPUT_ATTR, &in, sizeof in) ||
        n->api.query(n->ctx, RKNN_QUERY_OUTPUT_ATTR, &out, sizeof out) || in.n_elems != want_in ||
        out.n_elems != (uint32_t)(n->out_c * n->batch * n->out_p)) {
        say(err, err_len, "compiled model does not match its meta (input %u, output %u elements)", in.n_elems,
            out.n_elems);
        return -1;
    }
    if (zc) {
        memset(&n->in_attr, 0, sizeof n->in_attr);
        memset(&n->out_attr, 0, sizeof n->out_attr);
        n->in_attr.index = (uint32_t)n->code_index;
        if (n->api.query(n->ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &n->in_attr, sizeof n->in_attr) ||
            n->api.query(n->ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &n->out_attr, sizeof n->out_attr) ||
            !native_supported(&n->in_attr) || !native_supported(&n->out_attr)) {
            say(err, err_len, "native tensor layout not supported (input fmt %d type %d, output fmt %d type %d)",
                n->in_attr.fmt, n->in_attr.type, n->out_attr.fmt, n->out_attr.type);
            return -1;
        }
        if ((n->in_attr.type == RKNN_TENSOR_INT8 || n->out_attr.type == RKNN_TENSOR_INT8) && !n->l1_file[0]) {
            say(err, err_len, "quantized (int8) models are supported only in the host layer-1 form");
            return -1;
        }
        n->in_mem = new_input_mem(n);
        n->out_mem = n->api.create_mem(n->ctx, n->out_attr.size_with_stride);
        if (n->l1_file[0] && !(o->flags & NEURONTE_NPU_NO_PIPELINE))
            n->in_mem2 = new_input_mem(n);
        if (!n->in_mem || !n->out_mem || (n->l1_file[0] && !(o->flags & NEURONTE_NPU_NO_PIPELINE) && !n->in_mem2)) {
            say(err, err_len, "rknn_create_mem failed");
            return -1;
        }
        n->bound = n->in_mem;
        n->in_attr.pass_through = 1;
        n->out_attr.pass_through = 1;
        if ((rc = n->api.set_io_mem(n->ctx, n->in_mem, &n->in_attr)) ||
            (rc = n->api.set_io_mem(n->ctx, n->out_mem, &n->out_attr))) {
            say(err, err_len, "rknn_set_io_mem failed (%d)", rc);
            return -1;
        }
        n->zero_copy = 1;
        build_asm_table(n);
        if (n->pos_file[0] && bind_positions(n, o, err, err_len))
            return -1;
        if (n->l1_file[0] && load_host_l1(n, o, err, err_len))
            return -1;
    } else {
        n->in_f32 = calloc((size_t)n->in_c * (size_t)n->batch, sizeof *n->in_f32);
        n->out_f32 = calloc((size_t)n->out_c * (size_t)n->batch * (size_t)n->out_p, sizeof *n->out_f32);
        if (!n->in_f32 || !n->out_f32) {
            say(err, err_len, "out of memory");
            return -1;
        }
    }
    return 0;
}

static void close_rknn(neuronte_npu *n)
{
    if (n->have_ctx) {
        if (n->in_mem)
            n->api.destroy_mem(n->ctx, n->in_mem);
        if (n->out_mem)
            n->api.destroy_mem(n->ctx, n->out_mem);
        if (n->pos_mem)
            n->api.destroy_mem(n->ctx, n->pos_mem);
        if (n->in_mem2)
            n->api.destroy_mem(n->ctx, n->in_mem2);
        n->api.destroy(n->ctx);
    }
    n->have_ctx = 0;
    n->in_mem = n->out_mem = n->pos_mem = n->in_mem2 = n->bound = NULL;
    free(n->in_f32);
    free(n->out_f32);
    n->in_f32 = n->out_f32 = NULL;
    free(n->asm_off);
    n->asm_off = NULL;
    free(n->l1_w);
    free(n->l1_b);
    free(n->l1_pn);
    free(n->l1_c);
    free(n->l1_fv);
    n->l1_w = n->l1_b = n->l1_pn = n->l1_c = n->l1_fv = NULL;
    if (n->api.dl)
        dlclose(n->api.dl);
    memset(&n->api, 0, sizeof n->api);
}

static int open_cpu(neuronte_npu *n, const struct neuronte_npu_options *o, char *err, size_t err_len)
{
    int i, area = 0;
    for (i = 0; i < n->ntiles; i++) {
        char path[PATH_MAX];
        int rc;
        snprintf(path, sizeof path, "%s/%s", o->model_dir ? o->model_dir : ".", n->tiles[i].file);
        if ((rc = neuronte_open_file(path, &n->tiles[i].cpu))) {
            say(err, err_len, "cannot open CPU network %s (%d)", path, rc);
            return -1;
        }
        if (n->tiles[i].w * n->tiles[i].h > area)
            area = n->tiles[i].w * n->tiles[i].h;
    }
    n->cpu_buf = malloc((size_t)area * sizeof *n->cpu_buf);
    return n->cpu_buf ? 0 : -1;
}

int neuronte_npu_open(const struct neuronte_npu_options *o, neuronte_npu **out, char *reason, size_t reason_len)
{
    neuronte_npu *n;
    char err[512] = "";
    if (!o || !o->meta_path || !out)
        return -1;
    *out = NULL;
    n = calloc(1, sizeof *n);
    if (!n)
        return -1;
    if (parse_meta(n, o->meta_path, err, sizeof err)) {
        say(reason, reason_len, "%s", err);
        neuronte_npu_close(n);
        return -1;
    }
    if (!(o->flags & NEURONTE_NPU_FORCE_CPU)) {
        if (open_rknn(n, o, err, sizeof err) == 0) {
            float margin = o->refine_margin > 0.0f ? o->refine_margin : o->refine_margin < 0.0f ? 0.0f : n->meta_refine;
            if (margin > 0.0f) {
                if (open_cpu(n, o, err, sizeof err)) {
                    say(reason, reason_len, "refinement needs the CPU networks: %s", err);
                    neuronte_npu_close(n);
                    return -1;
                }
                n->refine_margin = margin;
            }
            n->backend = NEURONTE_NPU_RKNN;
            *out = n;
            return 0;
        }
        close_rknn(n);
        snprintf(n->reason, sizeof n->reason, "%s", err);
        if (!(o->flags & NEURONTE_NPU_ALLOW_FALLBACK)) {
            say(reason, reason_len, "%s", err);
            neuronte_npu_close(n);
            return -1;
        }
    } else {
        snprintf(n->reason, sizeof n->reason, "CPU requested");
    }
    if (open_cpu(n, o, err, sizeof err)) {
        say(reason, reason_len, "%s", err);
        neuronte_npu_close(n);
        return -1;
    }
    n->backend = (o->flags & NEURONTE_NPU_FORCE_CPU) ? NEURONTE_NPU_CPU : NEURONTE_NPU_CPU_FALLBACK;
    say(reason, reason_len, "%s", n->reason);
    *out = n;
    return 0;
}

void neuronte_npu_close(neuronte_npu *n)
{
    int i;
    if (!n)
        return;
    close_rknn(n);
    for (i = 0; i < n->ntiles; i++)
        neuronte_close(n->tiles[i].cpu);
    free(n->cpu_buf);
    free(n->blocks);
    free(n);
}

int neuronte_npu_backend(const neuronte_npu *n) { return n ? n->backend : 0; }
const char *neuronte_npu_fallback_reason(const neuronte_npu *n) { return n ? n->reason : ""; }
int neuronte_npu_batch(const neuronte_npu *n) { return n ? n->batch : 0; }
const char *neuronte_npu_versions(const neuronte_npu *n) { return n ? n->versions : ""; }

int neuronte_npu_cell(const neuronte_npu *n, int *w, int *h)
{
    if (!n)
        return -1;
    *w = n->cell_w;
    *h = n->cell_h;
    return 0;
}

/* ---- evaluation --------------------------------------------------------- */

static uint32_t tile_scalar(const struct tile *t, uint32_t s)
{
    return t->mask < 0 ? s : BRAILLE_FIRST | ((s - BRAILLE_FIRST) & (uint32_t)t->mask);
}

/* The 64 character features of `code_of`, as +-1. */
static void features(const struct tile *t, uint32_t s, float *v)
{
    int i;
    for (i = 0; i < 21; i++)
        v[i] = (s >> i) & 1 ? 1.0f : -1.0f;
    for (i = 0; i < NMASK; i++)
        v[21 + i] = __builtin_parity(s & t->masks[i]) ? 1.0f : -1.0f;
}

/* One tile of one glyph by the reference library, copied into the glyph. */
static int cpu_tile(neuronte_npu *n, const struct tile *t, uint32_t scalar, float *glyph)
{
    int rc = neuronte_coverage(t->cpu, tile_scalar(t, scalar), t->w, t->h, n->samples, n->cpu_buf), r;
    if (rc)
        return rc;
    for (r = 0; r < t->h; r++)
        memcpy(glyph + (size_t)(t->y + r) * n->cell_w + t->x, n->cpu_buf + (size_t)r * t->w,
               (size_t)t->w * sizeof *glyph);
    return 0;
}

static int cpu_coverage(neuronte_npu *n, const uint32_t *scalars, int count, float *out,
                        struct neuronte_npu_timing *tm)
{
    size_t plane = (size_t)n->cell_w * (size_t)n->cell_h;
    uint64_t t0 = now_ns();
    int b, i, rc;
    for (b = 0; b < count; b++)
        for (i = 0; i < n->ntiles; i++)
            if ((rc = cpu_tile(n, &n->tiles[i], scalars[b], out + (size_t)b * plane)))
                return rc;
    tm->cpu_eval += now_ns() - t0;
    return 0;
}

/* Recompute on the CPU each tile with a pixel within refine_margin of 0.5. */
static int refine(neuronte_npu *n, const uint32_t *scalars, int count, float *out, struct neuronte_npu_timing *tm)
{
    size_t plane = (size_t)n->cell_w * (size_t)n->cell_h;
    uint64_t t0 = now_ns();
    int b, i, x, y, rc;
    for (b = 0; b < count; b++) {
        float *glyph = out + (size_t)b * plane;
        for (i = 0; i < n->ntiles; i++) {
            const struct tile *t = &n->tiles[i];
            int hit = 0;
            for (y = t->y; y < t->y + t->h && !hit; y++)
                for (x = t->x; x < t->x + t->w && !hit; x++) {
                    float d = glyph[(size_t)y * n->cell_w + x] - 0.5f;
                    hit = d < n->refine_margin && d > -n->refine_margin;
                }
            if (!hit)
                continue;
            if ((rc = cpu_tile(n, t, scalars[b], glyph)))
                return rc;
            tm->refined++;
        }
    }
    tm->cpu_eval += now_ns() - t0;
    return 0;
}

/* One NC1HWC2 row of h1: dst[w][j] = ReLU(pn[w][j] + c[j]) for j < 8. */
static void h1_row8(half *dst, const float *pn, const float *c, size_t width)
{
    size_t w;
#ifdef HAVE_NEON_F16
    float32x4_t c0 = vld1q_f32(c), c1 = vld1q_f32(c + 4), z = vdupq_n_f32(0.0f);
    for (w = 0; w < width; w++, pn += 8, dst += 8) {
        float32x4_t a = vmaxq_f32(vaddq_f32(vld1q_f32(pn), c0), z);
        float32x4_t b = vmaxq_f32(vaddq_f32(vld1q_f32(pn + 4), c1), z);
        vst1q_f16((__fp16 *)dst, vcombine_f16(vcvt_f16_f32(a), vcvt_f16_f32(b)));
    }
#else
    int j;
    for (w = 0; w < width; w++, pn += 8, dst += 8)
        for (j = 0; j < 8; j++) {
            float v = pn[j] + c[j];
            dst[j] = (half)(v > 0.0f ? v : 0.0f);
        }
#endif
}

/* One int8 NC1HWC2 row of h1 for a quantized model:
 * dst[w][j] = sat8(round(ReLU(pn[w][j] + c[j]) / scale) + zp) for j < 16. */
static void h1_row16_i8(int8_t *dst, const float *pn, const float *c, size_t width, float inv, int zp)
{
    size_t w;
#ifdef HAVE_NEON_F16
    float32x4_t cv[4], z = vdupq_n_f32(0.0f), vi = vdupq_n_f32(inv);
    int32x4_t vz = vdupq_n_s32(zp);
    int k;
    for (k = 0; k < 4; k++)
        cv[k] = vld1q_f32(c + 4 * k);
    if (zp == -128) {
        /* The usual case (ReLU output, calibrated from 0). Same result with
         * fewer operations: round (v * inv) without the ReLU, saturate to
         * [0, 255] unsigned (negative v gives 0, as ReLU then round does),
         * and subtract 128 by flipping the top bit. */
        const uint8x16_t top = vdupq_n_u8(0x80);
        for (w = 0; w < width; w++, pn += 16, dst += 16) {
            uint16x4_t u[4];
            for (k = 0; k < 4; k++)
                u[k] = vqmovun_s32(vcvtnq_s32_f32(vmulq_f32(vaddq_f32(vld1q_f32(pn + 4 * k), cv[k]), vi)));
            uint8x16_t b = vcombine_u8(vqmovn_u16(vcombine_u16(u[0], u[1])), vqmovn_u16(vcombine_u16(u[2], u[3])));
            vst1q_s8(dst, vreinterpretq_s8_u8(veorq_u8(b, top)));
        }
        return;
    }
    for (w = 0; w < width; w++, pn += 16, dst += 16) {
        int32x4_t q[4];
        for (k = 0; k < 4; k++) {
            float32x4_t v = vmulq_f32(vmaxq_f32(vaddq_f32(vld1q_f32(pn + 4 * k), cv[k]), z), vi);
            q[k] = vaddq_s32(vcvtnq_s32_f32(v), vz);
        }
        int16x8_t lo = vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]));
        int16x8_t hi = vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3]));
        vst1q_s8(dst, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
    }
#else
    int j;
    for (w = 0; w < width; w++, pn += 16, dst += 16)
        for (j = 0; j < 16; j++) {
            float v = pn[j] + c[j];
            long q = lrintf((v > 0.0f ? v : 0.0f) * inv) + zp;
            dst[j] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
        }
#endif
}

/* Layer 1 on the host: character part per glyph, then h1 into in_mem. */
static void host_layer1(neuronte_npu *n, const uint32_t *scalars, int count, rknn_tensor_mem *mem)
{
    const rknn_tensor_attr *a = &n->in_attr;
    size_t c1n = a->dims[1], hh = a->dims[2], width = a->dims[3], c2n = a->dims[4];
    size_t ws = a->w_stride ? a->w_stride : width, stride = c1n * c2n, c1, w, j;
    int B = n->batch, b, t, g, o, i;
    for (b = 0; b < B; b++) {
        uint32_t s = scalars[b < count ? b : 0];
        float *c = n->l1_c + (size_t)b * stride;
        for (t = 0; t < n->ntiles; t++)
            features(&n->tiles[t], tile_scalar(&n->tiles[t], s), n->l1_fv + (size_t)t * CODE);
        for (g = 0; g < n->nblocks; g++) {
            const float *fv = n->l1_fv + (size_t)n->blocks[g].tile * CODE;
            for (o = 0; o < n->hidden; o++) {
                int row = g * n->hidden + o;
                const float *wr = n->l1_w + (size_t)row * (size_t)n->l1_k + (size_t)n->blocks[g].tile * CODE;
                float acc = n->l1_b[row];
                for (i = 0; i < CODE; i++)
                    acc += wr[i] * fv[i];
                c[row] = acc;
            }
        }
    }
    for (c1 = 0; c1 < c1n; c1++)
        for (b = 0; b < B; b++) {
            const float *pn = n->l1_pn + c1 * width * c2n;
            const float *c = n->l1_c + (size_t)b * stride + c1 * c2n;
            if (a->type == RKNN_TENSOR_FLOAT16 && c2n == 8) {
                h1_row8((half *)mem->virt_addr + (c1 * hh + (size_t)b) * ws * c2n, pn, c, width);
                continue;
            }
            if (a->type == RKNN_TENSOR_INT8 && c2n == 16) {
                h1_row16_i8((int8_t *)mem->virt_addr + (c1 * hh + (size_t)b) * ws * c2n, pn, c, width,
                            1.0f / a->scale, a->zp);
                continue;
            }
            for (w = 0; w < width; w++)
                for (j = 0; j < c2n; j++) {
                    size_t k = ((c1 * hh + (size_t)b) * ws + w) * c2n + j;
                    float v = pn[w * c2n + j] + c[j];
                    v = v > 0.0f ? v : 0.0f;
                    if (a->type == RKNN_TENSOR_INT8) {
                        long q = lrintf(v / a->scale) + a->zp;
                        ((int8_t *)mem->virt_addr)[k] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
                    } else if (a->type == RKNN_TENSOR_FLOAT16)
                        ((half *)mem->virt_addr)[k] = (half)v;
                    else
                        ((float *)mem->virt_addr)[k] = v;
                }
        }
}

/* Write one batch's input: host layer 1 into `mem`, or the character
 * features into the input buffer, and in zero-copy mode clean it from the
 * CPU caches for the NPU. A short batch is padded with its first scalar.
 * Runs on the fill thread while npu_exec runs another buffer; returns the
 * cache-sync status. */
static int npu_fill(neuronte_npu *n, const uint32_t *scalars, int count, rknn_tensor_mem *mem, uint64_t *ns,
                    uint64_t *sync_ns)
{
    uint64_t t0 = now_ns(), t1;
    int B = n->batch, b, t, i, rc = 0;
    float v[CODE];
    if (n->l1_file[0]) {
        host_layer1(n, scalars, count, mem);
        goto sync;
    }
    for (b = 0; b < B; b++) {
        uint32_t s = scalars[b < count ? b : 0];
        for (t = 0; t < n->ntiles; t++) {
            features(&n->tiles[t], tile_scalar(&n->tiles[t], s), v);
            if (n->zero_copy) {
                const rknn_tensor_attr *a = &n->in_attr;
                for (i = 0; i < CODE; i++) {
                    size_t k = native_index(a, t * CODE + i, b, 0);
                    if (a->type == RKNN_TENSOR_FLOAT16)
                        ((half *)mem->virt_addr)[k] = (half)v[i];
                    else
                        ((float *)mem->virt_addr)[k] = v[i];
                }
            } else {
                /* NHWC [1, B, 1, C]: the runtimes accept only NHWC source data for
                 * 4-D float inputs (2.3.2 logs an error for NCHW yet returns 0). */
                memcpy(n->in_f32 + (size_t)b * (size_t)n->in_c + (size_t)t * CODE, v, sizeof v);
            }
        }
    }
sync:
    t1 = now_ns();
    *ns += t1 - t0;
    if (n->zero_copy) {
        rc = n->api.mem_sync(n->ctx, mem, RKNN_MEMORY_SYNC_TO_DEVICE);
        *sync_ns += now_ns() - t1;
    }
    return rc;
}

/* Run the batch held in `mem` and assemble `count` glyphs into `out`. */
static int npu_exec(neuronte_npu *n, rknn_tensor_mem *mem, int count, float *out, struct neuronte_npu_timing *tm)
{
    size_t plane = (size_t)n->cell_w * (size_t)n->cell_h;
    int B = n->batch, b, i, g, r, rc = 0;
    uint64_t t1 = now_ns(), t2, t3, t4;

    if (n->zero_copy) {
        if (mem != n->bound) {
            rknn_tensor_attr a = n->in_attr; /* the filler thread reads in_attr */
            if ((rc = n->api.set_io_mem(n->ctx, mem, &a)))
                return rc;
            n->bound = mem;
        }
        /* npu_fill already synced `mem` to the device */
    } else {
        rknn_input in;
        memset(&in, 0, sizeof in);
        in.index = 0;
        in.buf = n->in_f32;
        in.size = (uint32_t)((size_t)n->in_c * (size_t)B * sizeof(float));
        in.type = RKNN_TENSOR_FLOAT32;
        in.fmt = RKNN_TENSOR_NHWC;
        rc = n->api.inputs_set(n->ctx, 1, &in);
    }
    if (rc)
        return rc;
    t2 = now_ns();
    if ((rc = n->api.run(n->ctx, NULL)))
        return rc;
    t3 = now_ns();
    if (n->zero_copy) {
        rc = n->api.mem_sync(n->ctx, n->out_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    } else {
        rknn_output o;
        memset(&o, 0, sizeof o);
        o.want_float = 1;
        o.is_prealloc = 1;
        o.index = 0;
        o.buf = n->out_f32;
        o.size = (uint32_t)((size_t)n->out_c * (size_t)B * (size_t)n->out_p * sizeof(float));
        rc = n->api.outputs_get(n->ctx, 1, &o, NULL);
        if (!rc)
            n->api.outputs_release(n->ctx, 1, &o);
    }
    if (rc)
        return rc;
    t4 = now_ns();
    if (n->asm_off) {
        const rknn_tensor_attr *a = &n->out_attr;
        const uint32_t *off = n->asm_off;
        size_t q;
        for (b = 0; b < count; b++) {
            float *d = out + (size_t)b * plane;
            size_t base = (size_t)b * n->asm_glyph;
            if (a->type == RKNN_TENSOR_FLOAT16) {
                const half *src = (const half *)n->out_mem->virt_addr + base;
                for (q = 0; q < plane; q++)
                    d[q] = (float)src[off[q]];
            } else if (a->type == RKNN_TENSOR_INT8) {
                const int8_t *src = (const int8_t *)n->out_mem->virt_addr + base;
                const int zp = a->zp;
                const float sc = a->scale;
                for (q = 0; q < plane; q++)
                    d[q] = (float)(src[off[q]] - zp) * sc;
            } else {
                const float *src = (const float *)n->out_mem->virt_addr + base;
                for (q = 0; q < plane; q++)
                    d[q] = src[off[q]];
            }
        }
    } else {
        /* per block: non-zero-copy output, or blocks that do not tile the cell */
        for (g = 0; g < n->nblocks; g++) {
            const struct block *bk = &n->blocks[g];
            for (b = 0; b < count; b++) {
                float *dst = out + (size_t)b * plane + (size_t)bk->y * n->cell_w + bk->x;
                if (n->zero_copy) {
                    const rknn_tensor_attr *a = &n->out_attr;
                    /* consecutive pixels of a block row are `step` elements apart */
                    size_t step = a->fmt == RKNN_TENSOR_NC1HWC2 ? a->dims[4]
                                  : a->fmt == RKNN_TENSOR_NHWC  ? a->dims[3]
                                                                : 1;
                    for (r = 0; r < n->bh; r++) {
                        size_t k = native_index(a, g, b, r * n->bw);
                        float *d = dst + (size_t)r * n->cell_w;
                        if (a->type == RKNN_TENSOR_FLOAT16) {
                            const half *src = (const half *)n->out_mem->virt_addr + k;
                            for (i = 0; i < n->bw; i++)
                                d[i] = (float)src[(size_t)i * step];
                        } else if (a->type == RKNN_TENSOR_INT8) {
                            const int8_t *src = (const int8_t *)n->out_mem->virt_addr + k;
                            int zp = a->zp;
                            float sc = a->scale;
                            for (i = 0; i < n->bw; i++)
                                d[i] = (float)(src[(size_t)i * step] - zp) * sc;
                        } else {
                            const float *src = (const float *)n->out_mem->virt_addr + k;
                            for (i = 0; i < n->bw; i++)
                                d[i] = src[(size_t)i * step];
                        }
                    }
                } else {
                    const float *src = n->out_f32 + ((size_t)g * (size_t)B + (size_t)b) * (size_t)n->out_p;
                    for (r = 0; r < n->bh; r++)
                        memcpy(dst + (size_t)r * n->cell_w, src + (size_t)r * n->bw, (size_t)n->bw * sizeof *dst);
                }
            }
        }
    }
    tm->input += t2 - t1;
    tm->run += t3 - t2;
    tm->output += t4 - t3;
    tm->assemble += now_ns() - t4;
    tm->launches++;
    return 0;
}

struct fill_job {
    neuronte_npu *n;
    const uint32_t *scalars;
    int count;
    rknn_tensor_mem *mem;
    uint64_t ns, sync_ns;
    int rc;
};

static void *fill_main(void *arg)
{
    struct fill_job *j = arg;
    j->rc = npu_fill(j->n, j->scalars, j->count, j->mem, &j->ns, &j->sync_ns);
    return NULL;
}

/* Batches in order; with two input buffers, the host fills batch k+1 on a
 * second thread while the NPU runs batch k. */
static int npu_coverage(neuronte_npu *n, const uint32_t *scalars, int count, float *out,
                        struct neuronte_npu_timing *tm)
{
    size_t plane = (size_t)n->cell_w * (size_t)n->cell_h;
    rknn_tensor_mem *buf[2] = {n->in_mem, n->in_mem2 ? n->in_mem2 : n->in_mem};
    int B = n->batch, done, i, rc = 0;
    if (count == 0)
        return 0;
    if ((rc = npu_fill(n, scalars, count < B ? count : B, buf[0], &tm->features, &tm->input)))
        return rc;
    for (i = 0, done = 0; done < count; i++, done += B) {
        int k = count - done < B ? count - done : B;
        int next = done + B < count;
        struct fill_job job;
        pthread_t th;
        int threaded = 0;
        if (next) {
            job.n = n;
            job.scalars = scalars + done + B;
            job.count = count - done - B < B ? count - done - B : B;
            job.mem = buf[(i + 1) & 1];
            job.ns = job.sync_ns = 0;
            job.rc = 0;
            if (n->in_mem2 && pthread_create(&th, NULL, fill_main, &job) == 0)
                threaded = 1;
        }
        rc = npu_exec(n, buf[i & 1], k, out + (size_t)done * plane, tm);
        if (!rc && n->refine_margin > 0.0f)
            rc = refine(n, scalars + done, k, out + (size_t)done * plane, tm);
        if (threaded)
            pthread_join(th, NULL);
        else if (next && !rc)
            fill_main(&job);
        if (next) {
            tm->features += job.ns;
            tm->input += job.sync_ns;
            if (!rc)
                rc = job.rc;
        }
        if (rc)
            break;
    }
    return rc;
}

int neuronte_npu_coverage(neuronte_npu *n, const uint32_t *scalars, int count, float *out,
                          struct neuronte_npu_timing *timing)
{
    struct neuronte_npu_timing local;
    uint64_t t0 = now_ns();
    int rc = 0;
    if (!n || !scalars || !out || count < 0)
        return -1;
    if (!timing)
        timing = &local;
    memset(timing, 0, sizeof *timing);
    if (n->backend != NEURONTE_NPU_RKNN)
        rc = cpu_coverage(n, scalars, count, out, timing);
    else
        rc = npu_coverage(n, scalars, count, out, timing);
    timing->total = now_ns() - t0;
    return rc;
}

int neuronte_npu_info(const neuronte_npu *n, struct neuronte_npu_info *info)
{
    if (!n || !info)
        return -1;
    memset(info, 0, sizeof *info);
    if (n->backend == NEURONTE_NPU_RKNN) {
        info->layer1 = n->l1_file[0] ? 2 : 1;
        info->zero_copy = n->zero_copy;
        info->pipeline = n->in_mem2 != NULL;
        info->init_flags = n->init_flags;
        info->refine_margin = n->refine_margin;
        info->core_mask = n->core_mask;
        info->npu_cores = n->npu_cores;
        info->dtype = n->zero_copy ? (n->in_attr.type == RKNN_TENSOR_INT8 ? 2 : n->in_attr.type == RKNN_TENSOR_FLOAT16)
                                   : 0;
    }
    return 0;
}
