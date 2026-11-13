/* rknn_probe: load any .rknn model, describe it and time it.
 *
 *   rknn_probe [options] LIBRKNNRT MODEL
 *     -n RUNS      measured runs (default 20)
 *     -w WARMUP    unmeasured runs first (default 3)
 *     -m MASK      rknn_set_core_mask value (1 = core 0; 0 = leave the runtime default)
 *     -f FLAGS     extra rknn_init flags (decimal or 0x hex)
 *     -z           zero-copy: native-layout input/output buffers bound with
 *                  rknn_set_io_mem, synced around each run (what the backend does);
 *                  without -z, float32 data goes through rknn_inputs_set and
 *                  rknn_outputs_get(want_float)
 *     -p           per-operator profile (RKNN_FLAG_COLLECT_PERF_MASK; slows runs)
 *     -j           print one JSON summary line last
 *     -i FILE      float32 data for input 0 (non zero-copy mode)
 *     -o FILE      write output 0 as float32 after the last run (non zero-copy mode)
 *     -c           NHWC instead of NCHW for -i data
 *
 * Input 0 gets a deterministic pattern unless -i is given; further inputs get zeros.
 * Timings: "set" (inputs_set or cache sync), "run" (rknn_run), "get" (outputs_get
 * or cache sync), and "npu" (RKNN_QUERY_PERF_RUN, the runtime's own measure).
 * Used to isolate operator support and runtime/driver behaviour, and to compare
 * compiled variants (data types, layouts, flags) on the same board. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rknn_api.h"

#define MAX_IO 8

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n)
{
    if (n <= 0)
        return NAN;
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void *load_file(const char *path, long *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)*len);
    if (p && fread(p, 1, (size_t)*len, f) != (size_t)*len) {
        free(p);
        p = NULL;
    }
    fclose(f);
    return p;
}

static void print_attr(const char *kind, const rknn_tensor_attr *a, const rknn_tensor_attr *nat)
{
    printf("%s %u %s: fmt=%s type=%s qnt=%s zp=%d scale=%g dims=[%u,%u,%u,%u,%u] n=%u", kind, a->index, a->name,
           get_format_string(a->fmt), get_type_string(a->type), get_qnt_type_string(a->qnt_type), a->zp, a->scale,
           a->dims[0], a->dims[1], a->dims[2], a->dims[3], a->dims[4], a->n_elems);
    if (nat)
        printf(" | native fmt=%s type=%s dims=[%u,%u,%u,%u,%u] w_stride=%u size=%u", get_format_string(nat->fmt),
               get_type_string(nat->type), nat->dims[0], nat->dims[1], nat->dims[2], nat->dims[3], nat->dims[4],
               nat->w_stride, nat->size_with_stride);
    printf("\n");
}

struct api {
    int (*init)(rknn_context *, void *, uint32_t, uint32_t, rknn_init_extend *);
    int (*query)(rknn_context, rknn_query_cmd, void *, uint32_t);
    int (*inputs_set)(rknn_context, uint32_t, rknn_input *);
    int (*run)(rknn_context, rknn_run_extend *);
    int (*outputs_get)(rknn_context, uint32_t, rknn_output *, rknn_output_extend *);
    int (*outputs_release)(rknn_context, uint32_t, rknn_output *);
    int (*set_core_mask)(rknn_context, rknn_core_mask);
    rknn_tensor_mem *(*create_mem)(rknn_context, uint32_t);
    int (*destroy_mem)(rknn_context, rknn_tensor_mem *);
    int (*set_io_mem)(rknn_context, rknn_tensor_mem *, rknn_tensor_attr *);
    int (*mem_sync)(rknn_context, rknn_tensor_mem *, rknn_mem_sync_mode);
    int (*destroy)(rknn_context);
};

static int load_api(const char *lib, struct api *a)
{
    void *dl = dlopen(lib, RTLD_NOW);
    if (!dl) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return -1;
    }
#define SYM(f, name)                                              \
    do {                                                          \
        *(void **)&a->f = dlsym(dl, name);                        \
        if (!a->f) {                                              \
            fprintf(stderr, "missing symbol %s in %s\n", name, lib); \
            return -1;                                            \
        }                                                         \
    } while (0)
    SYM(init, "rknn_init");
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
    SYM(destroy, "rknn_destroy");
#undef SYM
    return 0;
}

static void usage(void)
{
    fprintf(stderr, "usage: rknn_probe [-n RUNS] [-w WARMUP] [-m MASK] [-f FLAGS] [-z] [-p] [-j] [-i IN.f32] "
                    "[-o OUT.f32] [-c] LIBRKNNRT MODEL\n");
    exit(2);
}

int main(int argc, char **argv)
{
    int runs = 20, warmup = 3, mask = 0, zero_copy = 0, perf = 0, json = 0, nhwc = 0;
    unsigned long flags = 0;
    const char *in_file = NULL, *out_file = NULL;
    int i = 1;
    for (; i < argc && argv[i][0] == '-'; i++) {
        const char *o = argv[i];
#define VAL() (i + 1 < argc ? argv[++i] : (usage(), ""))
        if (!strcmp(o, "-n")) runs = atoi(VAL());
        else if (!strcmp(o, "-w")) warmup = atoi(VAL());
        else if (!strcmp(o, "-m")) mask = atoi(VAL());
        else if (!strcmp(o, "-f")) flags = strtoul(VAL(), NULL, 0);
        else if (!strcmp(o, "-z")) zero_copy = 1;
        else if (!strcmp(o, "-p")) perf = 1;
        else if (!strcmp(o, "-j")) json = 1;
        else if (!strcmp(o, "-i")) in_file = VAL();
        else if (!strcmp(o, "-o")) out_file = VAL();
        else if (!strcmp(o, "-c")) nhwc = 1;
        else usage();
#undef VAL
    }
    if (argc - i != 2 || runs < 1 || runs > 100000 || warmup < 0)
        usage();
    const char *lib = argv[i], *model_path = argv[i + 1];
    if (zero_copy && (in_file || out_file)) {
        fprintf(stderr, "-i/-o work only without -z\n");
        return 2;
    }
    struct api api;
    if (load_api(lib, &api))
        return 1;
    long len = 0;
    void *model = load_file(model_path, &len);
    if (!model) {
        fprintf(stderr, "cannot read %s: %s\n", model_path, strerror(errno));
        return 1;
    }
    if (perf)
        flags |= RKNN_FLAG_COLLECT_PERF_MASK;
    if (zero_copy)
        flags |= RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE | RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE;
    rknn_context ctx;
    double t0 = now_us();
    int rc = api.init(&ctx, model, (uint32_t)len, (uint32_t)flags, NULL);
    double init_us = now_us() - t0;
    printf("init rc=%d %.1f us flags=0x%lx\n", rc, init_us, flags);
    if (rc)
        return 1;
    if (mask && (rc = api.set_core_mask(ctx, (rknn_core_mask)mask))) {
        fprintf(stderr, "rknn_set_core_mask(%d) failed (%d)\n", mask, rc);
        return 1;
    }
    rknn_sdk_version v = {0};
    api.query(ctx, RKNN_QUERY_SDK_VERSION, &v, sizeof v);
    printf("versions api=%s driver=%s\n", v.api_version, v.drv_version);
    rknn_mem_size ms = {0};
    if (api.query(ctx, RKNN_QUERY_MEM_SIZE, &ms, sizeof ms) == 0)
        printf("memory weight=%u internal=%u dma_total=%llu sram_total=%u sram_free=%u\n", ms.total_weight_size,
               ms.total_internal_size, (unsigned long long)ms.total_dma_allocated_size, ms.total_sram_size,
               ms.free_sram_size);
    rknn_input_output_num io = {0};
    api.query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof io);
    if (io.n_input > MAX_IO || io.n_output > MAX_IO || io.n_input < 1 || io.n_output < 1) {
        fprintf(stderr, "unsupported tensor count: %u inputs, %u outputs\n", io.n_input, io.n_output);
        return 1;
    }
    printf("inputs %u outputs %u\n", io.n_input, io.n_output);
    rknn_tensor_attr in[MAX_IO], nin[MAX_IO], out[MAX_IO], nout[MAX_IO];
    for (uint32_t k = 0; k < io.n_input; k++) {
        in[k] = (rknn_tensor_attr){.index = k};
        nin[k] = (rknn_tensor_attr){.index = k};
        api.query(ctx, RKNN_QUERY_INPUT_ATTR, &in[k], sizeof in[k]);
        api.query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &nin[k], sizeof nin[k]);
        print_attr("input", &in[k], &nin[k]);
    }
    for (uint32_t k = 0; k < io.n_output; k++) {
        out[k] = (rknn_tensor_attr){.index = k};
        nout[k] = (rknn_tensor_attr){.index = k};
        api.query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out[k], sizeof out[k]);
        api.query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &nout[k], sizeof nout[k]);
        print_attr("output", &out[k], &nout[k]);
    }

    /* Buffers. */
    rknn_tensor_mem *in_mem[MAX_IO] = {0}, *out_mem[MAX_IO] = {0};
    rknn_input ri[MAX_IO];
    float *obuf[MAX_IO] = {0};
    if (zero_copy) {
        for (uint32_t k = 0; k < io.n_input; k++) {
            in_mem[k] = api.create_mem(ctx, nin[k].size_with_stride);
            if (!in_mem[k]) {
                fprintf(stderr, "rknn_create_mem(input %u) failed\n", k);
                return 1;
            }
            unsigned char *p = in_mem[k]->virt_addr;
            for (uint32_t b = 0; b < nin[k].size_with_stride; b++)
                p[b] = k ? 0 : (unsigned char)((b * 2654435761u) >> 26); /* small values in any type */
            nin[k].pass_through = 1;
            if ((rc = api.set_io_mem(ctx, in_mem[k], &nin[k]))) {
                fprintf(stderr, "rknn_set_io_mem(input %u) failed (%d)\n", k, rc);
                return 1;
            }
        }
        for (uint32_t k = 0; k < io.n_output; k++) {
            out_mem[k] = api.create_mem(ctx, nout[k].size_with_stride);
            if (!out_mem[k] || (rc = api.set_io_mem(ctx, out_mem[k], &nout[k]))) {
                fprintf(stderr, "output %u buffer failed (%d)\n", k, rc);
                return 1;
            }
        }
    } else {
        for (uint32_t k = 0; k < io.n_input; k++) {
            float *b = calloc(in[k].n_elems, sizeof *b);
            if (!b)
                return 1;
            if (k == 0) {
                for (uint32_t e = 0; e < in[0].n_elems; e++)
                    b[e] = (e * 2654435761u >> 31) ? 1.0f : -1.0f;
                if (in_file) {
                    FILE *fi = fopen(in_file, "rb");
                    if (!fi || fread(b, sizeof *b, in[0].n_elems, fi) != in[0].n_elems) {
                        fprintf(stderr, "cannot read %u floats from %s\n", in[0].n_elems, in_file);
                        return 1;
                    }
                    fclose(fi);
                }
            }
            ri[k] = (rknn_input){.index = k, .buf = b, .size = in[k].n_elems * 4, .type = RKNN_TENSOR_FLOAT32,
                                 .fmt = nhwc ? RKNN_TENSOR_NHWC : RKNN_TENSOR_NCHW};
        }
        for (uint32_t k = 0; k < io.n_output; k++)
            if (!(obuf[k] = malloc(out[k].n_elems * sizeof(float))))
                return 1;
    }

    /* Runs. */
    double *t_set = calloc((size_t)runs, sizeof *t_set), *t_run = calloc((size_t)runs, sizeof *t_run);
    double *t_get = calloc((size_t)runs, sizeof *t_get), *t_npu = calloc((size_t)runs, sizeof *t_npu);
    int failed = 0, n_npu = 0;
    for (int r = -warmup; r < runs && !failed; r++) {
        double a = now_us();
        int src = 0;
        if (zero_copy)
            for (uint32_t k = 0; k < io.n_input && !src; k++)
                src = api.mem_sync(ctx, in_mem[k], RKNN_MEMORY_SYNC_TO_DEVICE);
        else
            src = api.inputs_set(ctx, io.n_input, ri);
        double b = now_us();
        int rr = api.run(ctx, NULL);
        double c = now_us();
        int grc = 0;
        rknn_output ro[MAX_IO];
        if (zero_copy) {
            for (uint32_t k = 0; k < io.n_output && !grc; k++)
                grc = api.mem_sync(ctx, out_mem[k], RKNN_MEMORY_SYNC_FROM_DEVICE);
        } else {
            for (uint32_t k = 0; k < io.n_output; k++)
                ro[k] = (rknn_output){.want_float = 1, .is_prealloc = 1, .index = k, .buf = obuf[k],
                                      .size = out[k].n_elems * 4};
            grc = api.outputs_get(ctx, io.n_output, ro, NULL);
        }
        double d = now_us();
        rknn_perf_run pr = {0};
        api.query(ctx, RKNN_QUERY_PERF_RUN, &pr, sizeof pr);
        if (src || rr || grc) {
            printf("run %d failed: set rc=%d run rc=%d get rc=%d\n", r, src, rr, grc);
            failed = 1;
        } else if (r >= 0) {
            t_set[r] = b - a;
            t_run[r] = c - b;
            t_get[r] = d - c;
            if (pr.run_duration > 0)
                t_npu[n_npu++] = (double)pr.run_duration;
        }
        if (!json || failed)
            printf("run %d: set %.1f run %.1f get %.1f npu %lld us\n", r, b - a, c - b, d - c,
                   (long long)pr.run_duration);
        if (!zero_copy && !grc) {
            if (r == runs - 1 && out_file) {
                FILE *fo = fopen(out_file, "wb");
                if (!fo || fwrite(obuf[0], sizeof(float), out[0].n_elems, fo) != out[0].n_elems)
                    return 1;
                fclose(fo);
            }
            api.outputs_release(ctx, io.n_output, ro);
        }
    }
    if (perf) {
        rknn_perf_detail pd = {0};
        if (api.query(ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof pd) == 0 && pd.perf_data)
            printf("%s\n", pd.perf_data);
    }
    double run_min = INFINITY, run_max = 0;
    for (int r = 0; r < runs; r++) {
        run_min = fmin(run_min, t_run[r]);
        run_max = fmax(run_max, t_run[r]);
    }
    double m_set = median(t_set, runs), m_run = median(t_run, runs), m_get = median(t_get, runs),
           m_npu = median(t_npu, n_npu);
    printf("median over %d runs: set %.1f run %.1f [%.1f-%.1f] get %.1f npu %.1f us\n", runs, m_set, m_run, run_min,
           run_max, m_get, m_npu);
    if (json) {
        const char *base = strrchr(model_path, '/');
        printf("{\"model\":\"%s\",\"ok\":%s,\"zero_copy\":%d,\"core_mask\":%d,\"init_flags\":%lu,\"perf\":%d,"
               "\"runs\":%d,\"warmup\":%d,\"init_us\":%.1f,\"api\":\"%s\",\"driver\":\"%s\","
               "\"in_type\":\"%s\",\"out_type\":\"%s\",\"weight_bytes\":%u,\"internal_bytes\":%u,"
               "\"sram_total\":%u,\"set_us\":%.1f,\"run_us\":%.1f,\"run_min_us\":%.1f,\"run_max_us\":%.1f,"
               "\"get_us\":%.1f,\"npu_us\":%.1f}\n",
               base ? base + 1 : model_path, failed ? "false" : "true", zero_copy, mask, flags, perf, runs, warmup,
               init_us, v.api_version, v.drv_version, get_type_string(nin[0].type), get_type_string(nout[0].type),
               ms.total_weight_size, ms.total_internal_size, ms.total_sram_size, m_set, m_run, run_min, run_max,
               m_get, isnan(m_npu) ? -1.0 : m_npu);
    }
    for (uint32_t k = 0; k < io.n_input; k++)
        if (in_mem[k])
            api.destroy_mem(ctx, in_mem[k]);
    for (uint32_t k = 0; k < io.n_output; k++)
        if (out_mem[k])
            api.destroy_mem(ctx, out_mem[k]);
    api.destroy(ctx);
    return failed;
}
