/* Load a single-output .rknn model, feed input 0 deterministic float32 data
 * (or a file) and any further inputs zeros, run it N times and print timings,
 * output stats and, with PERF set in the environment, the per-operator
 * profile. Used to isolate operator support and runtime/driver behaviour. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rknn_api.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: rknn_probe LIBRKNNRT MODEL [RUNS] [CORE_MASK] [nchw|nhwc] [IN.f32] [OUT.f32]\n");
        return 2;
    }
    int runs = argc > 3 ? atoi(argv[3]) : 5, mask = argc > 4 ? atoi(argv[4]) : 0;
    int nhwc = argc > 5 && !strcmp(argv[5], "nhwc");
    const char *in_file = argc > 6 ? argv[6] : NULL, *out_file = argc > 7 ? argv[7] : NULL;
    void *dl = dlopen(argv[1], RTLD_NOW);
    if (!dl) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    int (*init)(rknn_context *, void *, uint32_t, uint32_t, rknn_init_extend *) = dlsym(dl, "rknn_init");
    int (*query)(rknn_context, rknn_query_cmd, void *, uint32_t) = dlsym(dl, "rknn_query");
    int (*inputs_set)(rknn_context, uint32_t, rknn_input *) = dlsym(dl, "rknn_inputs_set");
    int (*run)(rknn_context, rknn_run_extend *) = dlsym(dl, "rknn_run");
    int (*outputs_get)(rknn_context, uint32_t, rknn_output *, rknn_output_extend *) = dlsym(dl, "rknn_outputs_get");
    int (*outputs_release)(rknn_context, uint32_t, rknn_output *) = dlsym(dl, "rknn_outputs_release");
    int (*set_core_mask)(rknn_context, rknn_core_mask) = dlsym(dl, "rknn_set_core_mask");
    int (*destroy)(rknn_context) = dlsym(dl, "rknn_destroy");
    FILE *f = fopen(argv[2], "rb");
    if (!f)
        return 1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *model = malloc((size_t)len);
    if (fread(model, 1, (size_t)len, f) != (size_t)len)
        return 1;
    fclose(f);
    rknn_context ctx;
    double t0 = now_us();
    int rc = init(&ctx, model, (uint32_t)len, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
    printf("init rc=%d %.1f us\n", rc, now_us() - t0);
    if (rc)
        return 1;
    if (mask)
        printf("core_mask rc=%d\n", set_core_mask(ctx, (rknn_core_mask)mask));
    rknn_sdk_version v;
    query(ctx, RKNN_QUERY_SDK_VERSION, &v, sizeof v);
    printf("versions %s ; %s\n", v.api_version, v.drv_version);
    rknn_input_output_num io = {0};
    query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof io);
    printf("inputs %u outputs %u\n", io.n_input, io.n_output);
    rknn_tensor_attr in = {0}, out = {0}, nin = {0}, nout = {0};
    query(ctx, RKNN_QUERY_INPUT_ATTR, &in, sizeof in);
    query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out, sizeof out);
    query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &nin, sizeof nin);
    query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &nout, sizeof nout);
    printf("input  n=%u fmt=%s type=%s dims=%u,%u,%u,%u,%u native fmt=%s type=%s dims=%u,%u,%u,%u,%u wstride=%u size=%u\n",
           in.n_elems, get_format_string(in.fmt), get_type_string(in.type), in.dims[0], in.dims[1], in.dims[2],
           in.dims[3], in.dims[4], get_format_string(nin.fmt), get_type_string(nin.type), nin.dims[0], nin.dims[1],
           nin.dims[2], nin.dims[3], nin.dims[4], nin.w_stride, nin.size_with_stride);
    printf("output n=%u fmt=%s type=%s dims=%u,%u,%u,%u,%u native fmt=%s type=%s dims=%u,%u,%u,%u,%u wstride=%u size=%u\n",
           out.n_elems, get_format_string(out.fmt), get_type_string(out.type), out.dims[0], out.dims[1], out.dims[2],
           out.dims[3], out.dims[4], get_format_string(nout.fmt), get_type_string(nout.type), nout.dims[0],
           nout.dims[1], nout.dims[2], nout.dims[3], nout.dims[4], nout.w_stride, nout.size_with_stride);
    float *buf = malloc(in.n_elems * sizeof *buf), *ob = malloc(out.n_elems * sizeof *ob);
    for (uint32_t i = 0; i < in.n_elems; i++)
        buf[i] = (i * 2654435761u >> 31) ? 1.0f : -1.0f;
    if (in_file) {
        FILE *fi = fopen(in_file, "rb");
        if (!fi || fread(buf, sizeof *buf, in.n_elems, fi) != in.n_elems) {
            fprintf(stderr, "cannot read %u floats from %s\n", in.n_elems, in_file);
            return 1;
        }
        fclose(fi);
    }
    rknn_input ri[4] = {{.index = 0, .buf = buf, .size = in.n_elems * 4, .type = RKNN_TENSOR_FLOAT32,
                         .fmt = nhwc ? RKNN_TENSOR_NHWC : RKNN_TENSOR_NCHW}};
    for (uint32_t k = 1; k < io.n_input && k < 4; k++) {
        rknn_tensor_attr a = {.index = k};
        query(ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof a);
        ri[k] = (rknn_input){.index = k, .buf = calloc(a.n_elems, 4), .size = a.n_elems * 4,
                             .type = RKNN_TENSOR_FLOAT32, .fmt = ri[0].fmt};
        printf("input %u %s n=%u\n", k, a.name, a.n_elems);
    }
    for (int r = 0; r < runs; r++) {
        double a = now_us();
        rc = inputs_set(ctx, io.n_input < 4 ? io.n_input : 4, ri);
        double b = now_us();
        int rr = run(ctx, NULL);
        double c = now_us();
        rknn_output ro = {.want_float = 1, .is_prealloc = 1, .index = 0, .buf = ob, .size = out.n_elems * 4};
        int ro_rc = outputs_get(ctx, 1, &ro, NULL);
        double d = now_us();
        rknn_perf_run perf = {0};
        query(ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof perf);
        double sum = 0, mn = INFINITY, mx = -INFINITY;
        for (uint32_t i = 0; i < out.n_elems; i++) {
            sum += ob[i];
            mn = fmin(mn, ob[i]);
            mx = fmax(mx, ob[i]);
        }
        printf("run %d: set rc=%d %.1f us, run rc=%d %.1f us (npu %lld us), get rc=%d %.1f us, out sum=%.6g min=%.6g max=%.6g\n",
               r, rc, b - a, rr, c - b, (long long)perf.run_duration, ro_rc, d - c, sum, mn, mx);
        if (!ro_rc && out_file && r == runs - 1) {
            FILE *fo = fopen(out_file, "wb");
            if (!fo || fwrite(ob, sizeof *ob, out.n_elems, fo) != out.n_elems)
                return 1;
            fclose(fo);
        }
        if (!ro_rc)
            outputs_release(ctx, 1, &ro);
        if (rr)
            break;
    }
    if (getenv("PERF")) {
        rknn_perf_detail pd = {0};
        if (query(ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof pd) == 0 && pd.perf_data)
            printf("%s\n", pd.perf_data);
    }
    destroy(ctx);
    return 0;
}
