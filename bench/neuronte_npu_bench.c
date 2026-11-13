/* Benchmark one compiled model set on the NPU or the reference CPU path.
 *
 * One process measures one backend: opening (model load / network parse),
 * the first full-corpus call, warmup rounds, measured rounds and optional
 * single-glyph calls. Every round is printed; nothing is discarded. The
 * coverage of the final measured pass is written in scalar order so its
 * quality can be checked against teacher masks and the CPU output.
 *
 * Output: one JSON object on stdout.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "neuronte_npu.h"

#define CORPUS 256
#define MAX_ROUNDS 1000

static uint64_t clock_ns(clockid_t id)
{
    struct timespec ts;
    clock_gettime(id, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void read_text(const char *path, char *buf, size_t len)
{
    FILE *f = fopen(path, "r");
    size_t n = 0;
    buf[0] = 0;
    if (!f)
        return;
    n = fread(buf, 1, len - 1, f);
    fclose(f);
    buf[n] = 0;
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == ' '))
        buf[--n] = 0;
}

/* A snapshot of frequency, temperature and load, as a JSON object. */
static void snapshot(FILE *out, int cpu)
{
    char path[256], v[256];
    int z;
    fprintf(out, "{");
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu < 0 ? 0 : cpu);
    read_text(path, v, sizeof v);
    fprintf(out, "\"cpu_khz\":\"%s\"", v);
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpu < 0 ? 0 : cpu);
    read_text(path, v, sizeof v);
    fprintf(out, ",\"cpu_governor\":\"%s\"", v);
    read_text("/sys/class/devfreq/dmc/cur_freq", v, sizeof v);
    fprintf(out, ",\"dmc_hz\":\"%s\"", v);
    read_text("/proc/loadavg", v, sizeof v);
    fprintf(out, ",\"loadavg\":\"%s\"", v);
    read_text("/sys/class/devfreq/fdab0000.npu/cur_freq", v, sizeof v);
    fprintf(out, ",\"npu_hz\":\"%s\"", v);
    read_text("/sys/class/devfreq/fdab0000.npu/load", v, sizeof v);
    fprintf(out, ",\"npu_devfreq_load\":\"%s\"", v);
    fprintf(out, ",\"thermal\":{");
    for (z = 0; z < 16; z++) {
        char type[64], temp[64];
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/type", z);
        read_text(path, type, sizeof type);
        if (!type[0])
            break;
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/temp", z);
        read_text(path, temp, sizeof temp);
        fprintf(out, "%s\"%s\":\"%s\"", z ? "," : "", type, temp);
    }
    fprintf(out, "}}");
}

static void shuffle(uint32_t *v, int n, uint64_t seed)
{
    uint64_t x = seed * 6364136223846793005ull + 1442695040888963407ull;
    for (int i = n - 1; i > 0; i--) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        int j = (int)((x >> 33) % (uint64_t)(i + 1));
        uint32_t t = v[i];
        v[i] = v[j];
        v[j] = t;
    }
}

static void usage(void)
{
    fprintf(stderr,
            "usage: neuronte_npu_bench --meta FILE --models DIR [--rknn FILE] [--runtime LIB]\n"
            "       [--mode npu|cpu] [--no-fallback] [--zero-copy] [--core-mask N] [--cpu K]\n"
            "       [--no-pipeline] [--init-flags N] [--refine MARGIN]\n"
            "       [--warmup N] [--repeats N] [--passes N] [--single N] [--seed N] [--coverage-out FILE]\n");
    exit(2);
}

struct round {
    uint64_t wall, cpu;
    struct neuronte_npu_timing t;
};

static void timing_json(FILE *o, const struct neuronte_npu_timing *t)
{
    fprintf(o,
            "{\"features\":%llu,\"input\":%llu,\"run\":%llu,\"output\":%llu,\"assemble\":%llu,"
            "\"cpu_eval\":%llu,\"total\":%llu,\"launches\":%d,\"refined\":%d}",
            (unsigned long long)t->features, (unsigned long long)t->input, (unsigned long long)t->run,
            (unsigned long long)t->output, (unsigned long long)t->assemble, (unsigned long long)t->cpu_eval,
            (unsigned long long)t->total, t->launches, t->refined);
}

static void add(struct neuronte_npu_timing *a, const struct neuronte_npu_timing *b)
{
    a->features += b->features;
    a->input += b->input;
    a->run += b->run;
    a->output += b->output;
    a->assemble += b->assemble;
    a->cpu_eval += b->cpu_eval;
    a->total += b->total;
    a->launches += b->launches;
    a->refined += b->refined;
}

int main(int argc, char **argv)
{
    struct neuronte_npu_options opt = {0};
    const char *mode = "npu", *coverage_out = NULL;
    int warmup = 1, repeats = 10, passes = 4, single = 0, cpu = -1, i, p, r;
    uint64_t seed = 1;
    char reason[512] = "";
    neuronte_npu *n = NULL;
    opt.flags = NEURONTE_NPU_ALLOW_FALLBACK;

    for (i = 1; i < argc; i++) {
#define ARG() (i + 1 < argc ? argv[++i] : (usage(), (char *)0))
        if (!strcmp(argv[i], "--meta")) opt.meta_path = ARG();
        else if (!strcmp(argv[i], "--models")) opt.model_dir = ARG();
        else if (!strcmp(argv[i], "--rknn")) opt.rknn_path = ARG();
        else if (!strcmp(argv[i], "--runtime")) opt.runtime_lib = ARG();
        else if (!strcmp(argv[i], "--mode")) mode = ARG();
        else if (!strcmp(argv[i], "--no-fallback")) opt.flags &= ~NEURONTE_NPU_ALLOW_FALLBACK;
        else if (!strcmp(argv[i], "--zero-copy")) opt.flags |= NEURONTE_NPU_ZERO_COPY;
        else if (!strcmp(argv[i], "--no-pipeline")) opt.flags |= NEURONTE_NPU_NO_PIPELINE;
        else if (!strcmp(argv[i], "--init-flags")) opt.init_flags = (unsigned)strtoul(ARG(), NULL, 0);
        else if (!strcmp(argv[i], "--refine")) opt.refine_margin = strtof(ARG(), NULL);
        else if (!strcmp(argv[i], "--core-mask")) opt.core_mask = atoi(ARG());
        else if (!strcmp(argv[i], "--cpu")) cpu = atoi(ARG());
        else if (!strcmp(argv[i], "--warmup")) warmup = atoi(ARG());
        else if (!strcmp(argv[i], "--repeats")) repeats = atoi(ARG());
        else if (!strcmp(argv[i], "--passes")) passes = atoi(ARG());
        else if (!strcmp(argv[i], "--single")) single = atoi(ARG());
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(ARG(), NULL, 10);
        else if (!strcmp(argv[i], "--coverage-out")) coverage_out = ARG();
        else usage();
#undef ARG
    }
    if (!opt.meta_path || warmup < 0 || repeats < 1 || repeats > MAX_ROUNDS || passes < 1 || single < 0)
        usage();
    if (!strcmp(mode, "cpu"))
        opt.flags |= NEURONTE_NPU_FORCE_CPU;
    else if (strcmp(mode, "npu"))
        usage();
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof set, &set)) {
            fprintf(stderr, "cannot pin to CPU %d: %s\n", cpu, strerror(errno));
            return 1;
        }
    }

    FILE *o = stdout;
    fprintf(o, "{\"tool\":\"neuronte_npu_bench\",\"mode\":\"%s\",\"meta\":\"%s\",\"rknn\":\"%s\",\"cpu\":%d,", mode,
            opt.meta_path, opt.rknn_path ? opt.rknn_path : "", cpu);
    fprintf(o, "\"flags\":%d,\"init_flags\":%u,\"core_mask\":%d,\"warmup\":%d,\"repeats\":%d,\"passes\":%d,"
            "\"seed\":%llu,", opt.flags, opt.init_flags, opt.core_mask, warmup, repeats, passes, (unsigned long long)seed);
    fprintf(o, "\"hardware_start\":");
    snapshot(o, cpu);

    uint64_t w0 = clock_ns(CLOCK_MONOTONIC), c0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
    int rc = neuronte_npu_open(&opt, &n, reason, sizeof reason);
    uint64_t open_wall = clock_ns(CLOCK_MONOTONIC) - w0, open_cpu = clock_ns(CLOCK_PROCESS_CPUTIME_ID) - c0;
    if (rc) {
        fprintf(o, ",\"error\":\"open failed: %s\"}\n", reason);
        return 1;
    }
    int cw, ch, backend = neuronte_npu_backend(n);
    neuronte_npu_cell(n, &cw, &ch);
    fprintf(o, ",\"backend\":\"%s\",\"fallback_reason\":\"%s\",\"versions\":\"%s\",\"batch\":%d,\"cell\":[%d,%d]",
            neuronte_npu_backend_name(backend), backend == NEURONTE_NPU_RKNN ? "" : neuronte_npu_fallback_reason(n),
            neuronte_npu_versions(n), neuronte_npu_batch(n), cw, ch);
    fprintf(o, ",\"open_ns\":%llu,\"open_cpu_ns\":%llu", (unsigned long long)open_wall, (unsigned long long)open_cpu);
    struct neuronte_npu_info info;
    neuronte_npu_info(n, &info);
    fprintf(o, ",\"info\":{\"layer1\":\"%s\",\"zero_copy\":%d,\"pipeline\":%d,\"init_flags\":%u,"
            "\"refine_margin\":%g,\"core_mask\":%d,\"npu_cores\":%d,\"dtype\":\"%s\"}",
            info.layer1 == 2 ? "host" : info.layer1 == 1 ? "npu" : "cpu", info.zero_copy, info.pipeline,
            info.init_flags, (double)info.refine_margin, info.core_mask, info.npu_cores,
            info.dtype == 2 ? "int8" : info.dtype == 1 ? "float16" : "float32");

    size_t plane = (size_t)cw * (size_t)ch;
    uint32_t corpus[CORPUS];
    for (i = 0; i < CORPUS; i++)
        corpus[i] = 0x2800u + (uint32_t)i;
    shuffle(corpus, CORPUS, seed);
    float *out = malloc(plane * CORPUS * sizeof *out);
    struct round *rounds = calloc((size_t)(warmup + repeats), sizeof *rounds);
    if (!out || !rounds)
        return 1;

    struct neuronte_npu_timing t;
    w0 = clock_ns(CLOCK_MONOTONIC);
    c0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
    if (neuronte_npu_coverage(n, corpus, CORPUS, out, &t)) {
        fprintf(o, ",\"error\":\"first coverage call failed\"}\n");
        return 1;
    }
    fprintf(o, ",\"first\":{\"wall\":%llu,\"cpu\":%llu,\"stages\":",
            (unsigned long long)(clock_ns(CLOCK_MONOTONIC) - w0),
            (unsigned long long)(clock_ns(CLOCK_PROCESS_CPUTIME_ID) - c0));
    timing_json(o, &t);
    fprintf(o, "}");

    for (r = 0; r < warmup + repeats; r++) {
        struct round *rd = &rounds[r];
        w0 = clock_ns(CLOCK_MONOTONIC);
        c0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
        for (p = 0; p < passes; p++) {
            if (neuronte_npu_coverage(n, corpus, CORPUS, out, &t)) {
                fprintf(o, ",\"error\":\"coverage call failed in round %d\"}\n", r);
                return 1;
            }
            add(&rd->t, &t);
        }
        rd->wall = clock_ns(CLOCK_MONOTONIC) - w0;
        rd->cpu = clock_ns(CLOCK_PROCESS_CPUTIME_ID) - c0;
    }
    fprintf(o, ",\"glyphs_per_round\":%d,\"rounds\":[", CORPUS * passes);
    for (r = 0; r < warmup + repeats; r++) {
        fprintf(o, "%s{\"round\":%d,\"warmup\":%s,\"wall\":%llu,\"cpu\":%llu,\"stages\":", r ? "," : "", r - warmup,
                r < warmup ? "true" : "false", (unsigned long long)rounds[r].wall, (unsigned long long)rounds[r].cpu);
        timing_json(o, &rounds[r].t);
        fprintf(o, "}");
    }
    fprintf(o, "]");

    if (single) {
        float *one = malloc(plane * sizeof *one);
        fprintf(o, ",\"single\":[");
        for (i = 0; i < single; i++) {
            w0 = clock_ns(CLOCK_MONOTONIC);
            if (neuronte_npu_coverage(n, &corpus[i % CORPUS], 1, one, &t))
                return 1;
            fprintf(o, "%s%llu", i ? "," : "", (unsigned long long)(clock_ns(CLOCK_MONOTONIC) - w0));
        }
        fprintf(o, "]");
        free(one);
    }

    if (coverage_out) {
        FILE *f = fopen(coverage_out, "wb");
        int ok = f != NULL;
        for (i = 0; ok && i < CORPUS; i++) {
            int k;
            for (k = 0; k < CORPUS && corpus[k] != 0x2800u + (uint32_t)i; k++)
                ;
            ok = fwrite(out + (size_t)k * plane, sizeof *out, plane, f) == plane;
        }
        if (f)
            ok = fclose(f) == 0 && ok;
        fprintf(o, ",\"coverage_out\":\"%s\",\"coverage_written\":%s", coverage_out, ok ? "true" : "false");
    }
    fprintf(o, ",\"hardware_end\":");
    snapshot(o, cpu);
    fprintf(o, "}\n");
    neuronte_npu_close(n);
    free(out);
    free(rounds);
    return 0;
}
