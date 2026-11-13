#!/bin/sh
# run-set.sh SPECFILE OUTDIR: run a set of benchmarks back to back on the board.
#
# SPECFILE, one run per line (blank lines and # comments are skipped):
#     cpu STEM [BENCH OPTIONS]        reference CPU path for MODELDIR/STEM.meta
#     npu STEM[@TAG] [BENCH OPTIONS]  NPU run of MODELDIR/STEM.meta + STEM.rknn
# Options given on a line replace the defaults below (--warmup 2 --repeats 9
# --passes 4); a b1 model (STEM contains -b1-) defaults to single-glyph
# latency (--warmup 1 --repeats 3 --passes 1 --single 64). @TAG names a
# second run of the same model, e.g. "npu X@r05 --refine 0.05".
#
# Environment (defaults in brackets):
#     MODELDIR  directory of .meta/.rknn/.l1 files               [.]
#     NRT_DIR   directory of the .nrt networks the metas name    [models]
#     BENCH     neuronte_npu_bench binary                        [build/neuronte_npu_bench]
#     RUNTIME   librknnrt.so                                     [librknnrt.so]
#     CPU       core every run is pinned to                      [unpinned]
#     RUN       command prefix, e.g. tools/npu-run.sh for device access  [none]
#     TIMEOUT   seconds per run                                  [1200]
#     LOCK      a lock file shared with other benchmark users: the whole set
#               runs under flock on it (waiting for other holders), and the
#               owner note LOCK.owner (LOCK minus .lock) names this run [none]
#     OWNER     name written to the owner note                    [run-set]
#
# OUTDIR gets KIND-STEM[-TAG].json (every round, stage times, hardware
# snapshots), .f32 (coverage of the 256 Braille scalars), .err and .freq
# (each devfreq device's cumulative milliseconds per frequency before and
# after the run, from trans_stat; tools/summarize.py turns it into the mean
# DDR/NPU/GPU clock over the run) for each run, start.txt/end.txt (time, kernel, load, binary checksums), progress.txt,
# and exit-status (0 when every run succeeded) last.
set -u
[ $# -eq 2 ] || { echo "usage: run-set.sh SPECFILE OUTDIR" >&2; exit 2; }
if [ -n "${LOCK:-}" ] && [ -z "${RUN_SET_LOCKED:-}" ]; then
    RUN_SET_LOCKED=1 exec flock "$LOCK" sh "$0" "$@"
fi
spec=$1; out=$2
MODELDIR=${MODELDIR:-.}; NRT_DIR=${NRT_DIR:-models}; BENCH=${BENCH:-build/neuronte_npu_bench}
RUNTIME=${RUNTIME:-librknnrt.so}; CPU=${CPU:-}; RUN=${RUN:-}; TIMEOUT=${TIMEOUT:-1200}
mkdir -p "$out"
snapshot() {
    date -u +%FT%T.%3NZ; uname -r; echo "load $(cat /proc/loadavg)"
    for z in /sys/class/thermal/thermal_zone*; do
        [ -r "$z/temp" ] && echo "thermal $(cat "$z/type") $(cat "$z/temp")"
    done
    for d in /sys/class/devfreq/*; do
        [ -r "$d/cur_freq" ] && echo "devfreq $(basename "$d") $(cat "$d/cur_freq")"
    done
    sha256sum "$BENCH" "$RUNTIME" 2>/dev/null
}
freqstat() {  # "DEVICE HZ MS" for every devfreq frequency: time spent there since boot
    for d in /sys/class/devfreq/*; do
        [ -r "$d/trans_stat" ] || continue
        awk -v dev="${d##*/}" '{ sub(/^\*/, "") } $1 ~ /^[0-9]+:$/ { sub(/:$/, "", $1); print dev, $1, $NF }' \
            "$d/trans_stat" 2>/dev/null
    done
}
if [ -n "${LOCK:-}" ]; then
    note="${LOCK%.lock}.owner"
    echo "${OWNER:-run-set} pid=$$ start=$(date -u +%FT%TZ)" > "$note"
fi
snapshot > "$out/start.txt" 2>&1
pin=""; [ -n "$CPU" ] && pin="--cpu $CPU"
fail=0
while read -r kind stem rest; do
    case "$kind" in ''|'#'*) continue ;; esac
    name=${stem%@*}; tag=""; [ "$name" != "$stem" ] && tag="-${stem#*@}"
    if [ -n "$rest" ]; then opts=$rest
    else case "$name" in *-b1-*) opts="--warmup 1 --repeats 3 --passes 1 --single 64" ;;
                         *) opts="--warmup 2 --repeats 9 --passes 4" ;; esac
    fi
    f="$out/$kind-$name$tag"
    case "$kind" in
        cpu) set -- --mode cpu --meta "$MODELDIR/$name.meta" ;;
        npu) set -- --mode npu --no-fallback --meta "$MODELDIR/$name.meta" --rknn "$MODELDIR/$name.rknn" ;;
        *) echo "bad spec line: $kind $stem" >> "$out/progress.txt"; fail=1; continue ;;
    esac
    freqstat | sed 's/^/before /' > "$f.freq"
    rc=0
    # shellcheck disable=SC2086  # RUN, pin and opts are word lists
    timeout "$TIMEOUT" $RUN "$BENCH" --models "$NRT_DIR" --runtime "$RUNTIME" $pin "$@" $opts \
        --coverage-out "$f.f32" < /dev/null > "$f.json" 2> "$f.err" || rc=$?
    freqstat | sed 's/^/after /' >> "$f.freq"
    if [ "$rc" -eq 0 ]; then
        echo "$(date -u +%T) ok $kind $stem" >> "$out/progress.txt"
    else
        echo "$(date -u +%T) FAILED($rc) $kind $stem" >> "$out/progress.txt"; fail=1
    fi
done < "$spec"
snapshot > "$out/end.txt" 2>&1
echo "$fail" > "$out/exit-status"
exit "$fail"
