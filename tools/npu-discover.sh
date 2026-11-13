#!/bin/sh
# npu-discover.sh [LIBRKNNRT]: read-only report on a board's Rockchip NPU and
# whether neuronte-npu can use it. Changes nothing. Reads debugfs and the
# kernel log only when passwordless sudo works (sudo -n), and lists device
# users by process name only (never command lines, which can hold secrets).
#
# Sections: board, RKNPU device and driver, per-core completion interrupts
# (the check behind the backend's automatic core mask), power domains, NPU
# SRAM, frequencies, device access, runtime library, recent NPU errors.
set -u
lib=${1:-}
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -n"
say() { printf '%s\n' "$*"; }
sec() { printf '\n== %s\n' "$*"; }

sec board
tr -d '\0' < /proc/device-tree/model 2>/dev/null && echo
tr '\0' ' ' < /proc/device-tree/compatible 2>/dev/null && echo
uname -sr; grep -E '^PRETTY_NAME=' /etc/os-release 2>/dev/null | cut -d= -f2
say "cpus: $(nproc); parts: $(grep 'CPU part' /proc/cpuinfo | sort | uniq -c | awk '{printf "%sx%s ", $1, $5}')"

sec "RKNPU device"
npu=""
for c in /sys/class/drm/card*; do
    [ "$(basename "$(readlink "$c/device/driver" 2>/dev/null)")" = RKNPU ] || continue
    npu=$(basename "$(readlink "$c/device")"); say "drm $(basename "$c") -> platform device $npu (driver RKNPU)"
done
for r in /sys/class/drm/renderD*; do
    [ "$(basename "$(readlink "$r/device/driver" 2>/dev/null)")" = RKNPU ] && say "render node /dev/dri/$(basename "$r")"
done
[ -e /dev/rknpu ] && say "/dev/rknpu present (older RKNPU char device)"
if [ -z "$npu" ] && [ ! -e /dev/rknpu ]; then
    say "no RKNPU device: neuronte-npu will report cpu-fallback here"
    [ -r /sys/module/rknpu/version ] && say "rknpu module $(cat /sys/module/rknpu/version) loaded, but no device bound"
fi
v=$($SUDO cat /sys/kernel/debug/rknpu/version 2>/dev/null); [ -n "$v" ] && say "$v"
dmesg_ok=""; $SUDO dmesg >/dev/null 2>&1 && dmesg_ok=1
[ -n "$dmesg_ok" ] && $SUDO dmesg | grep -m1 -o 'Initialized rknpu [0-9.]* [0-9]*' | sed 's/^/driver: /'

if [ -n "$npu" ]; then
    dt=$(find /proc/device-tree/ -maxdepth 1 -name "*@${npu%%.*}" 2>/dev/null | head -1)
    sec "cores: device tree vs registered interrupts"
    [ -n "$dt" ] && say "device tree interrupt-names: $(tr '\0' ' ' < "$dt/interrupt-names" 2>/dev/null)"
    n=$(grep -c -E " $npu(\$|,| )" /proc/interrupts)
    say "interrupt lines with an $npu handler: $n"
    grep -E " $npu(\$|,| )" /proc/interrupts | sed -E 's/ +/ /g; s/^/  /'
    case "$n" in
        1) say "=> only core 0 can complete jobs. Pin core 0 (neuronte-npu does this when core_mask is 0);"
           say "   a job on core 1 or 2 hangs until the driver times out and resets the NPU." ;;
        3) say "=> three cores registered" ;;
    esac
    sec "power domains"
    $SUDO cat /sys/kernel/debug/pm_genpd/pm_genpd_summary 2>/dev/null | grep -E '^npu' || say "(needs debugfs)"
    sec "NPU SRAM"
    if [ -n "$dt" ]; then
        if [ -e "$dt/rockchip,sram" ]; then say "device tree: rockchip,sram set"
        else say "device tree: no rockchip,sram (no SRAM for the NPU)"; fi
    fi
    cfg=/boot/config-$(uname -r); [ -r "$cfg" ] && grep -E 'CONFIG_ROCKCHIP_RKNPU' "$cfg"
    sec frequencies
    for d in /sys/class/devfreq/*; do
        [ -e "$d/cur_freq" ] && say "$(basename "$d"): $(cat "$d/cur_freq") Hz, governor $(cat "$d/governor"), max $(cat "$d/max_freq")"
    done
    l=$($SUDO cat /sys/kernel/debug/rknpu/load 2>/dev/null); [ -n "$l" ] && say "$l"
    p=$($SUDO cat /sys/kernel/debug/rknpu/power 2>/dev/null); [ -n "$p" ] && say "power: $p"
    sec "device access"
    for node in /dev/dri/renderD* /dev/dri/card*; do
        [ -e "$node" ] && say "$(stat -c '%n %U:%G %a' "$node")"
    done
    say "you: $(id -un), groups: $(id -Gn)"
    if [ -n "$SUDO" ]; then
        users=$($SUDO find /proc/[0-9]*/fd -lname '/dev/dri/*' 2>/dev/null | cut -d/ -f3 | sort -u |
            while read -r pid; do cat "/proc/$pid/comm" 2>/dev/null; done | sort | uniq -c | tr -s ' \n' ' ')
        say "processes holding a DRM node (names only):${users:- none}"
    fi
fi

sec "runtime library"
for f in $lib /usr/lib/librknnrt.so /usr/lib/aarch64-linux-gnu/librknnrt.so; do
    if [ -z "$f" ] || [ ! -e "$f" ]; then continue; fi
    say "$f: $(strings "$f" 2>/dev/null | grep -m1 -o 'librknnrt version: [^ ]* ([^)]*)') sha256 $(sha256sum "$f" | cut -c1-16)"
done

if [ -n "$dmesg_ok" ]; then
    sec "recent NPU errors (kernel log)"
    $SUDO dmesg | grep -E 'RKNPU: (job timeout|soft reset|failed)|nobody cared|Disabling IRQ' | tail -6
fi
exit 0
