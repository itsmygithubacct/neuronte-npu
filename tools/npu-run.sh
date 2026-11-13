#!/bin/sh
# npu-run.sh COMMAND [ARGS...]: run COMMAND as the current user with the
# render and video groups added, which is what the RKNPU device nodes
# (/dev/dri/renderD*, /dev/dri/card*) need, without changing any group
# membership on the system. Needs passwordless sudo for setpriv; no
# capabilities are inherited. Users already in those groups do not need it.
RG=$(getent group render | cut -d: -f3); VG=$(getent group video | cut -d: -f3)
if [ -z "$RG" ] || [ -z "$VG" ]; then
    echo "npu-run.sh: no render or video group" >&2
    exit 1
fi
exec sudo -n setpriv --reuid="$(id -u)" --regid="$(id -g)" --groups="$RG,$VG" --inh-caps=-all -- "$@"
