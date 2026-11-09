#!/usr/bin/env bash
# Refuse to run when free disk space is too low.
# Inside WSL2, `df` on / shows the size of the virtual disk (about 1 TB), NOT your real Windows free space.
# The real limit is the Windows drive, which is visible at /mnt/c, so both are checked.
# usage: check_disk.sh [dir]      env: MIN_FREE_GB (default 20)
dir="${1:-.}"
min="${MIN_FREE_GB:-20}"
ok=1
check() {
  [ -d "$1" ] || return 0
  free_gb=$(df -Pk "$1" | awk 'NR==2{printf "%d", $4/1024/1024}')
  if [ "$free_gb" -lt "$min" ]; then
    echo "ERROR: only ${free_gb} GB free on $1 (need at least ${min} GB). Free some space or set MIN_FREE_GB lower." >&2
    ok=0
  fi
}
check "$dir"
check /mnt/c
[ "$ok" = 1 ]
