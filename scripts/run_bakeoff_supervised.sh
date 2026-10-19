#!/usr/bin/env bash
# Rerun the bake-off until it succeeds. Laptop GPUs can drop out mid-run, which leaves the
# process's CUDA context unusable; a fresh process resumes from the per-fold checkpoints.
#
#   scripts/run_bakeoff_supervised.sh <log file> <candidates>
set -u
log="$1"
candidates="$2"
max_attempts="${MAX_ATTEMPTS:-10}"

: > "$log"
for attempt in $(seq 1 "$max_attempts"); do
    echo "=== supervisor: attempt $attempt/$max_attempts at $(date '+%H:%M:%S') ===" >> "$log"
    python -m src.bakeoff.run select --candidates "$candidates" >> "$log" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "exit 0" >> "$log"
        exit 0
    fi
    echo "=== supervisor: attempt $attempt exited $status; restarting in 60 s ===" >> "$log"
    sleep 60
done
echo "exit 1" >> "$log"
exit 1
