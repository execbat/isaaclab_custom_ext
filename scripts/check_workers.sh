#!/usr/bin/env bash
set -euo pipefail

MASTER=${1:-$(pgrep -f -n 'python .*isaac_hydra_ext\.scripts\.reinforcement_learning\.appo\.train')}
[[ -z "$MASTER" ]] && { echo "master not found"; exit 2; }

descendants() { local pid="$1"; echo "$pid"; for c in $(pgrep -P "$pid"); do descendants "$c"; done; }
PIDS=$(descendants "$MASTER" | tr '\n' ',' | sed 's/,$//')

echo "MASTER=$MASTER"
ps -o pid,ppid,stat,etime,cmd -p "$PIDS"

echo "---- summary ----"
ps -o stat= -p "$PIDS" | cut -c1 | sort | uniq -c | sed 's/^/count /'
LIVE=$(ps -o pid=,stat= -p "$PIDS" | awk -v m="$MASTER" '$1!=m {s=substr($2,1,1); if (s!="Z" && s!="T") c++} END{print c+0}')
echo "live descendants (not Z/T): $LIVE"
