#!/usr/bin/env bash
# Stop the local observability stack. Idempotent.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$BASE/observability/pids"

for name in grafana prometheus pushgateway; do
  f="$PID_DIR/$name.pid"
  if [ -f "$f" ]; then
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      # Wait briefly for graceful shutdown before SIGKILL
      for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
      echo "stopped $name (pid=$pid)"
    fi
    rm -f "$f"
  fi
done
