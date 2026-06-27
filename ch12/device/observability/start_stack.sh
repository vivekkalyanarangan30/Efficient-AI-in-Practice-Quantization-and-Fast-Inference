#!/usr/bin/env bash
# Start the local observability stack (loopback only).
# Refuses to start if any port is already busy; refuses to run as root.
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "do not run as root — services must run as the project user" >&2
  exit 2
fi

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBS="$BASE/observability"
mkdir -p "$OBS/pids"

for port in 9090 9091 3000; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port already in use — refusing to start"
    exit 3
  fi
done

# Pushgateway
nohup "$OBS/bin/pushgateway" \
  --web.listen-address=127.0.0.1:9091 \
  --web.config.file="$OBS/pushgateway/web.yml" \
  --persistence.file="$OBS/pushgateway/data.pgw" \
  >> "$OBS/pushgateway.log" 2>&1 &
echo $! > "$OBS/pids/pushgateway.pid"

# Prometheus
nohup /opt/homebrew/bin/prometheus \
  --config.file="$OBS/prometheus/prometheus.yml" \
  --storage.tsdb.path="$OBS/prometheus/data" \
  --storage.tsdb.retention.time=12h \
  --web.listen-address=127.0.0.1:9090 \
  --web.enable-lifecycle \
  >> "$OBS/prometheus.log" 2>&1 &
echo $! > "$OBS/pids/prometheus.pid"

# Grafana admin password from file -> env, never on disk in grafana.ini
GF_SECURITY_ADMIN_PASSWORD="$(cat "$OBS/secrets/grafana.admin")" \
  nohup /opt/homebrew/bin/grafana server \
  --homepath=/opt/homebrew/share/grafana \
  --config="$OBS/grafana/grafana.ini" \
  >> "$OBS/grafana.log" 2>&1 &
echo $! > "$OBS/pids/grafana.pid"

sleep 4
echo "started:"
echo "  pushgateway  pid=$(cat "$OBS/pids/pushgateway.pid")  https://127.0.0.1:9091"
echo "  prometheus   pid=$(cat "$OBS/pids/prometheus.pid")   http://127.0.0.1:9090"
echo "  grafana      pid=$(cat "$OBS/pids/grafana.pid")      http://127.0.0.1:3000"
