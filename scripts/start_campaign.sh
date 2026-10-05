#!/usr/bin/env bash
# Restart the watchdog after FITLAB creates a new container.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
nohup setsid /usr/bin/python3 -u ops/watch_campaign.py \
  >> runs/campaign_watchdog_v1.log 2>&1 < /dev/null &
echo "Watchdog launch requested (PID $!). Check: python status.py"
