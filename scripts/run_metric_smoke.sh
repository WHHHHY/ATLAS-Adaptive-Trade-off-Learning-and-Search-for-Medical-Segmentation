#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m unittest tests.test_metric_smoke