#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m search.search_loop \
  --program search/programs/example_organ.yaml \
  --metrics tests/fixtures/search/example_organ_metrics.json \
  --max-trials 1 \
  --dry-run