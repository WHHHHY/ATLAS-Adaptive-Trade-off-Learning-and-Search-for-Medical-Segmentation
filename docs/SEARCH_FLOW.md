# Search Flow

## Stage 1

Command:

```bash
python -m search.collect_activations --program search/programs/example_organ.yaml --device cpu
```

Flow:

1. load typed program
2. load base checkpoint and runtime config
3. run calibration subset
4. collect `enc/skip/dec` activations
5. compute per-submodule metrics
6. aggregate into per-unit metrics
7. write metrics JSON under `runs/search_artifacts/metrics/`

## Stage 2

Command:

```bash
python -m search.search_loop \
  --program search/programs/example_organ.yaml \
  --metrics tests/fixtures/search/example_organ_metrics.json \
  --max-trials 1 \
  --dry-run
```

Flow:

1. load program YAML
2. load metrics JSON
3. evaluate mother baseline
4. proposer creates typed proposal
5. candidate builder serializes typed candidate
6. search space validates candidate
7. evaluator applies config patch and runs dry-run or quick eval
8. utility is computed against baseline resources
9. accept or reject against current best state
10. append `history.jsonl`

## MVP Notes

- pruning is expressed as controlled `model.prune_flags_list` patches
- quantization is expressed as typed bit allocation metadata and effective resource proxies
- `patch_embed` and `seg_head` remain protected at all times
- rollback is based on `best_state.json`, not git reset