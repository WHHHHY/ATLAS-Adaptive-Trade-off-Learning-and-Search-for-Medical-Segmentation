# Experiment Schema v1

`runs/<run_id>/metrics.json` and `runs/history.jsonl` now share the same core schema for every run.

## Core fields

These fields exist in both `metrics.json` and each `history.jsonl` line.

| Field | Type | Unit | Meaning | Nullable |
| --- | --- | --- | --- | --- |
| `schema_version` | string | - | Schema version identifier. Current value: `experiment_schema_v1`. | no |
| `run_id` | string | - | Unique run directory id. | no |
| `timestamp` | string | ISO 8601 | Final write timestamp for the run summary. | no |
| `organ` | string | - | Target organ name from config. | yes |
| `dataset` | string | - | Dataset name, usually from `dataset.json` / plans. | yes |
| `search_method` | string | - | How this candidate was proposed. Defaults to `manual` if not provided. | no |
| `status` | string | - | Final run status: `success`, `failed`, or `invalid_trial`. | no |
| `seed` | integer | - | Random seed used for the run. | no |
| `parent_run_id` | string | - | Upstream run id for warm-start / mutation / proposal lineage. | yes |
| `val_dice` | number | Dice | Validation Dice used as the main segmentation quality metric. | yes |
| `latency_ms` | number | milliseconds per case | Validation inference latency. Currently mapped from `infer_time_ms_per_case`. | yes |
| `params_m` | number | millions of parameters | Total parameter count converted from `params_total / 1e6`. | yes |
| `peak_vram_mb` | number | MiB | Peak GPU memory allocated during training. | yes |
| `train_time_s` | number | seconds | End-to-end training time for the run. | yes |
| `utility` | number | score | Search-facing utility score. Current default is `val_dice` for successful runs. TODO: replace with a task-specific composite utility if needed. | yes |

## Status semantics

| Status | When written |
| --- | --- |
| `success` | Training finished and the run produced a valid final summary. |
| `failed` | The run entered training/execution and then failed because of runtime errors such as OOM, code errors, or checkpoint errors. |
| `invalid_trial` | The candidate is rejected before training starts, for example config load/preflight validation failure. |

## metrics.json structure

`metrics.json` is the detailed per-run summary.

It contains:

1. The shared top-level core fields.
2. Backward-compatible legacy fields such as `val_loss`, `train_loss_last`, `params_total`, `trainable_params`, `elapsed_time_sec`, and checkpoint paths.
3. Nested sections:
   - `summary`: normalized copy of the core fields.
   - `metrics`: metric-focused view.
   - `timing`: timing-focused view.
   - `resources`: parameter / FLOPs / VRAM view.
   - `artifacts`: output directory and checkpoint paths.
   - `context`: pruning, quantization, epochs completed, failure stage.
   - `error_info`: structured error metadata if the run is not successful.

## history.jsonl structure

`history.jsonl` is the global append-only experiment index.

Each line is one JSON object with:

1. The same shared core fields as `metrics.json`.
2. A small set of extra trace fields: `failure_stage`, `error`, `best_epoch`, `train_epochs_completed`, `save_dir`, `config_path`.

This file is intended for random search, TPE, agent proposal, and plotting scripts that need a compact summary without loading every `metrics.json`.

## Write timing

1. `metrics.json` is written in the `run_train.py` finalization block, regardless of run outcome.
2. `history.jsonl` is appended immediately after `metrics.json` write is attempted.
3. Failed and invalid trials must still produce structured JSON records.

## Nullable fields

These fields may be `null` when unavailable:

1. `parent_run_id`
2. `val_dice`
3. `latency_ms`
4. `params_m`
5. `peak_vram_mb`
6. `train_time_s`
7. `utility`
8. Any legacy metric/resource field that is not available for a failed or invalid trial

## Relationship between metrics.json and history.jsonl

1. `metrics.json` is the source of truth for one run.
2. `history.jsonl` is a compact index derived from the same normalized fields.
3. The shared core field names, units, and semantics are identical across both files.
4. Downstream tools should prefer the shared core fields when they only need one summary row per run.