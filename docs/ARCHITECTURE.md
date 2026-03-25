# Architecture

## Layers

### 1. Substrate / Core Layer

Files:

- `model.py`
- `dataset.py`
- `engine.py`
- `train.py`
- `validate.py`
- `checkpoint.py`
- `metrics.py`
- `losses.py`
- `run_train.py`
- `run_val.py`

Responsibility:

- SlimFormer model construction
- BTCV data loading
- training and validation loops
- checkpoint IO

This layer is intentionally stable and is not the place for open-ended search logic.

### 2. Search Runtime Layer

Files under `search/` except `search/proposers/`.

Responsibility:

- typed program schema
- activation collection
- metric computation and unit aggregation
- candidate schema and validation
- search-space validation
- evaluator and config patch application
- history logging
- search loop orchestration

Key rule:

- all mutable search behavior must flow through typed candidate actions and typed config patches

### 3. Proposer Layer

Files under `search/proposers/`.

Responsibility:

- convert metrics and history into proposals
- keep proposer implementations isolated from evaluator/runtime details

Current implementation:

- `search/proposers/heuristic.py`

Reserved extension point:

- future Qwen/VLM proposer implementations should live beside the heuristic proposer and return the same proposal / candidate schema

## Terminology

- `patch_embed`: canonical name for the protected patch embedding block
- `seg_head`: canonical name for the protected segmentation output head
- `unit_i`: search-time mutable unit corresponding to `(encoder_i, skip_i, decoder_i)`
- `program`: typed YAML describing search runtime settings
- `proposal`: proposer output before candidate serialization
- `candidate`: typed action bundle to be evaluated
- `config patch`: controlled config mutation derived from a candidate

## Runtime Artifacts

Runtime outputs belong under `runs/`:

- collected metrics JSON
- dry-run and quick-eval trial directories
- best candidate/state/metrics
- search history

Source directories should not accumulate trial outputs.