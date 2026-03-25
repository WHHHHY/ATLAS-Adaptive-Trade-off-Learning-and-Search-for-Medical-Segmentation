# SlimFormer Typed Search MVP

这个目录是从 nnUNet SlimFormer 训练链路中抽取出的极简单任务仓库。

保留内容：
- BTCV 预处理数据读取
- SlimFormer 模型定义
- 训练循环
- patch 验证循环
- full-volume 验证循环
- Dice + CE loss
- Dice 指标
- checkpoint
- YAML config 接口

删除内容：
- nnUNet 通用 CLI
- planner
- DDP
- 多任务兼容层
- 复杂 trainer 继承
- batchgenerators 增强框架
- predictor/export/evaluator 工程壳层

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 确认 BTCV 数据已经在当前仓库下

默认路径：

```text
data/Dataset002_BTCV
```

如果还没有复制，可以运行：

```bash
python copy_btcv.py
```

3. 开始训练

```bash
python run_train.py --config config/base.yaml
```

4. 独立验证

```bash
python run_val.py --config config/base.yaml --ckpt runs/<run_id>/checkpoints/best.pt
```

## AutoResearch Mirror

这个目录是给后续 `AutoResearch` / git / proposer 扩展准备的长期工作副本。

兼容策略：

- 原始工作目录仍保留在 `/root/autodl-tmp/nnUNet(integration)`
- 这里保留相同的 CLI、测试和模块边界，便于后续独立初始化 git
- 当前模板仍默认引用原始目录中的现有 BTCV 数据和已训练 checkpoint，以保证不改功能即可运行

## Typed Search MVP

当前 repo 已包含一个受控 typed search MVP，分成三层：

1. substrate / core
	`model.py`, `train.py`, `validate.py`, `engine.py`, `dataset.py`, `checkpoint.py`
2. search runtime
	`search/program_schema.py`, `search/collect_activations.py`, `search/candidate_builder.py`, `search/evaluator.py`, `search/search_loop.py` 等
3. proposer layer
	`search/proposers/`

最小运行方式：

1. 生成 activation metrics

```bash
python -m search.collect_activations --program search/programs/example_organ.yaml --device cpu
```

2. 运行 dry-run search loop

```bash
python -m search.search_loop \
  --program search/programs/example_organ.yaml \
  --metrics tests/fixtures/search/example_organ_metrics.json \
  --max-trials 1 \
  --dry-run
```

3. 运行 smoke tests

```bash
python -m unittest tests.test_metric_smoke tests.test_search_smoke
```

运行产物默认写入 `runs/`，源码目录不应再承载 trial 输出。

## AutoResearch Template

更贴近长期 AutoResearch 布局的模板配置见：

- `configs/search/autoresearch_example_organ.yaml`

它保持与当前 `search/programs/example_organ.yaml` 等价的功能范围，只是把字段组织得更适合后续 proposer 和 runtime 扩展。

## MVP 范围

当前已完成：

- metrics JSON -> typed candidate
- 自动发现 mutable units
- 强制冻结 `patch_embed` / `seg_head`
- heuristic proposer
- adaptive bit allocator
- config patch 映射
- dry-run evaluator
- quick evaluator 框架
- utility 计算
- accept/reject + best_state rollback
- history.jsonl

当前仍是 TODO：

- real quantized kernel
- VLM proposer
- accept 后自动重采 activation metrics
- 更完整的 latency/profile 接入

## 文件职责与原逻辑替代

| 文件 | 职责 | 替代原项目逻辑 |
| --- | --- | --- |
| `config.py` | 读取 YAML，补全 dataset/plans/splits 元数据 | 替代 nnUNet 的 plans manager 和 CLI 拼参 |
| `preprocess.py` | 读取 `.b2nd/.npz/.npy`，做 patch 采样与 crop-pad | 替代 dataloader 底层预处理 |
| `dataset.py` | 生成 train/val patch 数据集与 full-volume 验证集 | 替代 trainer 数据装配 |
| `model.py` | SlimFormer 与 token mixer 超参数入口 | 替代 trainer 中内嵌的模型定义 |
| `losses.py` | Dice + CE 训练损失 | 替代 nnUNet loss 模块 |
| `metrics.py` | Dice 指标和 summary 输出 | 替代 nnUNet 评估最小子集 |
| `checkpoint.py` | 保存与加载 latest/best checkpoint | 替代 trainer checkpoint 管理 |
| `engine.py` | epoch loop、patch validation、full-volume tiled validation | 替代 trainer 主循环 |
| `train.py` | 最小训练入口 | 替代 `nnUNetv2_train` |
| `validate.py` | 最小验证入口 | 替代 `--val` 和 predictor 调度 |
| `copy_btcv.py` | 复制 BTCV 预处理数据到当前仓库 | 替代外部手工迁移数据 |
| `AUDIT_AND_DESIGN.md` | 依赖审计、设计说明、旧文件到新文件映射表 | 替代口头迁移说明 |

## 配置接口

配置文件保留了后续 Agent 搜索 pruning、quantization 和结构超参数所需的稳定入口：

- `model.base_channels`
- `model.num_heads`
- `model.metaformer_layers`
- `model.token_mixer`
- `model.prune_flags_list`
- `train.lr`
- `train.weight_decay`
- `train.epochs`
- `train.batch_size`
- `validation.overlap`

## 已知差异

1. 训练与验证直接读取 `nnUNetPlans_3d_fullres` 预处理结果。
2. 当前 full-volume 验证默认不使用 gaussian blending。
3. 当前不复刻 nnUNet 的重型数据增强，仅保留随机 patch + 前景过采样。
4. 如果后续要直接从原始 NIfTI 训练，需要额外补一个 raw preprocessing 脚本。
