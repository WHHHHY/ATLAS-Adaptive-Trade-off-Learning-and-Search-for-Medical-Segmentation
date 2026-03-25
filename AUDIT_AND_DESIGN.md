# SlimFormer BTCV 极简仓库审计与设计稿

本文档分三部分：
1. 原始 nnUNet 集成版的 SlimFormer 训练/验证依赖审计
2. 极简仓库设计稿
3. 旧文件到新文件映射表

目标边界：
- 仅保留 BTCV 单任务、单卡、SlimFormer 训练/验证闭环
- 明确删除 nnUNet 通用 CLI、planner、DDP、多任务兼容层、复杂 trainer 继承和重后处理
- 新仓库仅依赖拷入的 BTCV 预处理产物，不再依赖原始 nnUNet 运行时路径

## 1. 依赖审计

### 1.1 训练入口调用链

原始训练调用链：

1. `nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerSlimFormer`
2. `nnunetv2/run/run_training.py:run_training_entry`
3. `nnunetv2/run/run_training.py:run_training`
4. `nnunetv2/run/run_training.py:get_trainer_from_args`
5. `nnunetv2/training/nnUNetTrainer/nnUNetTrainerSlimFormer.py:nnUNetTrainerSlimFormer`
6. `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:initialize`
7. `nnunetv2/training/nnUNetTrainer/nnUNetTrainerSlimFormer.py:build_network_architecture`
8. `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:get_dataloaders`
9. `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:run_training`
10. `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:train_step`

对极简仓库真正有用的核心逻辑：
- 解析 `dataset.json`、`nnUNetPlans.json`、`splits_final.json`
- 读取 `nnUNetPlans_3d_fullres` 预处理结果
- 构建 SlimFormer
- 组合 Dice + CE loss
- 单卡 epoch/batch 循环
- checkpoint 保存与恢复

### 1.2 验证入口调用链

原始验证存在两条路径：

1. 训练中的 patch 级验证
   - `nnUNetTrainer.run_training`
   - `nnUNetTrainer.validation_step`
   - `nnUNetTrainer.on_validation_epoch_end`
   - 这里产出的是 online pseudo dice，不是最终导出的全体积 Dice

2. 训练结束后的正式验证
   - `nnUNetTrainer.perform_actual_validation`
   - `nnUNetPredictor.predict_sliding_window_return_logits`
   - `export_prediction_from_logits`
   - `compute_metrics_on_folder`

对极简仓库保留策略：
- 保留 patch 级验证，用于训练期间快速反馈 `val_loss` 和 `pseudo_dice`
- 重写正式验证为极简版 tiled inference + Dice 汇总
- 不再保留 nnUNet 的 predictor、导出池、gaussian cache、next-stage cascade 逻辑

### 1.3 SlimFormer 真正依赖的文件

#### 必须读取的原始文件

1. `nnUNet/nnUNet_preprocessed/Dataset002_BTCV/dataset.json`
   - 标签定义、通道数、数据集名
2. `nnUNet/nnUNet_preprocessed/Dataset002_BTCV/nnUNetPlans.json`
   - `3d_fullres` 的 patch size、batch size、spacing、data identifier
3. `nnUNet/nnUNet_preprocessed/Dataset002_BTCV/splits_final.json`
   - fold 划分
4. `nnUNet/nnUNet_preprocessed/Dataset002_BTCV/nnUNetPlans_3d_fullres/*`
   - 预处理后的 `b2nd/pkl` 训练样本

#### 必须参考并抽取逻辑的原始代码

1. `nnUNet/nnunetv2/training/nnUNetTrainer/variants/slimformer.py`
   - SlimFormer 网络主体
2. `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerSlimFormer.py`
   - SlimFormer 与 plans/config 的对接方式
3. `nnUNet/nnunetv2/training/dataloading/nnunet_dataset.py`
   - `.b2nd/.npz/.npy` 读取接口
4. `nnUNet/nnunetv2/training/dataloading/data_loader.py`
   - patch 采样和 foreground oversampling
5. `nnUNet/nnunetv2/training/loss/dice.py`
   - Dice loss 和 TP/FP/FN 计算
6. `nnUNet/nnunetv2/training/loss/compound_losses.py`
   - Dice + CE 组合方式
7. `nnUNet/nnunetv2/training/lr_scheduler/polylr.py`
   - poly lr
8. `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
   - `run_training`、`train_step`、`validation_step`、`save_checkpoint`、`load_checkpoint`、`do_split`

#### 运行时外部依赖

- `torch`
- `monai`
- `einops`
- `mamba-ssm`
- `causal-conv1d`
- `blosc2`
- `pyyaml`
- `numpy`

### 1.4 必须保留 / 可以合并 / 可以删除

#### 必须保留

1. BTCV 预处理产物读取
2. SlimFormer 模型定义
3. Dice + CE loss
4. Dice 指标统计
5. 单卡训练循环
6. 单卡验证循环
7. checkpoint
8. config 读写接口
9. `splits_final.json` fold 划分

#### 可以合并

1. `PlansManager` 的使用场景
   - 极简版合并为 `config.py` 的 plans 解析
2. `nnUNetDatasetNumpy` 和 `nnUNetDatasetBlosc2`
   - 合并为一个统一的 case loader
3. `train_step` / `validation_step` / `run_training`
   - 合并到 `engine.py`
4. `save_checkpoint` / `load_checkpoint`
   - 合并到 `checkpoint.py`
5. patch 采样 / crop-pad / 读取数组
   - 合并到 `preprocess.py + dataset.py`

#### 可以删除

1. `nnUNetv2_train` CLI
2. `recursive_find_python_class`
3. `PlansManager` 全量能力
4. `nnUNetPredictor`
5. `compute_metrics_on_folder`
6. DDP / SyncBN / 多卡 oversampling 修正
7. cascade / next-stage
8. batchgenerators v2 的复杂增强流水线
9. logger/plot/debug dump 体系
10. export worker pool / background segmentation export

#### 不确定但当前先不保留

1. Gaussian weighted sliding window
   - 当前极简版先用无 gaussian tiled inference
   - 风险：边界表现可能低于原始 nnUNet 正式验证
2. 原始 CTNormalization 复刻
   - 当前极简版默认直接消费 nnUNet 已预处理数据
   - 风险：如果后续切换到原始 NIfTI 输入，需要补 raw preprocessing

## 2. 极简仓库设计稿

### 2.1 新目录树

```text
nnUNet(integration)/
  AUDIT_AND_DESIGN.md
  README.md
  requirements.txt
  config.py
  preprocess.py
  dataset.py
  model.py
  losses.py
  metrics.py
  checkpoint.py
  engine.py
  train.py
  validate.py
  copy_btcv.py
  configs/
    btcv_slimformer.yaml
  data/
    Dataset002_BTCV/
      dataset.json
      nnUNetPlans.json
      splits_final.json
      gt_segmentations/
      nnUNetPlans_3d_fullres/
```

### 2.2 每个文件职责

| 新文件 | 职责 | 替代原项目逻辑 |
| --- | --- | --- |
| `AUDIT_AND_DESIGN.md` | 审计、设计、映射和风险记录 | 替代分散在 trainer/plans/运行脚本里的人工追踪工作 |
| `README.md` | 最小仓库使用说明与文件职责说明 | 替代 nnUNet CLI 文档入口 |
| `requirements.txt` | 极简依赖定义 | 替代 nnUNet 全量 setup 依赖 |
| `config.py` | 读取 YAML、解析 `dataset.json/plans/splits`、补全默认值 | 替代 `PlansManager`、CLI 参数拼装和 trainer 初始化参数解析 |
| `preprocess.py` | 统一读取 `.b2nd/.npz/.npy`、patch bbox 采样、crop/pad | 替代 `nnunet_dataset.py` 和 `data_loader.py` 的底层数据处理 |
| `dataset.py` | 构建 patch dataset 与 full-volume dataset | 替代 `do_split/get_tr_and_val_datasets/get_dataloaders` 的数据部分 |
| `model.py` | SlimFormer 定义与 config 驱动构建 | 替代 `variants/slimformer.py` 和 `nnUNetTrainerSlimFormer.build_network_architecture` |
| `losses.py` | Dice、Dice+CE loss | 替代 `dice.py/compound_losses.py` |
| `metrics.py` | Dice 指标统计和 summary 输出 | 替代 `validation_step` 的指标部分与 `evaluate_predictions.py` 的最小需求 |
| `checkpoint.py` | latest/best checkpoint 保存恢复 | 替代 trainer checkpoint 逻辑 |
| `engine.py` | epoch 训练、patch 验证、full-volume tiled 验证 | 替代 `nnUNetTrainer.run_training/train_step/validation_step/perform_actual_validation` |
| `train.py` | 训练入口 | 替代 `nnUNetv2_train` + 自定义 trainer 绑定 |
| `validate.py` | 独立验证入口 | 替代 `--val` 和部分 predictor/evaluator 逻辑 |
| `copy_btcv.py` | 将 BTCV 预处理数据复制进新仓库 | 替代外部手工复制步骤 |
| `configs/btcv_slimformer.yaml` | Agent 可搜索的超参数入口 | 替代 nnUNet CLI 参数和 trainer 常量 |

### 2.3 风险点和待确认项

1. 当前极简版默认训练输入来自 nnUNet 已预处理的 BTCV 数据，而不是原始 `nii.gz`。
2. 当前正式验证采用极简 tiled inference，不包含 nnUNet 的 gaussian blending。
3. 当前训练增强仅保留 patch 随机采样与 foreground oversampling，不保留 nnUNet 的复杂增强堆栈。
4. SlimFormer 仍依赖 `monai + mamba-ssm`，这部分没有进一步去依赖化。
5. 如果后续 Agent 要搜索 raw preprocessing 超参数，需要再补一个 raw BTCV 预处理脚本。
6. 目前默认读取 `3d_fullres`，不再兼容 2d / lowres / cascade。

## 3. 旧文件到新文件映射表

| 旧文件 | 新文件 | 处理方式 |
| --- | --- | --- |
| `nnUNet/nnunetv2/run/run_training.py` | `train.py` | 保留单卡训练入口，删除 CLI 框架层 |
| `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` | `engine.py`, `checkpoint.py`, `dataset.py` | 拆分保留训练/验证/checkpoint/切分逻辑 |
| `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerSlimFormer.py` | `model.py`, `config.py` | 将模型构建与 plans 适配拆出 |
| `nnUNet/nnunetv2/training/nnUNetTrainer/variants/slimformer.py` | `model.py` | 直接抽出 SlimFormer 主体 |
| `nnUNet/nnunetv2/training/dataloading/nnunet_dataset.py` | `preprocess.py` | 合并为统一 case loader |
| `nnUNet/nnunetv2/training/dataloading/data_loader.py` | `preprocess.py`, `dataset.py` | 保留 patch 采样，删除 batchgenerators 依赖 |
| `nnUNet/nnunetv2/training/loss/dice.py` | `losses.py`, `metrics.py` | 分离训练 loss 与验证指标 |
| `nnUNet/nnunetv2/training/loss/compound_losses.py` | `losses.py` | 仅保留 Dice+CE |
| `nnUNet/nnunetv2/training/lr_scheduler/polylr.py` | `engine.py` | 内联 poly lr |
| `nnUNet/nnunetv2/utilities/plans_handling/plans_handler.py` | `config.py` | 仅保留最小 JSON plans 解析 |
| `nnUNet/nnunetv2/evaluation/evaluate_predictions.py` | `metrics.py` | 保留 Dice 汇总 |
| `nnUNet/nnunetv2/inference/predict_from_raw_data.py` | `validate.py`, `engine.py` | 用极简 tiled inference 替代 |
| `nnUNet/nnUNet_preprocessed/Dataset002_BTCV/*` | `data/Dataset002_BTCV/*` | 直接复制训练/验证所需数据 |
