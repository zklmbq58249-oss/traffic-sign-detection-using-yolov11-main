# YOLO26 实验摘要

本文件记录 `migrate-yolo26` 支线中保留的核心实验。三次训练使用同一数据配置、YOLO26n、随机种子 42、50 个 epoch、batch size 16、4 个 workers 和 GPU `0`；因此表中变化主要对应每次实验的单一改动。

## 实验对比

| 实验 | 训练脚本 | 关键改动 | 最佳验证 Precision / Recall | 最佳验证 mAP50 / mAP50-95 | 测试集结果 |
| --- | --- | --- | --- | --- | --- |
| Baseline | `train_yolo26.py` | `imgsz=416`，保留默认增强 | 0.82565 / 0.88541 | 0.92124 / 0.78655 | P 0.85871，R 0.82901，mAP50 0.89291，mAP50-95 0.75921 |
| EXP-1 | `train_yolo26_exp1_640.py` | 仅将输入尺寸从 416 提升到 640 | 0.92937 / 0.86834 | 0.94811 / 0.81278 | 测试输出已保留；该次运行未单独写出标量 JSON |
| EXP-2 | `train_yolo26_exp2_640_no_fliplr.py` | 基于 EXP-1，关闭水平翻转：`fliplr=0.0` | 0.88672 / 0.93032 | 0.96676 / 0.82969 | P 0.90180，R 0.87395，mAP50 0.93591，mAP50-95 0.79451 |

验证集数值取各运行 `results.csv` 中 mAP50-95 最高的 epoch；测试集数值取对应的 `test_metrics.json`。EXP-1 的测试目录保留了曲线和混淆矩阵，但原始运行没有保存标量指标文件。

## 变更要点

- Baseline 建立 416px 的 YOLO26n 迁移基线，并在训练后使用 `best.pt` 做 end-to-end 测试。
- EXP-1 只改变 `imgsz=640`，用于隔离更高输入分辨率的影响。
- EXP-2 保持 EXP-1 的 640px 设置，关闭 `fliplr`，避免数字限速标志被水平镜像，作为针对数字标志的增强消融实验。
- `resume_yolo26_exp2_640_no_fliplr.py` 用于从 EXP-2 的 `last.pt` 恢复训练并重新执行测试评估。

## 保留的结果文件

每个核心运行仅保留参数 (`args.yaml`)、训练曲线与混淆矩阵、`results.csv`/`results.png`、测试指标（若已生成）及最佳权重 `weights/best.pt`。训练批次图、验证样本图、`last.pt`、旧 YOLO11 结果和本地数据集不纳入仓库。
