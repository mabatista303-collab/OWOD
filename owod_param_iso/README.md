# OWOD Parameter Isolation

这个文件夹集中放了“基于原 OWOD 代码的参数隔离增量学习改动”，不去改原始 `tools/train_net.py` 主干流程。

## 文件说明

- `owod_param_iso/config.py`
  - 给原配置树追加参数隔离相关开关
- `owod_param_iso/parameter_isolation.py`
  - 真正的参数冻结和旧类梯度屏蔽逻辑
- `owod_param_iso/trainer.py`
  - 基于原 `DefaultTrainer` 的轻量包装
- `owod_param_iso/train_net.py`
  - 单独训练入口，保持原项目训练方式不变，只替换增量阶段策略
- `owod_param_iso/configs/`
  - 参数隔离版任务配置

## 训练入口

```bash
python owod_param_iso/train_net.py --num-gpus 1 --config-file owod_param_iso/configs/t2_train_param_iso.yaml MODEL.WEIGHTS ./output/t1/model_final.pth OUTPUT_DIR ./output/t2_param_iso
```

## 方法

- 冻结 `backbone`
- 冻结 `proposal_generator`
- 冻结 ROI box feature extractor
- 对旧类别分类器参数做梯度 mask
- 对旧类别 bbox 回归参数做梯度 mask
