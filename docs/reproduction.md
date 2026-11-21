# 实验复现

## 复核存档指标

在仓库根目录用任意 Python 3.11 环境执行 python tools/recompute_metrics.py。脚本校验各实验使用相同的 500 个验证 ID，并从逐图交集、误检和漏检重算 micro-IoU、逐图平均 IoU、精确率、召回率、Dice 与小区域 IoU。该操作复核存档统计口径，使用已经保存的像素计数。原图分割使用固定阈值 0.5。小区域定义为编辑像素占比小于 1%。空并集的单图 IoU 在本地协议计为 1。

## 复跑冻结候选的训练

在候选 Notebook 所在目录运行以下命令。将 ARCHIVE_ZIP 替换为自己的课程数据路径。脚本以固定 split.json 复现正类加权 BCE 与 Dice 的 10 轮训练。完整预算应使用项目验证过的 GPU 环境。

~~~bash
cd experiments/phase6/submission_candidate
python -B train_candidate.py --archive "ARCHIVE_ZIP" --cache cache256 --out reproduction
~~~

reproduction/model.pth 是新生成的权重，test.ipynb 使用 model_path 指向该文件即可。原候选权重的散列单独记录。跨机器复跑时以同一训练协议和合理的数值接近程度评估复现。

## 使用原验证图重跑模型评价

将 experiments 的所需内容复制到独立运行目录，保持 baseline 与 phase2–phase6 为同级目录。课程 ZIP 的默认位置是这些目录旁的 AIGC Detection Dataset 2026.zip。模型权重放在表中对应的阶段目录。本地代码生成的曲线和数据预览留在独立运行目录。

| 阶段 | 入口与关键本地输入 |
| --- | --- |
| 1 | baseline/train.ipynb 可先处理课程 ZIP 并创建 cache256，再训练 V1。已有基线可在提供 baseline/model.pth 后运行 baseline/evaluate.py。 |
| 2 | phase2/diagnose.py 可选 v1 或 weighted，使用课程 ZIP、baseline/cache256 及对应 checkpoint。重训入口为 phase2/train_weighted.py，需为 weighted 运行保留空输出目录，并带上 V1 历史曲线。Phase 2 会为当前副本的 baseline 生成本地校验清单。原始 V1 清单在 docs/provenance/，用于溯源。 |
| 3 | phase3/prepare.py 核查并构建 cache384，phase3/train.py 执行 384 控制训练，phase3/evaluate.py 复核原图结果。准备时保留 baseline/cache256，训练和评价需要此前结果表。 |
| 4 | phase4/preflight.py 核对特征和学习流程，phase4/train.py 训练 SRM 对照，phase4/evaluate.py 评估。依赖此前的缓存、结果表、分组及对照权重。 |
| 5 | phase5/inference_preflight.py 验证真实图的滑窗推理，phase5/prepare.py 建裁块缓存，phase5/train.py 训练 A 和 B，phase5/evaluate.py 评估。需要 V1 权重及 Phase 3 数值统计，完整裁块缓存约 24 GB。 |
| 6 | phase6/architecture_audit.py 严格检查 V1、Weighted BCE 和 384 原权重。将其分别放在 baseline/model.pth、phase2/weighted/model.pth、phase3/resolution384/model.pth。用 phase6/train_deep.py --out RUN_DIR 新建三尺度实验，phase6/evaluate.py --out RUN_DIR 在课程 ZIP 上复核。已有 checkpoint 可放入运行目录并配套其 training_history.csv 来复算。 |

全量重训使用新输出目录或只包含所需输入的阶段副本。保留的训练历史会使 Phase 2–5 训练入口拒绝覆盖，所以重训副本应把目标阶段的旧输出留在档案目录。Phase 6 的 freeze_candidate.py 是本地打包脚本，会生成包含权重的 ZIP 和公测 RLE 验证产物，因此从独立运行目录操作。

PR-AUC 的统计网格需随数值一起报告。Phase 4 和 Phase 6 使用统一的 256 网格，Phase 5 保存的是原图分箱近似，二者可分别复核。
