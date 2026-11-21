# CS5489 Phase 6｜提交基线冻结与整图对照

## 课程交付与 Kaggle 待核实项

| 项目 | 已查到的要求或状态 |
| --- | --- |
| 任务和选模 | 课件第 3 页要求从 RGB 预测像素级 0/1 掩码，第 4–5 页要求留出训练数据做验证，500 张图进入 Kaggle，另外 500 张留作 TA 评估。Lecture7_1.pdf（本地留存） |
| 截止及文件 | 课件第 5 页给出的 Canvas 截止时间是 **2026-12-13 23:59:59**。需提交含可复现训练代码、执行输出、观察和分析的项目报告 Notebook，以及可运行的 test.ipynb 和 model.pth。TA 公告要求附 Python 环境说明 TXT。课程课件（本地留存） · [TA 提供的环境清单](../../environment/ta_requirements_reference.txt) |
| 推理要求 | TA 模板支持把模型定义放在随附文件中。Notebook 应只需改路径即可运行，输入为测试 RGB 图和模型权重，输出为原图尺寸的二值掩码。TA 推理模板（本地留存） |
| TA 推理 CSV 和 RLE | 表头按顺序为 ID,Predicted，ID 用原文件名去掉扩展名，保留前导零。掩码按列扫描、从 1 开始做起点/长度编码，空掩码写空字符串。TA 推理模板的编码和导出单元（本地留存） |
| Kaggle 文件格式 | **UNVERIFIED**。数据 ZIP 的 sample_submission.csv 与 test.csv 逐字节相同，表头为 ID,annotation,height,width,Usage。它与 TA 模板不同。登录课程竞赛后查看 **Data → sample submission**，核实列名、列序、RLE 及空值表示。课程数据 ZIP（本地留存） · Kaggle 邀请（课程渠道） |
| Kaggle 官方分数 | **UNVERIFIED**。公开访问课程邀请链接与按课程信息检索，均未取得评分条款。请查看 **Overview → Evaluation**，确认像素指标名称、公式、逐图或全局聚合、空 mask 规则，以及公榜和 TA 保留集是否采用同一公式。课件第 5–6 页（本地留存） · Kaggle 邀请（课程渠道） |
| 模型与数据许可 | TA 提供了 PyTorch 模型接口，所查课件未指定架构。外部数据与预训练权重的许可 **UNVERIFIED**，需查看竞赛 **Rules**。本阶段只用课程数据，模型均为随机初始化训练。TA 模板（本地留存） · Kaggle 邀请（课程渠道） |
| 运行环境 | TA 公告允许按硬件自行选择包与版本，并要求附环境说明。独立运行的 Canvas 要求已由模板确认。若 Kaggle 还要求使用其 Notebook 或指定运行环境，需在 **Rules / Overview** 核实，目前 **UNVERIFIED**。TA 模板（本地留存） · Kaggle 邀请（课程渠道） |

## 冻结候选与移交

以 **Weighted BCE，256×256，Epoch 7** 冻结独立候选。它在既有三份权重中有最高的原图 micro-IoU 和 micro Dice。按逐图平均 IoU 排序，384 版在既有候选中领先。确认官方评分后，用固定的 500 张验证图复核选模。三份历史权重都以严格键和张量形状校验匹配同一个 118,273 参数的 SmallUNet。V1、Weighted BCE 和 384 分别使用各自输入分辨率下第 9、7、9 轮微平均 IoU 最高的 checkpoint。[历史权重审查](architecture_audit.json)

模型输入为双线性整图缩放的 RGB 像素，除以 255。训练标签做最近邻缩放。推理将 logits 双线性恢复为原图尺寸后，以 sigmoid 概率 0.5 二值化。Weighted BCE 的正类权重 2.764543 仅用于训练目标，推理和其他整图候选相同。[模型和推理](submission_candidate/project.py) · [配置](submission_candidate/config.json)

两个 Notebook 已分别在新 Jupyter 内核和迁移后的目录中从头执行。推理目录只有测试 Notebook、权重、模型定义和 3 张图。核对了 CSV 列序、行数、ID、掩码尺寸、二值性及 RLE 的起点和边界。报告 Notebook 展示已保存的训练与验证结果，其可选重训开关保持关闭。14 份包内文件已逐项验 SHA-256，ZIP 可正常读取。候选权重与 Phase 2 源文件逐字节相同，历史基线和三份候选权重校验通过。验证记录（本地留存） · [完整性记录](../../docs/provenance/phase6_integrity.json) · [包内校验清单](../../docs/provenance/frozen_package_SHA256SUMS.txt)

三张公测图的尺寸、样本 ID 和 RLE 段数见本地运行记录。共享副本保留验证结论。

提交验证包 ZIP（本地留存） · [报告 Notebook](submission_candidate/project_report.ipynb) · [推理 Notebook](submission_candidate/test.ipynb) · 冻结权重（本地留存） · [环境说明](submission_candidate/python_environment.txt) · [使用说明](submission_candidate/README.md) · 3 张图的验证 CSV（本地留存）

## V1 架构和训练诊断

V1 的编码分辨率依次为 256、128，通道为 16、32，每层使用两个带 padding 的 3×3 卷积和 ReLU。两次 2×2 最大池化后到达 64×64、64 通道的桥接层。解码将特征双线性放大，和同尺度的编码特征拼接，再用两层卷积融合。1×1 卷积输出 256×256 的单通道 logits。V1 已具有 U-Net 的多尺度编码、解码和跳接结构。[V1 定义](../baseline/project.py) · [U-Net 原论文](https://arxiv.org/abs/1505.04597)

根据卷积、池化和双线性插值逐层追踪输入依赖，V1 中心输出像素最多可依赖 52×52 个输入像素。其理论窗口边长随像素对齐变化，内部为 48 或 52。三尺度版中心为 108×108，内部边长为 108 或 116。这个范围描述结构上能传递的信息，训练后的实际权重决定网络利用了其中多少。[计算和审查](architecture_audit.py) · [计算结果](architecture_audit.json)

V1 最佳权重在训练集和验证集同一 256 网格上的 micro-IoU 为 13.19% 和 12.99%，损失为 1.2018 和 1.2128。Phase 4 的 8 张固定训练图短试验里，IoU 从 14.58% 升至 71.11%。这说明优化器、损失和标注路径能带来学习。全量训练中两者 IoU 都偏低，且训练后段的损失降幅小，因此需要检验加大上下文是否能改善固定训练预算下的表现。单凭训练与验证差距较小，无法确定容量就是瓶颈。[小样本与全量诊断](../phase4/preflight.json) · [V1 训练曲线数据](../baseline/training_history.csv)

## 三尺度整图对照

在 V1 的 64 通道层后加一级下采样、128 通道桥接、匹配的上采样和第三条跳接。参数从 118,273 增至 487,297，增加 369,024。参考 U-Net 的分层结构及 PyTorch 实现中的多尺度融合方式，保留本项目的卷积块、双线性插值和同尺寸输出。使用本地 PyTorch 实现来精确控制共有层以及额外的模块初始化。SMP 的 U-Net 是可替换的标准实现，当前实验无需更换编码器或引入其预训练配置。IMDL-BenCo 启发了同协议比较和同时报告多个定位指标的做法。[新模型](deep_unet.py) · [U-Net](https://arxiv.org/abs/1505.04597) · [PyTorch U-Net](https://github.com/milesial/Pytorch-UNet) · [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) · [IMDL-BenCo](https://github.com/scu-zjz/IMDLBenCo)

使用相同的 4500/500 划分、256×256 整图 RGB、无增强、无权重 BCE + soft Dice、Adam 0.001、批大小 8、种子 5489、训练 10 轮。审查确认共有张量的随机初始化、开始打乱数据前的随机数状态都和 V1 相同。模型按阈值 0.5 时的 256 网格验证 micro-IoU 选择最早最佳轮次，原图评价统一使用相同推理函数。训练在 cs5489 的 RTX 5090 上完成。[实验配置](deep256/config.json) · [训练日志](train_deep.log) · 最佳权重（本地留存）

### 500 张固定验证图的结果

以固定阈值 0.5 计算原图指标。micro-IoU 用全局 TP / (TP + FP + FN)，逐图 IoU 对每张图的 IoU 取平均，Dice 列为全局 Dice。双空时本地协议给 IoU 和 Dice 记 1，本验证集 500 张真实掩码均含前景。三份历史模型复算的逐图 TP、FP、FN 均与此前档案逐项相同。PR-AUC 在公用 256 网格上汇总 32,768,000 个像素的概率，采用梯形积分，与原图阈值指标的空间尺度不同。[原图对照](deep256/native_comparison.csv) · [逐图指标](deep256/validation_by_image.csv) · [复核记录](deep256/validation_summary.json)

| 模型 | 原图 micro-IoU | 原图逐图 IoU | Precision | Recall | Dice | PR-AUC（256 网格） | 小于 1% micro-IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 256 | 13.36% | 14.30% | 37.63% | 17.16% | 23.58% | 24.48% | 2.53% |
| **Weighted BCE 256** | **14.05%** | 14.39% | 27.66% | 22.22% | **24.64%** | 22.85% | 1.98% |
| V1 384 | 13.20% | **14.91%** | 37.82% | 16.87% | 23.33% | 22.87% | **2.61%** |
| 三尺度 U-Net 256 | 11.45% | 12.97% | 38.84% | 13.97% | 20.55% | 24.57% | 2.31% |
| 全前景 | 13.07% | 12.31% | 13.07% | 100.00% | 23.12% | — | 0.53% |

| 模型 | FP 像素 | FN 像素 | 预测前景占比 | 最佳轮次 | 参数量 | 训练加轮内验证 | 原图推理合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 256 | 64,155,192 | 186,802,222 | 5.96% | 9 | 118,273 | 65.36 s | 4.03 s |
| Weighted BCE 256 | 131,047,663 | 175,399,293 | 10.50% | 7 | 118,273 | 48.48 s | 4.04 s |
| V1 384 | 62,537,350 | 187,474,879 | 5.83% | 9 | 118,273 | 122.93 s | 4.47 s |
| 三尺度 U-Net 256 | 49,610,984 | 193,998,906 | 4.70% | 7 | 487,297 | 74.22 s | 4.19 s |
| 全前景 | 1,500,022,455 | 0 | 100.00% | — | — | — | — |

推理计时对同一 500 张已解码 RGB 图执行，从 256 或 384 缩放到返回原图掩码，包含设备传输，排除读取、解码、指标和 CSV 导出。三尺度训练峰值分配为约 652 MiB，计时首轮含冷启动，后九轮平均约 5.33 s，V1 后九轮约 4.93 s。历史训练用时受启动和缓存状况影响。[实验配置](deep256/config.json) · [训练数据](deep256/training_history.csv) · [训练曲线](deep256/training_curves.png)

### 各面积组的原图 micro-IoU

| 真实编辑占比 | 图数 | V1 | Weighted BCE | 384 | 三尺度 U-Net | 全前景 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [0%, 1%) | 73 | 2.53% | 1.98% | 2.61% | 2.31% | 0.53% |
| [1%, 5%) | 171 | 15.30% | 10.44% | 15.31% | 12.47% | 2.72% |
| [5%, 15%) | 139 | 13.27% | 13.37% | 12.96% | 12.23% | 8.38% |
| [15%, 30%) | 52 | 22.45% | 23.95% | 22.40% | 21.15% | 20.92% |
| [30%, 100%] | 65 | 11.10% | 14.18% | 10.80% | 8.59% | 55.37% |

三尺度版各面积组的 micro-IoU 均低于 V1。小区域组逐图平均 IoU 从 6.24% 升至 7.91%，但该组召回从 23.32% 降至 17.01%，像素汇总 IoU 从 2.53% 降至 2.31%。[分组明细](deep256/area_bins.csv)

### 结果判断与后续

- 三尺度版原图 micro-IoU 比 V1 低 **1.91 个百分点**，比 Weighted BCE 低 **2.60 个百分点**，比全前景低 **1.62 个百分点**。相比 V1，误检减少约 1,454 万像素，漏检增加约 720 万像素。验证目标真实前景占比为 13.07%，模型只预测了 4.70%，召回降至 13.97%。PR-AUC 与 V1 接近，说明概率排序能力的变化小于固定阈值下的定位变化。[原图和 PR 指标](deep256/native_comparison.csv)
- 三尺度版训练末轮损失 1.1903，低于 V1 的 1.2000。验证 micro-IoU 在第 7 轮达到 11.16%，第 10 轮为 10.04%。更宽的理论感受野和更低的训练损失在此次固定预算内没有转化为更好的固定阈值定位。结论适用于这一结构、训练预算和单个随机种子。[曲线](deep256/training_curves.png) · [日志](train_deep.log)
- Weighted BCE 仍是原图 micro-IoU 的冻结候选，比全前景高 **0.99 个百分点**，同时逐图 IoU 和 Dice 也较高。差距来自这批固定验证图，Kaggle 官方选模还取决于评分规则。现阶段优先核实规则并用同一验证集重排候选。Phase 5 两个纯裁块模型及本轮加深整图模型均未超过全前景 micro-IoU，目前没有足够证据优先追加局部细化。[Phase 5 报告](../phase5/Phase5_裁块采样实验报告.md)
- 提交前需填 Canvas 小组名、成员名及学号，在最终项目报告补入多阶段对照与失败分析，并确认 Kaggle 和 TA 保留集的评价口径。独立包已在本机运行，TA 实际机器与保留集仍待 TA 评估。[报告 Notebook](submission_candidate/project_report.ipynb) · [推理 Notebook](submission_candidate/test.ipynb)

本阶段共完成一次全量模型训练。复现实验时，在 Phase 6 目录使用 cs5489，运行 architecture_audit.py 后，给 train_deep.py 与 evaluate.py 都传入同一个新目录参数，例如 --out deep256_reproduction。冻结提交包的训练复现、推理路径和环境说明见包内 README。
