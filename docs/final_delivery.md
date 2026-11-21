# 课程交付与剩余工作

## A. VERIFIED｜原始课程要求

| 项目 | 要求与原始出处 |
| --- | --- |
| 任务 | 输入 RGB 图像，输出标示生成或编辑像素的 0/1 掩码。教授的 Project 说明见 Lecture7_1.pdf 第 2–3 页。 |
| 数据与选模 | 课程有 5,000 张训练图、500 张 Kaggle 公测图和 500 张保留评估图。留出部分训练数据验证和选模。包内 README.txt 说明训练标注和公测图在 ZIP 内，保留集和测试答案在包外。见 Lecture7_1.pdf 第 4–5 页。 |
| 报告 | 向 Canvas 提交含可复现训练代码、执行输出、观察与分析的 Notebook。评分包含实验对照、失败分析和清晰简洁的报告。见 Lecture7_1.pdf 第 5–6 页。 |
| 推理 | 向 Canvas 提交可运行的 test.ipynb 和 model.pth。TA 只需修改路径，推理使用 RGB 图和权重，无需真值。模型定义和依赖可随附，预处理与训练一致。见 Lecture7_1.pdf 第 5 页及 TestTemplate_2026_Inference_Verification_ColumnMajor.ipynb 第 1–4、9 单元。 |
| 输出 | TA 模板输出原图尺寸二值掩码。RLE 按列扫描，从 1 编号，用起点和长度表示连续前景，空掩码写空字符串。每张图对应一行 ID,Predicted，ID 来自文件名并保留前导零。见 TA 原始推理模板第 3、5–9 单元。 |
| 小组与环境 | 每组至多五人，所有成员加入同一 Canvas Course Project Group，项目报告用 Canvas 组名。包与版本可按硬件选用，在 Canvas 附简短 Python 环境说明 TXT。见 Lecture7_1.pdf 第 5 页及本会话提供的 TA 项目公告第 1–2 节。 |
| 截止时间 | **2026-12-13 23:59:59**。见 Lecture7_1.pdf 第 5 页。 |
| 汇报 | 项目展示可选。TA 汇报公告第 1 点要求小组在 11 月 30 日前告知 TA 是否参加展示。见 Lecture7_1.pdf 第 6 页及本会话的 TA 汇报公告。 |

## B. PROJECT DECISION｜本组剩余工作

- 完成 [报告 Notebook](../experiments/phase6/submission_candidate/project_report.ipynb) 的正式提交副本，补入六阶段对照、失败案例和局限，保留执行输出及复现入口。
- 使用固定的 4,500/500 划分和本地 micro-IoU 保留 Weighted BCE 作为候选。填入 Canvas 小组名与成员，复跑 [推理 Notebook](../experiments/phase6/submission_candidate/test.ipynb)，随正式提交副本提供权重、模型定义及 [环境说明](../experiments/phase6/submission_candidate/python_environment.txt)。
