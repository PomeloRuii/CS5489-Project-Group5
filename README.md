# CS5489 Image Manipulation Localization

课程任务是从一张 RGB 图像定位 AI 编辑像素。模型输出与输入原图同尺寸的二值掩码，0 表示背景，1 表示编辑区域。数据中的掩码用 RLE 表示。实验使用固定的 4,500 张训练图和 500 张验证图。数据和权重通过经课程允许的渠道在组内单独取得。

当前冻结候选是 Phase 2 的 **Weighted BCE，256×256，Epoch 7**，原图验证 micro-IoU **14.05%**。V1 的 384 版在逐图平均 IoU 上更高。这些分数来自本地验证。Kaggle 正式评分为 **UNVERIFIED**。 [六阶段对照及局限](docs/final_delivery.md) · [原图验证指标](experiments/phase6/deep256/native_comparison.csv)

## 文件结构

- [experiments/baseline](experiments/baseline/) 保存 Phase 1 的模型代码、训练 Notebook、固定划分和验证记录。
- [experiments/phase2](experiments/phase2/) 到 [experiments/phase6](experiments/phase6/) 保存各轮源码、配置、训练日志、验证指标及实验报告。保留相邻目录结构，以兼容已有的相对导入和输出路径。
- [experiments/phase6/submission_candidate](experiments/phase6/submission_candidate/) 是当前候选的可移植 Notebook 与所需源码。冻结权重由课程允许的渠道单独提供。
- [environment](environment/) 保存本项目实测依赖，以及供比对的 TA 环境建议。
- [docs](docs/) 保存文件清单、复现入口、组内协作流程、提交检查和校验记录。

## 建立环境

实测为 Windows x64、Anaconda、Python 3.11.16 和 PyTorch 2.9.1+cu128，硬件为 RTX 5090。推理 Notebook 可使用 CPU。本仓库的训练时长和 CUDA 指标均来自以上 GPU 环境。

~~~bash
conda create -n cs5489-project python=3.11 pip
conda activate cs5489-project
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements.txt
python -m ipykernel install --user --name cs5489-project --display-name "Python (cs5489-project)"
~~~

若只用 CPU，安装 PyTorch 时可将源换成 PyTorch 官方的 CPU wheel 索引 https://download.pytorch.org/whl/cpu。已有 cs5489 环境可以继续使用。TA 环境建议可供比对。 [实测依赖](environment/requirements.txt) · [PyTorch 官方版本安装指引](https://pytorch.org/get-started/previous-versions/#v291)

## 打开候选报告与运行推理

1. 用 VS Code 或 Jupyter 打开候选目录中的 [project_report.ipynb](experiments/phase6/submission_candidate/project_report.ipynb)，选择项目 Python 内核，并将工作目录设为该 Notebook 所在目录。默认关闭重训，可直接读取存档结果。
2. 从课程允许的渠道取得冻结的 Weighted BCE 权重，放到候选目录下并命名为 model.pth。可对照 [本地产物 SHA-256](docs/provenance/local_artifact_SHA256SUMS.txt) 中原包的路径和摘要查验。
3. 在 [test.ipynb](experiments/phase6/submission_candidate/test.ipynb) 的配置单元填写测试图片路径。默认使用同目录的 test_images/ 和 model.pth，输出 submission.csv。保留随附的 project.py。Notebook 只读图片及权重，保留文件名 ID 的前导零，输出原图尺寸 RLE。
4. 导出的 ID,Predicted 是本地 TA 模板格式。正式 Kaggle 提交前按竞赛页面核实字段、评分方式和空掩码处理。

## 复核和复现

在仓库根目录运行以下命令，可从存档逐图计数复核各模型的原图指标。无需图像、权重或 GPU。

~~~bash
python tools/recompute_metrics.py
~~~

如需从图像复现训练或验证，先把课程 ZIP 放在本地可读目录，再按 [复现指引](docs/reproduction.md) 使用独立输出。带预览的原始报告、权重和数据均留在课程工作目录。在此共享目录中，Notebook 执行输出已清除。当前候选报告可从表格重建数值结果。

## 组员协作

仓库公开可读。组员 clone 后按上面的环境说明配置内核。main 保持可复现，各自从 main 建 feature 分支。用 Pull Request 提交改动，由另一名组员复核代码、Notebook 和指标。用 Issues 记录报告改动、运行问题、评分核实和提交任务。 [命令及评审流程](docs/collaboration.md) · [原始文件处置](docs/file_inventory.csv) · [首批文件清单](docs/git_file_list.txt)
