# CS5489 Image Manipulation Localization

从 RGB 图像定位 AI 编辑像素，输出原图尺寸的 0/1 掩码。TA 推理 Notebook 将掩码编码为 RLE 并导出两列 CSV（ID,Predicted）。我们从课程训练数据中固定划出 4,500 张训练图和 500 张验证图。

## 实验进度

冻结候选为 **Weighted BCE，256×256，Epoch 7**。以下为固定验证集上的原图 micro-IoU，全前景参照为 13.07%。

| 阶段 | 模型或对照 | micro-IoU |
| --- | --- | ---: |
| 1 | [V1](experiments/baseline/README.md) | 13.36% |
| 2 | [Weighted BCE](experiments/phase2/Phase2_诊断与实验报告.md) | **14.05%** |
| 3 | [V1 384×384](experiments/phase3/Phase3_分辨率实验报告.md) | 13.20% |
| 4 | [RGB + SRM](experiments/phase4/Phase4_SRM实验报告.md) | 11.35% |
| 5 | [Uniform Crop / Mask-aware Crop](experiments/phase5/Phase5_裁块采样实验报告.md) | 9.79% / 10.51% |
| 6 | [三尺度 U-Net](experiments/phase6/Phase6_提交冻结与整图对照报告.md) | 11.45% |

## 目录

- [experiments](experiments/) 保存 Phase 1–6 的代码、固定划分、配置、实验结果与报告。
- [最终候选](experiments/phase6/submission_candidate/) 包含报告和推理 Notebook、模型定义及复训入口。
- [environment](environment/) 保存实测依赖和 TA 环境建议。[docs](docs/) 收录复现、协作和交付说明。

## 建立环境

实测环境为 Windows、Python 3.11.16、PyTorch 2.9.1+cu128 和 RTX 5090。

~~~bash
conda create -n cs5489-project python=3.11 pip
conda activate cs5489-project
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements.txt
python -m ipykernel install --user --name cs5489-project --display-name "Python (cs5489-project)"
~~~

CPU 推理可安装 PyTorch 的 CPU 版本。 [PyTorch 安装指引](https://pytorch.org/get-started/previous-versions/#v291)

## 运行与复现

- **查看结果**。从候选目录打开 [project_report.ipynb](experiments/phase6/submission_candidate/project_report.ipynb)，选择项目内核。默认读取存档结果，重训入口可按需开启。
- **运行推理**。单独取得课程图片和冻结的 model.pth，在同一目录运行 [test.ipynb](experiments/phase6/submission_candidate/test.ipynb)。设定图片与权重路径，保留随附的 project.py。 [权重 SHA-256](docs/provenance/local_artifact_SHA256SUMS.txt)
- **重训或复算**。从仓库根目录运行以下命令可用存档计数复算原图指标。使用课程 ZIP 重训时参照 [各阶段运行入口](docs/reproduction.md)。

~~~bash
python tools/recompute_metrics.py
~~~

## 协作与交付

组员从 main 建分支并通过 PR 合并。Canvas 截止前需补齐带运行输出的正式报告，核对小组信息，并配齐推理 Notebook、权重、依赖和环境说明。 [协作命令](docs/collaboration.md) · [课程要求与剩余工作](docs/final_delivery.md)
