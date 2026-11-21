# 三人小组协作

仓库公开可读，由所有者向课程小组成员授予写入权限。成员可分别负责推理和打包、实验结果复核、报告及 Canvas 核对。需要共同修改同一份 Notebook 时，先在 Issue 里确认负责人和顺序。

## 第一次同步

使用以下命令取得代码。组员推送分支时使用有写入权限的 GitHub 账号完成认证。

~~~bash
git clone https://github.com/PomeloRuii/CS5489-Project-Group5.git
cd CS5489-Project-Group5
git switch main
~~~

接着按 [README 的环境说明](../README.md#建立环境) 配置 Python。先执行指标复算，再打开报告 Notebook。需要完整推理时，再单独放置经许可获得的权重和图片。 [GitHub 克隆文档](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

## 日常改动

提交或保存手头分支的工作后，在 main 上拉取最新版本并开新分支。示例为调整报告的分支。

~~~bash
git switch main
git pull --ff-only
git switch -c feature/report-analysis
~~~

完成修改后，清除 Notebook 的输出，检查差异并逐项加入计划提交的文件。

~~~bash
git status --short
git diff
git add docs/final_delivery.md
git diff --cached
git commit -m "Clarify validation results in delivery checklist"
git push -u origin feature/report-analysis
~~~

在 GitHub 开 Pull Request，目标选 main。写明改动、验证命令、输入依赖及剩余问题。另一名组员核对 Notebook 能否在干净内核运行、路径是否可移植、指标口径是否一致，确认这次改动仅含拟共享文件后合并。涉及训练结果时，保留原运行记录和配置。 [GitHub Pull Request 文档](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

可先建立四个 Issues，分别跟踪 Kaggle 官方规则、TA Notebook 独立运行、六轮实验与失败分析、Canvas 小组与截止前检查。在对应 PR 里引用 Issue 编号。 [GitHub Issues 文档](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue)
