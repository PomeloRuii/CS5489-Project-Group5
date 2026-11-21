# 组员协作

首次获取仓库。

~~~bash
git clone https://github.com/PomeloRuii/CS5489-Project-Group5.git
cd CS5489-Project-Group5
~~~

从最新的 main 建自己的分支。

~~~bash
git switch main
git pull --ff-only
git switch -c feature/report-analysis
~~~

检查差异后，仅暂存相关文件。Notebook 提交前清除执行输出。

~~~bash
git diff
git add docs/final_delivery.md
git commit -m "Update project delivery checklist"
git push -u origin feature/report-analysis
~~~

在 GitHub 创建面向 main 的 Pull Request，说明改动和检查结果，由另一名组员审阅后合并。
