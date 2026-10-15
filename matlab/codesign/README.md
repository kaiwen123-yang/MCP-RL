# MATLAB数学核验

`runCodesignMathAudit.m`核验闭环响应共同设计的三个基础问题：可实现条件、纯重参数化无控制能力增益、段内反馈与输入占用的真实取舍。它不训练RL，不加载机器人模型，不运行外部仓库代码。

本机已在MATLAB R2025b运行，使用Symbolic Math Toolbox和Optimization Toolbox。Code Analyzer最终无告警。函数不清空工作区或关闭图窗；结果写入调用者指定目录。

在本目录作为当前MATLAB目录时：

```matlab
projectRoot = fileparts(fileparts(pwd));
outputDir = fullfile(projectRoot, "work", "codesign-decision-2026-10-15", "matlab");
report = runCodesignMathAudit(outputDir);
```

输出JSON与MAT文件。公开研究结果已保存于[结果JSON](../../notes/codesign-decision-2026-10-15/math-audit-results.json)，推导和解释见[理论记录](../../notes/codesign-decision-2026-10-15/theory-and-matlab.md)。

负例确保坐标变化不被误报为新增能力；正例明确充分整定经典反馈已达到例中最优。因此这些结果不能用作新RL算法优于现有方法的实验证据。
