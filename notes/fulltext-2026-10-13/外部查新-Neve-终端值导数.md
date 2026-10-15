# 外部查新：Neve等终端值导数工作

日期：2026-10-13。**不纳入主库100篇阅读数量，不分配或回收MR编号。**

Thomas Neve, Tom Lefebvre, Guillaume Crevecoeur, *Fast Nonlinear Model Predictive Control Combining Online Trajectory Optimization and Value Function Regression*, Optimal Control Applications and Methods 46(2), 709–725 (2025)。DOI：[10.1002/oca.3225](https://onlinelibrary.wiley.com/doi/full/10.1002/oca.3225)。出版社标正式在线2024-10-28；更早公开首稿本次未核。

## 实际覆盖

- 已读出版社HTML中摘要、§1–5、作者贡献、尾注、附录A/B/C、数据声明和全部52条参考文献的可提取文字与公式文本；正文段落由多次连续定位读取覆盖，约对应浏览文本L135–680。
- 后续从出版社正式[PDF入口](https://onlinelibrary.wiley.com/doi/pdf/10.1002/oca.3225)读取到17页版的解析文本，并补读第6页Algorithm 1完整步骤；没有将未逐页重读的17页PDF另计一轮精读。
- 正常浏览器验证自行完成后，在出版社原图查看器逐幅查看图1–9、图C1和Algorithm 1原图；附录A末尾的矩阵图片和表B1也已查看。没有手工完成验证码或绕过权限。
- 正式PDF文本可读，但本次本地下载返回403或超时，因此**未取得本地PDF文件，不计PDF下载覆盖**。图式核查采用官网HTML原图，未逐符号认证全部公式，也未从图中重建全部原始数值。
- 无独立证明审核、代码阅读、复现或新实验。

所以最终状态为**出版社HTML全文文字已读；所有编号图/算法及唯一参数表已视觉核查；部分PDF文本补核；本地PDF未取得；逐字符认证与独立证明未完成**。这项补充查新仍不进入原100篇统计。

## 对候选A的必要约束

§3.1.2已用DMS值/导数标签训练终端近似；§3.2式(22)–(24)分析导数/曲率扰动对局部优化解和首动作的影响；§4.4有导数监督消融。数值误差在§3.2单列，实验用IPOPT（附录B）。因此后续应核查**实际K次未收敛更新与切换边界**是否仍有重要失败，不能只提梯度监督或局部敏感性。具体研究判断见[主线独立审视](主线独立审视.md)。
