# 已存候选池中的选择器遗漏与正确共同gate重选

日期：2026-10-17。[复算脚本](../../scripts/audit_stage_selector.py)、[全部768查询证据](selector-audit.json)。本轮只读48个旧NPZ及已存成本/margin，**没有生成命令候选、调用模型或环境、训练或修改权重**。

## 1. 独立复核与明确分层

- support的384查询全部有共同gate可行候选，包括全部3个fixed_support checkpoint的192查询；没有空pool。
- penalty的384查询中140个空pool（joint_penalty68/192、fixed_penalty72/192）。其余244个已有可行候选，但reference-only主gate的actual-cost赢家有163个tube超界；旧reference-cost赢家为156个。
- 本脚本从NPZ独立重建掩码与argmin，768查询的原primary索引、secondary是否可行以及可行时两个secondary索引全部与旧JSON一致。

**163个问题首先是选择器未要求执行已知门槛，不是固定反馈能力不足的证据。140个空pool只证明这些有限保存候选未满足条件，不证明连续命令规划无解，更不证明物理不可达。**

## 2. gate完全沿用，未放宽门槛

共同筛选是`reference_margin≤0`、`actual_domain_margin≤1e−7`、`input_margin≤0`、`tube_margin≤1e−8`，另要求两类成本有限。两个容差来自此前原native violation标志，不是本次调整。

support空集、fallback及一步/最坏耗散excess仍作诊断，没有偷偷加入或删除新的条件。因此这里只称“既有名义共同gate”，不是全扰动鲁棒证书，也不是已经定义好的长程机器人阶段入口合同。空pool的corrected selection为null，没有把index0或最小违反候选伪装成成功fallback。

在同一gate内分别最小化actual成本和reference成本，所有候选及预测保持相同。正确actual选择对声明gate的违反数为0；这由筛选定义保证，不是学习或新控制算法效果。

| seed | checkpoint arm | 查询 | 空pool | 有可行候选却选tube违反 | 正确actual选择违反 |
|---:|---|---:|---:|---:|---:|
| 17 | joint_support | 64 | 0 | 0 | 0 |
| 17 | fixed_support | 64 | 0 | 0 | 0 |
| 17 | joint_penalty | 64 | 18 | 30 | 0 |
| 17 | fixed_penalty | 64 | 12 | 37 | 0 |
| 29 | joint_support | 64 | 0 | 0 | 0 |
| 29 | fixed_support | 64 | 0 | 0 | 0 |
| 29 | joint_penalty | 64 | 34 | 23 | 0 |
| 29 | fixed_penalty | 64 | 38 | 24 | 0 |
| 43 | joint_support | 64 | 0 | 0 | 0 |
| 43 | fixed_support | 64 | 0 | 0 | 0 |
| 43 | joint_penalty | 64 | 16 | 23 | 0 |
| 43 | fixed_penalty | 64 | 22 | 26 | 0 |

## 3. 遵守既有条件的任务成本代价

仅在**相同244个已有可行候选的penalty查询**上比较，不把140个失败删除后冒充全体表现：

- reference-only gate的actual赢家平均actual cost：`0.0186991500463`。
- 正确共同gate的actual赢家平均actual cost：`0.02118758448`。
- 同一共同gate的reference赢家、按actual模型评价的平均cost：`0.021209722347`。
- 正确actual相对原primary的配对平均成本增加：`0.00248843443366`；均值相对变化`13.3077%`。

这是满足已声明条件的有限池代价取舍；不把约束纠正称为无代价收益。corrected actual成本不高于corrected reference的actual成本，是同池argmin定义的结果。没有执行新计划，因此不能报告新的闭环成功率、恢复能力或完成时间。

正确actual选择后，仍有`0`个可行penalty查询出现support-empty诊断、`204`个出现一步excess诊断；这些不属于当前四项gate，必须保留而非改称完整鲁棒可行。

## 4. 所有失败的结构，不选代表checkpoint

按全部固定tick/组别分解：

| tick | family | 查询 | 空pool | 有可行候选却选tube违反 |
|---:|---|---:|---:|---:|
| 0 | support | 96 | 0 | 0 |
| 0 | penalty | 96 | 0 | 74 |
| 60 | support | 96 | 0 | 0 |
| 60 | penalty | 96 | 52 | 26 |
| 120 | support | 96 | 0 | 0 |
| 120 | penalty | 96 | 42 | 38 |
| 180 | support | 96 | 0 | 0 |
| 180 | penalty | 96 | 46 | 25 |

140个空pool的首个失效gate统计为`{"tube": 140}`。JSON为每个query保留finite→reference→actual-domain→input→tube的候选数，和已通过前四层候选中的最小tube margin；后者只定位有限池缺口，不作为有效计划返回。

另只读取原保存metric的V与tube标志，对规划起点分层：140个空pool中**105个进入规划时已经tube外，35个仍在tube内**。tick0没有空pool。另有**2个初始tube外却有可行候选**的查询，因此初始违反不能直接推出不可恢复。tick>0使用已存poststep`V[tick−1]`及其原tube标志；tick0因为x=z而误差V=0，并非物理x=0。没有新状态预测。

这揭示了合同语义需要分开：当前筛选要求每个未来prefix立即满足旧tube；长程任务中的恢复阶段是否允许一段过渡时间，应在完整stage合同中预先定义。不能因初始超界就自动放宽门槛，也不能把全部105个例子直接归为固定反馈的物理极限。

所有失败ID都公开列出，不按结果挑某个checkpoint。接下来若做连续命令或更充分规划，应在预声明相同合同下使用这些失败全集或明确抽样规则；只能由新增规划/可达性证据进一步区分预算不足与固定反馈限制。

## 5. 资源和本轮结论

读取48个旧pool文件、12个旧metric文件、768查询、26112条已保存候选记录；新增candidate、模型转移、环境转移、actor调用、训练、checkpoint读写均为0。pool NPZ hash与原记录核对，旧metric来源hash另列，未改任何原数组。

本轮可排除“所有违反都来自固定反馈无能”的解释：已知可行候选存在时，正确选择能直接修复163个penalty tube误选。剩余140个空pool仍是有限搜索失败。此时尚无证据把它们命名为必须由RL创造的新能力，也没有把旧servo门槛包装为狗/臂长程任务验证。
