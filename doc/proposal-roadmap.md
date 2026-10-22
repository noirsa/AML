# Dry Bean Proposal 路线

核查日期：2026-10-22。当前进度：已导入基础 libraries、加载数据，并写有 research question / loss 的初稿；尚未正式完成 data inspection、EDA、preprocessing、baseline 或 evaluation。

本路线依据本地 rubric、project guidelines、course handout、实际 Dry Bean 数据及 `proposal_dry_bean.ipynb`。本次仅核查和规划，未训练模型。

## 1. Proposal 明确要求

Proposal 按 rubric §5 的六项等权评分，占课程总成绩 5%。后续 milestones / final report 的十项通用 rubric 不应直接当作 proposal 的十项要求。

| Rubric | 明确要求 | 需要什么 |
| --- | --- | --- |
| **P1 问题** | 一个明确研究问题，以及答案对谁有价值 | 文字 |
| **P2 数据** | 来源、许可、n、d、每个变量类型、missing、target；必须核对实际文件 | **代码检查＋实际结果** |
| **P3 损失** | 指定 loss、解释理由、说明它对应的最优预测器 | 文字＋简短公式 |
| **P4 Baseline** | 拟合正则化 linear / logistic 模型，说明 penalty 如何选择，报告结果 | **代码＋真实结果** |
| **P5 Protocol** | resampling、held-out 的安排；held-out 不参与调参 | **实际执行＋文字说明** |
| **P6 预期与引用** | 非线性可能增加什么、1–2 篇 references、预计学到什么 | 文字；**无需实现非线性模型** |

**必须有 baseline 结果，但不必在 proposal 阶段打开最终 test。** Project guidelines 要求说明什么数据保留到最后。因此本路线采用：proposal 报告训练内部 CV 结果，最终 held-out evaluation 留到项目末尾。

提交硬要求（project guidelines §§3、5）：

- **最多 2 页 PDF**，使用课程模板，包含标题及两名成员姓名。
- 模板中的 project card / results ledger 贯穿项目；ledger 持续追加，不删除旧结果。
- 通过 Racó 提交一个压缩包：PDF、`.tex` 源文件、可复现代码、数据或合法链接、README。
- README 写明复现步骤；随机种子统一设置，记录 package versions；披露 AI 用于哪些部分。
- 核查时当前目录未见课程模板，Notebook 只列了一名作者，提交前需要补齐。
- 本地 course handout 标注 proposal 截止 **2026-10-23**，批准日期 **2026-10-28**；未给具体截止时刻。

## 2. 实际数据核查结果

实际读取了 `data/dry+bean+dataset.zip` 内的 `.xlsx` 和 documentation，并与 Notebook 已保存的输出核对。

| 检查 | 实测结果 | 对路线的影响 |
| --- | --- | --- |
| **Shape / target** | 原始 **n=13,611，d=16**；含 `Class` 共17列；七分类 | d 不包含 target；使用分类 baseline，不用 Ridge 回归 |
| **变量类型** | `Area`、`ConvexArea` 是整数面积测量；其余14个输入是连续形态量；`Class` 是无序类别 | 不仅看 pandas dtype；输入没有 encoded categorical，无需 one-hot |
| **缺失 / ID** | 无缺失、无无穷值、无显式 ID、无常量列 | 当前无需插补；不要把行号加入 X |
| **重复与泄漏** | **68 条多余完全重复记录**，涉及68对、136行，全部为 HOROZ；无相同 X 对应冲突标签 | 划分前处理；去重后 **n=13,543**；若保留，重复组不能跨 split 或 CV folds |
| **类别不平衡** | 最大类 DERMASON 3546，最小类 BOMBAY 522，最大/最小约 **6.8:1** | 分层抽样；辅助检查 macro-F1 / 每类 recall，不自动加权或过采样 |
| **数据顺序** | 原表按类别连续排列，共7个类别区段 | 不能按前80% / 后20%切分；必须打乱后分层 |
| **基本范围** | 已查正值、长轴不小于短轴、凸包面积不小于面积等约束，未发现这些规则下的明显违规 | 不代表不存在其他异常；箱线图极端值不能自动删除 |
| **尺度与冗余** | 输入尺度差异大；多列由面积、轴长等计算而来 | 标准化＋L2 合适；派生关系不等于 target leakage，不必 PCA 或大量删列 |
| **来源差异** | 本地列名为 `AspectRation` / `roundness`，Notebook 的 UCI 输出为 `AspectRatio` / `Roundness`；文档 `Dermosan` 对应实际标签 `DERMASON` | 固定数据版本与命名，target 标签以实际数据为准 |
| **文档局限** | Extent / Solidity 的文字比值方向含糊，四个 ShapeFactor 没有完整公式；无采集批次 ID | 必要时查原论文，不自行猜测含义；不能据此证明对新采集条件的泛化 |

原始 target 分布（去重只使 HOROZ 变为1,860，其余计数不变）：

| Class | n | 原始占比 |
| --- | ---: | ---: |
| DERMASON | 3,546 | 26.05% |
| SIRA | 2,636 | 19.37% |
| SEKER | 2,027 | 14.89% |
| HOROZ | 1,928 | 14.17% |
| CALI | 1,630 | 11.98% |
| BARBUNYA | 1,322 | 9.71% |
| BOMBAY | 522 | 3.84% |

逐列语义类型（proposal 应保留这种完整映射）：

| 类型 | 变量 |
| --- | --- |
| 整数测量输入 | `Area`、`ConvexArea` |
| 连续测量输入 | `Perimeter`、`MajorAxisLength`、`MinorAxisLength`、`AspectRation`（UCI：`AspectRatio`）、`Eccentricity`、`EquivDiameter`、`Extent`、`Solidity`、`roundness`（UCI：`Roundness`）、`Compactness`、`ShapeFactor1`、`ShapeFactor2`、`ShapeFactor3`、`ShapeFactor4` |
| 无序类别 target | `Class` |

复现注意：Notebook 留存的保存路径为旧的 `D:\AML`，核查时当前目录没有 `data/dry_bean_raw.csv`。下一步先确认当前实际加载的数据来源与路径。数据为 UCI Dry Bean，许可 **CC BY 4.0**，需保留数据来源署名。

## 3. 从现在到 proposal 的执行顺序

“必须做”包含完成 rubric 所需的基本方法检查，不代表每个检查都是单独评分项。**Full 仅用于结构、完整性、固定领域规则和类别计数检查；影响模型选择的 EDA 只用 train。**

| 阶段 | 做什么／为什么 | Python library / function | 看什么 → 如何处理 | 数据范围 | Proposal 写什么／Rubric |
| --- | --- | --- | --- | --- | --- |
| **① 必须做：数据检查** | 核对 n、d、语义类型、target、缺失、ID、异常、重复和类别数 | pandas：`shape`、`info`、`isna`、`nunique`、`duplicated`、`value_counts`；NumPy：`isfinite` | 文档与实际不符→记录差异；重复→去重并说明假设，或保留为同组 | **Full** | 数据概况、逐列类型、清理前后 n、处理理由；**P2/P5** |
| **② 必须做：问题与 loss** | 明确预测七类概率，服务自动分选；确定评判标准 | 无需代码；评价用 `log_loss` | 概率质量是目标→保留 log loss；不能将其称为 accuracy | 文档＋任务定义 | 问题、意义、loss 与最优预测器；**P1/P3** |
| **③ 必须做：X/y 与划分** | X=16个输入，y=`Class`；建议80% train、20%最终 test，固定 seed | `train_test_split(stratify=y, random_state=SEED)` | 检查各类覆盖、重复组无交叉；保留重复时改用分组划分与 `StratifiedGroupKFold` | 清理后的 **Full → train/test** | 比例、seed、重复处理、test 封存规则；**P5** |
| **④ 必须做：basic EDA** | 看分布、类别重叠、异常候选、变量相关性，理解难点 | `describe`、`corr`；seaborn：`histplot`、`boxplot` | 极端值→核实测量含义，不直接删；高相关→记录，不自动删列 | **Train** | 2–3条关键发现；图仅在有解释价值时放入；**P2/P6** |
| **⑤ 必须做：最小预处理** | 建立标准化＋L2 多分类 logistic baseline | `Pipeline`、`StandardScaler`、`LogisticRegression` | 当前无缺失/类别输入→不插补、不编码；若后来确认需要，才加入 `SimpleImputer` / `OneHotEncoder` | **每个 CV training fold 内 fit** | 保留16输入及预处理理由；**P4/P5** |
| **⑥ 必须做：CV＋penalty selection** | 训练内5折分层 CV，搜索小型对数尺度 C 网格 | `StratifiedKFold`、`GridSearchCV(scoring="neg_log_loss")` | 选择平均 log loss 最低的 C；若最优落网格边缘→仅在 train 扩展；不收敛→检查尺度、增加迭代 | **Train 内部** | C 网格、最优 C、CV log loss 均值与折间标准差；**P4/P5** |
| **⑦ 必须做：解释 baseline** | 看概率损失、少数类表现和主要混淆，说明局限 | CV 预测；`f1_score(average="macro")`、`confusion_matrix` | 少数类差→报告问题，不自动加权或过采样；系数不作因果解释 | **Train/CV** | baseline 表现、主要困难、评估局限；**P4/P6** |
| **⑧ 可选：更独立的评估** | 外层 CV 评价、内层 CV 选 C，减少选择偏差 | `cross_validate(GridSearchCV(...))` | 报外层均值/波动；若不做，明确⑥是调参 CV 分数 | **Train 内部** | 更严格的 baseline 估计；**P4/P5** |
| **⑨ 必须做：预期＋引用** | 提出非线性边界/交互可能改善重叠类别的假设 | 无需代码 | baseline 很强→可能提升有限；混淆明显→作为后续待验证方向 | **Train/CV＋文献** | 非线性价值、1–2篇引用、预计学到什么；**P6** |
| **⑩ 必须做：完成 proposal** | 按六项压缩到2页；补模板、作者、ledger、README、AI声明 | Notebook 全量重跑；课程 LaTeX 模板 | 数字能否复现、路径是否有效、六项是否齐全；失败先修复 | 已有结果；**不打开 test** | 完整提交包；**P1–P6** |
| **⑪ 留到 final project：held-out evaluation** | 全部方法与参数冻结后，用全 train 重拟合并评估 | `predict_proba`、`log_loss`，预先约定的辅助指标 | 无论好坏都报告；不能据此回调模型或预处理 | **最终 test** | Proposal 只写计划；最终报告给测试结果；**P5** |

## 4. 必须守住的方法边界

- **Loss：**multiclass log loss 的 population-risk 最优预测器是完整概率向量 `p*_k(x) = P(Y=k | X=x)`；`argmax` 给出类别，对应0–1分类错误的最优决策。训练目标含 L2 惩罚，评价的 log loss 不含惩罚项。
- **正则化：**C 越小，正则越强。`neg_log_loss` 越大越好；展示时转换回正的 log loss，越小越好。
- **CV 结果：**用来挑选 C 的最佳 CV 分数存在选择偏差，不能称为独立 test 成绩；折间标准差不是置信区间。Nested CV 是可选加强项，并非 proposal 明文强制。
- **预处理：**scaling、imputation、encoding 等需要学习参数的步骤只在各 training fold 内 fit，优先放入 Pipeline。最终 test 不参与 EDA、预处理决策、选模或调参。
- **重复：**去重需说明“相同特征与标签视为重复记录”的假设；完全相同不能证明是同一物理豆粒。若保留，split 和 CV 均须保证重复组不交叉。
- **类别不平衡：**先保留自然分布、分层抽样，用 macro-F1 / 每类 recall 补充检查；不因不平衡自动使用 `class_weight="balanced"`，因为这会改变优化目标及概率含义。
- **解释：**共线变量下的系数不代表因果影响。原始输入已经含数学派生量，logistic 的“线性”是相对这16个特征而言。
- **后续范围：**kernel、neural、hybrid / ensemble 分别留到后续 milestones；跨方法整合留到 final report。Proposal 不实现 nonlinear models，不做复杂 feature engineering 或 PCA，除非实际证据要求。
- **计算量：**13,611行略超课程建议的约1,000–10,000行，但不是硬性禁用；后续 kernel 方法要考虑 n² 内存，必要时仅对 training 设计一致的子采样方案。

## 5. References 与本地依据

建议的1–2篇学术参考：

1. Köklü, M., & Özkan, İ. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques*. Computers and Electronics in Agriculture, 174, 105507. [DOI](https://doi.org/10.1016/j.compag.2020.105507)。用于数据背景及已有研究；论文成绩不能直接作为同协议比较结果。
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed. [作者网站](https://hastie.su.domains/ElemStatLearn/main.html)。用于分类、正则化与模型评估背景。

数据和工具来源另外署名：

- [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry%2Bbean)，[Dataset DOI](https://doi.org/10.24432/C50S4B)，CC BY 4.0。
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)、[Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)。
- 本地 [rubric](rubric.pdf) §5；[project guidelines](project-guidelines.pdf) §§3–6；[course handout](course-handout.pdf)。
- [当前 Notebook](../proposal_dry_bean.ipynb)；[原始数据包](../data/dry+bean+dataset.zip)。
- AI 使用记录：2026-10-22，OpenAI Codex 协助核查本地要求、数据与 Notebook，并整理本路线。后续提交时按实际用途更新代码注释和文档中的披露；作者需核查并能解释方法与结果。

## 下一步：现在只做什么

在 Notebook 加一个 **data inspection 单元**：确认当前数据来源，并输出 **shape、逐列类型/含义、缺失数、重复数、类别计数及基本范围**。

先核对是否复现 **13,611×17、0缺失、68条多余重复、7类**；核对完再决定重复处理与划分。此时先不生成后续全部代码。
