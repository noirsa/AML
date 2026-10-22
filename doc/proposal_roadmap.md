# Dry Bean 项目：从数据导入到 Proposal 完成的路线图

> 基于当前项目要求、实际数据和 notebook 的学习与实施计划。
> 整理日期：2026-10-22。
> 本文用于指导逐步完成分析，不是最终 proposal，也不代表下列模型实验已经完成。

## 1. 当前结论与下一步

**现在首先完成「Data inspection and quality audit」：明确每行和每列的含义，核验缺失、重复、不合理值、target 和泄漏风险，写下处理决定。随后固定 held-out split，再开展正式 EDA 与建模。**

建议主路线：

```text
已成功导入 dataset
  → 结构与变量语义核验
  → 缺失、重复、不合理值、target、ID 与泄漏审计
  → 明确 research question 与主要 loss
  → 确定分析样本，定义 X/y，固定 stratified held-out split
  → development set 的 numerical summary 与最小 EDA
  → StandardScaler + L2 multinomial LogisticRegression Pipeline
  → development 内选择 C，并评估整个选择流程
  → 解释 baseline 结果，记录限制
  → 非线性预期、references、可复现材料
  → 使用课程模板整理不超过两页的 proposal
```

**当前推荐方案：**保留原始数据；分析副本对完全相同记录保留一条；分层保留约 20% final test；其余约 80% 作为 training/development set；使用标准化后的 L2 multinomial logistic baseline；以 multiclass log loss 为主要评价指标；在 development 中采用小规模 nested CV。所有建议都需要你理解并能解释。

## 2. 依据与已经完成的内容

### 2.1 项目依据

| 材料 | 已核对的内容 |
|---|---|
| [rubric.pdf](rubric.pdf) | 第 5 节 proposal 六项独立评分要求 |
| [project-guidelines.pdf](project-guidelines.pdf) | 第 3 节 proposal；第 5 节提交材料；第 6 节 preprocessing |
| [course-handout.pdf](course-handout.pdf) | 日期、loss 与最优预测对象、可复现性、AI 使用说明 |
| [proposal_dry_bean.ipynb](../proposal_dry_bean.ipynb) | 当前代码、Markdown 与已保存输出 |
| 本地 `data/dry_bean_raw.csv` | 实际数据结构与基本质量核验；该本地快照不随本文提交 |
| [dry+bean+dataset.zip](../data/dry+bean+dataset.zip) | TXT 数据说明与 ARFF schema |
| [UCI 官方页面](https://archive.ics.uci.edu/dataset/602/dry%2Bbean) | 数据来源、许可证、变量说明 |

本地 handout（2026-10-14 版本）注明 proposal 于 **2026-10-23** 提交。本文依据本地材料；课程后续公告以 Racó 为准。

### 2.2 Notebook 进度

| 内容 | 状态 |
|---|---|
| 基础 imports、`SEED = 2026`、部分 package versions | 已完成 |
| UCI 数据下载与本地原始 CSV 保存 | 已完成 |
| `head()`、原始/输入 shape、target 名称与类别数 | 已有执行输出 |
| 来源变量说明 | 已显示 |
| Research question、数据来源和许可证 | 已有 Markdown 草稿 |
| 正式 inspection、quality audit 与处理决定 | 尚未完成 |
| 固定 held-out split | 尚未完成 |
| 正式 EDA、preprocessing、baseline、CV、evaluation | 尚未完成 |

已有 `X` 和 `y` 是来源提供的原始输入与标签。确定分析样本后，需要重新核对建模版本的 `X/y`，不必为此重新下载数据。

前序环境检查发现项目 `.venv` 尚无 `scikit-learn`，已有 `matplotlib`；`seaborn` 可选。进入建模前安装所需包并记录实际版本即可。

### 2.3 已核验的数据事实

下表来自本地文件审计，不是 baseline 或 EDA 实验结果。你后续应在 notebook 中用自己的检查代码复现并解释。

| 项目 | 结果 |
|---|---|
| 完整表 shape | `(13611, 17)` |
| 原始 n | 13,611 条记录 |
| 输入 d | 16；不包含 target，也不额外计入截距 |
| Target | `Class`，七个无序类别 |
| 输入类型 | 16 个数值输入；未发现类别型输入 |
| 存储 dtype | `Area`、`ConvexArea` 为整数；其余输入为浮点数 |
| 缺失值 | 所有列为 0 |
| 数值中的无穷值 | 0 |
| 完全重复的多余记录 | 68 条，均属 `HOROZ` |
| 涉及重复的全部记录 | 136 条，即 68 对 |
| 相同全部输入、不同标签的冲突组 | 0 |
| 完全去重后的 n | 13,543 |
| 明确 ID 列 | 未发现 |
| 行顺序 | 按类别连续排列，不能直接按原始行顺序切分 |

原始 target 分布：

| Class | n | 比例 |
|---|---:|---:|
| DERMASON | 3,546 | 26.05% |
| SIRA | 2,636 | 19.37% |
| SEKER | 2,027 | 14.89% |
| HOROZ | 1,928 | 14.17% |
| CALI | 1,630 | 11.98% |
| BARBUNYA | 1,322 | 9.71% |
| BOMBAY | 522 | 3.84% |

最大类与最小类约为 6.8:1。完全去重后只有 `HOROZ` 数量变为 1,860，其余不变，所有比例需要重新计算。

基本合理性检查未发现非正输入、`Area > ConvexArea`、`MinorAxisLength > MajorAxisLength`、`AspectRatio < 1`，也未发现 `Eccentricity`、`Extent`、`Solidity`、`Roundness`、`Compactness` 超出 0–1。**这不能推出「没有任何异常值」。**

来源注意事项：

- 原始 ARFF 使用 `AspectRation`、`roundness`；当前 CSV 使用 `AspectRatio`、`Roundness`。代码按当前 CSV 列名写。
- 文档类别拼写有 `Dermosan`，实际标签为 `DERMASON`；实际标签清单应从文件核验。
- 原始说明未完整给出四个 ShapeFactor 的公式；可以记录为连续形状描述量，不自行编造公式。
- 本地数值核验支持 `Solidity = Area / ConvexArea`。文字说明中比值方向含糊时，应结合定义与实际数值，而不是照抄。

## 3. Rubric 对应表

Proposal 按以下六项单独评分，等权。Rubric 中另外十项详细标准主要用于 milestones 和 final report，不应误当成 proposal 必须完成全部高级实验的要求。

| 编号 | Proposal 评分项 | 必须体现的内容 |
|---|---|---|
| P1 | The question | 一个精确问题，以及答案对谁有价值 |
| P2 | The data | 来源、许可证、实际 n/d、每个变量类型、缺失情况和 target |
| P3 | The loss | loss、选择理由、它对应的最优预测对象 |
| P4 | The baseline | 已拟合的 regularized linear/logistic model、penalty 如何选择、实际结果 |
| P5 | The protocol | split/resampling、held-out 数据、不在 held-out 上调参 |
| P6 | Expectation and references | 非线性可能增加什么、1–2 篇相关参考、预期学到什么 |

标记约定：

- **必做**：rubric 明确要求，或保证这些结果可靠所必需的检查。
- **可选**：有助理解，但不是 proposal 的独立硬性要求。
- **后续**：对应之后的 milestones 或 final，不在当前阶段实现。

## 4. 数据使用规则

| 数据范围 | 允许的工作 | 边界 |
|---|---|---|
| Full dataset，划分前 | schema、文档、缺失/重复、预先定义的合理性规则、总体类别数量 | 不通过全数据特征与 target 的关系选模型或特征 |
| Training/development set | 正式 EDA、方法开发、CV、调参、诊断 | 训练误差不能当泛化性能 |
| CV training fold | fit scaler、imputer、encoder、特征选择器、模型 | 不能先在整个 development 上 fit 再交叉验证 |
| CV validation fold | 评价该折模型；内层用于选参数，外层用于评估选择流程 | 不参与对应模型和预处理的 fit |
| Final held-out test set | 全部方案冻结后的最终评估 | proposal 阶段不做 EDA、调参、模型选择或结果解释 |

这里的 training/development set 指最外层划分得到的约 80% 数据。它还会在 CV 内继续划分。CV validation fold 和 final held-out test 不是同一个角色。

原始数据上的结构与质量核验用于建立分析对象。它不是允许先浏览完整数据的所有关系再切分的理由。任何按数据学习的异常阈值、缩放参数、插补参数或编码参数，均遵守训练折内 fit。

## 5. 分步骤实施计划

### 步骤 1：建立 schema 和变量字典【必做】

1. **具体做什么：**确认完整表 shape、n/d、所有列名、存储 dtype、变量语义类型、角色、单位与观测单位。
2. **为什么需要：**防止把整数型测量当类别、把高唯一性的连续量当 ID，满足实际文件核验要求。
3. **对应 rubric：**P2。
4. **Library / function：**`pandas`；`shape`、`columns`、`info()`、`dtypes`、`nunique()`；来源变量说明。
5. **检查或代码类型：**一张包含「名称、dtype、语义类型、角色、含义、单位/是否未知」的变量表；核对 `d = 16` 而非 17。
6. **重点观察：**有无额外索引列、重复列名、漏列、名称差异、文档未定义的含义。
7. **不同结果如何处理：**类型与语义不一致先查文档；多余索引列追查来源；未知定义如实记录，暂不构造依赖该定义的新特征。
8. **数据范围：**full dataset，只做结构和语义检查。
9. **Proposal 写什么：**观测单位、n/d、target、全部输入的类型分组；完整变量字典留在 notebook，正文紧凑呈现。

建议分类：

| 语义类型 | 列 |
|---|---|
| 整数计数型数值测量 | `Area`, `ConvexArea` |
| 连续数值测量/描述量 | `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4` |
| 无序类别 target | `Class` |

**完成标准：**可以解释每行代表一条豆粒测量记录、每列的角色，以及哪些信息文档尚未说明。

### 步骤 2：缺失、重复和不合理值审计【必做】

1. **具体做什么：**检查 NaN、空白或特殊缺失标记、无穷值、完全重复、同输入异标签，以及由文档支持的几何规则。
2. **为什么需要：**确定分析样本，避免错误记录和相同测量跨集合造成信息重叠。
3. **对应 rubric：**P2、P5；支持 P4 的可信度。
4. **Library / function：**`pandas`、`numpy`；`isna()`、`value_counts(dropna=False)`、`np.isfinite()`、`duplicated()`、`groupby()`、`np.isclose()`。
5. **检查或代码类型：**逐列缺失表；全行与仅 X 的重复计数；按全部输入分组检查标签冲突；面积、轴长、比例范围和公式一致性检查。
6. **重点观察：**统计的是重复的多余条数还是所有成员；极端值是否违反定义；重复来源是否可确认。
7. **不同结果如何处理：**有缺失才设计插补；不可能值先追查；合理极端值保留；标签冲突不能直接随机保留一个；重复按预先说明的规则处理。
8. **数据范围：**划分前 full dataset 的确定性质量规则。按分位数/IQR 学习的处理阈值留到 training fold 内。
9. **Proposal 写什么：**实际 missing 结果、68 条完全重复的多余记录、处理决定、原始与分析 n，以及没有发现哪些明确规则违规。

重复处理比较：

| 方案 | 优点 | 局限 |
|---|---|---|
| 完全相同记录只保留一条 | 简单；避免同记录跨集合；可使用普通分层 split/CV | 不能证明这些行对应同一颗豆子；改变重复记录的权重 |
| 保留记录，相同输入绑定为 group | 保留原始频次；相同记录不跨集合 | 最外层 split、内外层 CV 都必须遵守 group 边界，较复杂 |

**推荐：**保留 raw data，在分析副本中完全去重，并说明此规则使分析单位成为唯一的已观测记录。未确认原始来源前，不声称已证明重复是录入错误。如果有证据表明重复是需要保留的真实重复观测，则采用 group-aware 协议。

**完成标准：**每个问题都有计数、判断和处理记录；原始文件保留；分析样本量可复现。

### 步骤 3：Target、class imbalance、ID 与 leakage 审查【必做】

1. **具体做什么：**确认七个合法标签和频次；审查每个输入在预测时是否可获得；检查 ID、类别编码、行索引及答案代理。
2. **为什么需要：**保证监督任务定义正确，并避免输入中间接包含标签。
3. **对应 rubric：**P1、P2、P5。
4. **Library / function：**`unique()`、`value_counts()`、`nunique()`、字符串检查、变量文档人工核对。
5. **检查或代码类型：**标签清单与计数；空格/大小写检查；X 中排除 `Class`；确认索引不入模；检查原始行排序与潜在实体/批次字段。
6. **重点观察：**约 6.8:1 的类别数量差异；高唯一性测量与真实标识符的区别；输入是否由 target 或预测后信息生成。
7. **不同结果如何处理：**无预测意义的 ID 不入模；泄漏变量删除并记录；真实组结构采用 group split；不能仅因不均衡就自动 SMOTE 或加权。
8. **数据范围：**full dataset 的标签、总体计数和语义审计。特征按类别的关系探索放到 development。
9. **Proposal 写什么：**target、类别不均衡概况、分层理由、已识别风险与数据适用范围。

当前没有明确 ID；连续变量接近唯一并不等于 ID。行顺序按类别排列，所以行索引不能作为输入。几何派生量间的确定关系属于冗余，不自动等于 target leakage。

**完成标准：**可以说明预测时已知什么、未知什么，并解释为什么当前 X 中的每列可以用于预测。

### 步骤 4：明确 research question【必做】

1. **具体做什么：**在已有 Markdown 草稿上明确预测对象、输入、应用背景和研究范围。
2. **为什么需要：**proposal 的核心是一个问题，而不是先列一串模型。
3. **对应 rubric：**P1，衔接 P3、P6。
4. **Library / function：**不需要新增 library；Markdown 与来源阅读即可。
5. **检查或代码类型：**核对问题是否可由现有 16 个输入和 `Class` 回答；确认是类别/概率预测，而非原始图像识别或因果研究。
6. **重点观察：**谁需要答案；置信度是否有价值；当前数据是否支持声称跨批次、产地或相机泛化。
7. **不同结果如何处理：**问题超出现有数据就缩小范围；如果关注概率质量，措辞应体现概率预测；不要只用「accuracy」笼统指 log loss。
8. **数据范围：**文档和已完成审计；后续证据来自 development，不根据 test 表现改问题。
9. **Proposal 写什么：**一个精确问题、一段动机和必要的适用范围。

可供自行改写的方向：利用已有的大小与形状描述量，能多好地预测七种豆类的概率，并为后续非线性方法建立怎样的线性参照？

**完成标准：**一句话说清输入、输出、评价目标和意义；不预先承诺某种模型一定最好。

### 步骤 5：选择 loss，并解释最优预测对象【必做】

1. **具体做什么：**预先确定主要 loss、辅助指标，以及它们的角色。
2. **为什么需要：**loss 决定预测目标与模型选择依据；不能只说「分类所以用 accuracy」。
3. **对应 rubric：**P3，支持 P4、P5。
4. **Library / function：**后续使用 `sklearn.metrics.log_loss`、`accuracy_score`、`f1_score`。
5. **检查或代码类型：**核对主指标是否要求概率；明确 `predict_proba` 与 `predict` 的区别；检查是否存在真实错分成本信息。
6. **重点观察：**是否关心置信度；是否已知不同错误成本；按样本平均与每类别等权是否代表不同目标。
7. **不同结果如何处理：**关注概率质量选 log loss；只关注等成本错分可选 0–1 loss；缺少成本依据时不随意构造 cost matrix。
8. **数据范围：**任务定义、文档及 development；不根据 test 结果换指标。
9. **Proposal 写什么：**主要 loss、公式、选择理由、Bayes-optimal 预测对象，以及辅助指标用于什么诊断。

| 方案 | 优点 | 局限 | 本项目建议 |
|---|---|---|---|
| Multiclass log loss | 评价概率与置信度，惩罚自信错误 | 数值不如准确率直观；按样本平均时大类贡献较多 | 主要 loss |
| 0–1 loss / error rate | 直接对应等成本错分 | 不评价置信度 | accuracy 作为辅助 |
| Macro-F1 | 帮助发现类别间表现差异 | 非逐样本可加的概率 loss；不是当前最简单训练目标 | 辅助诊断 |

推荐延续已有 question 中的 multiclass log loss：

\[
\ell(y,\hat{\mathbf p})=-\log\hat p_y,
\qquad
\widehat R=\frac{1}{n}\sum_{i=1}^{n}-\log\hat p_{i,y_i}.
\]

在不限制模型函数形式的理想条件下，最优预测是整个真实条件概率向量：

\[
\mathbf p^*(x)=\big(P(Y=k\mid X=x)\big)_{k=1}^{7}.
\]

这不是声称拟合的 LogisticRegression 一定恢复真实后验。实际模型受到函数形式、数据量与正则化约束。如果要输出类别，再取预测概率的 `argmax`。

**完成标准：**能区分训练时的正则化目标、评价时的预测 loss，以及辅助指标。

### 步骤 6：定义建模 X/y，固定 train / held-out split【必做】

1. **具体做什么：**按已确定的清理规则生成分析副本，定义 16 列 X 与一维 y，固定分层随机划分。
2. **为什么需要：**保留最终评估机会，为整个课程提供一致的数据协议。
3. **对应 rubric：**P2、P5。
4. **Library / function：**`drop(columns='Class')`、`train_test_split`；需要 groups 时使用 group-aware splitter。
5. **检查或代码类型：**X/y 索引对齐、target 未混入 X、记录数、split 索引互斥、完全相同输入不跨集合；保存 seed 与划分索引。
6. **重点观察：**是否先处理完全重复；是否使用 `stratify=y`、随机打乱及固定 `SEED`；是否保留全部类别。
7. **不同结果如何处理：**有实体/批次结构改用 group split；有真实时间任务才考虑时间 split；不能寻找让模型分数更高的 seed。
8. **数据范围：**full analysis dataset 只用于划分；此后 final test 封存。利用标签进行分层属于划分机制，不是调参。
9. **Proposal 写什么：**清理后 n、实际 development/test 数量、比例、seed、分层原因、test 的最终使用时机。

| 方案 | 优点 | 局限 |
|---|---|---|
| 80/20，development 内 CV | 数据利用充分，方便沿用到后续模型 | 需要理解内层/外层验证角色 |
| 60/20/20 | train/validation/test 概念直观 | 单次 validation 不稳定，训练数据更少 |

**推荐 80/20，比例不是 rubric 的硬性规定。** 若按建议去重，分析 n 为 13,543；最终 split 数量以代码输出为准。无需为了 proposal 对 test 做特征分布图或模型评分。

**完成标准：**split 固定且可复现；后续代码默认只使用 development；final test 的使用位置明确留到最后。

### 步骤 7：Numerical summary 与异常值诊断【必做最小版】

1. **具体做什么：**统计 development 输入的均值、标准差、中位数、分位数、范围、常数列和尺度差异。
2. **为什么需要：**理解数值尺度，解释标准化，并区分真实尾部记录与数据错误。
3. **对应 rubric：**支持 P2、P4、P6；不是要求制作完整统计图册。
4. **Library / function：**`describe()`、`quantile()`、`std()`、`nunique()`、`hist()`；`matplotlib`，可选 `seaborn`。
5. **检查或代码类型：**紧凑数值表、少量直方图/箱线图、极端记录检查；必要时按训练类别解释多峰或尾部。
6. **重点观察：**面积与 ShapeFactor 的量级差异、偏态、常数输入、真实类别差异是否造成整体极端值。
7. **不同结果如何处理：**尺度差异用 scaler；合理极端值保留；明显错误先查源数据；不要默认按 IQR、z-score 删除记录。
8. **数据范围：**training/development set。若采用数据驱动的截断或异常处理，阈值在 CV training fold 内 fit。
9. **Proposal 写什么：**尺度差异与标准化的理由，异常值处理原则；完整数值表留在 notebook。

LogisticRegression 不要求每个输入服从正态分布。偏态本身不构成必须取对数、删样本或更换模型的理由。

**完成标准：**能说明为什么需要 scaling，以及哪些极端记录值得调查、哪些保留。

### 步骤 8：Basic EDA【可选，建议做最小版】

1. **具体做什么：**观察少量代表性输入的类别分布、二维重叠及输入间关联。
2. **为什么需要：**为模型限制与非线性预期提供事实依据。
3. **对应 rubric：**支持 P2、P4、P6；rubric 未规定必须交多少张 EDA 图。
4. **Library / function：**`groupby()`、`corr()`、`matplotlib`；可选 `seaborn.boxplot()`、`scatterplot()`、`heatmap()`。
5. **检查或代码类型：**例如 `Area`/`AspectRatio` 的分类别箱线图，两个轴长的散点图，以及输入相关矩阵。每张图配一句观察与局限。
6. **重点观察：**类别重叠、可能的弯曲边界、输入冗余；不要仅凭两个输入的图判定整个任务是否线性可分。
7. **不同结果如何处理：**相关性高先保留 L2 baseline；没有明确收益假设就不增加特征工程；明显异常按质量流程追查。
8. **数据范围：**training/development set；不展示 test 的特征或分类别分布。
9. **Proposal 写什么：**最有信息量的一两条观察；只有直接服务论点的图才考虑放入两页正文。

建议事先固定 baseline 使用全部 16 个输入、标准化和 L2。Development EDA 是探索性分析；若根据其结果选择特征、变换或阈值，相关选择必须纳入训练内协议，不能把已经用过所有 development 标签的选择伪装成外层独立评估。

**完成标准：**每张保留的图都能回答一个具体问题。全面 pairplot、t-SNE、UMAP 暂不需要。

### 步骤 9：最小 preprocessing Pipeline【必做】

1. **具体做什么：**把 StandardScaler 和模型放入 Pipeline；只根据实际需要添加插补或类别编码。
2. **为什么需要：**让每个 CV 训练折独立学习预处理参数，防止 leakage。
3. **对应 rubric：**P4、P5；符合 guidelines 第 6 节。
4. **Library / function：**`Pipeline`、`StandardScaler`；必要时 `SimpleImputer`、`ColumnTransformer`、`OneHotEncoder`。
5. **检查或代码类型：**检查是否在 split/CV 之前调用了完整 X 上的 `fit_transform`；确认预处理随 estimator 一起进入 resampling。
6. **重点观察：**当前无缺失、全部数值输入，所以没有必须 impute 或 one-hot 的依据。
7. **不同结果如何处理：**无缺失省略 imputer；数值缺失可考虑 median imputation；确有无序类别输入才 one-hot；验证中的未知类别用训练时固定的编码规则处理。
8. **数据范围：**每个 training fold fit；对应 validation 只 transform。最终 refit 时在全部 development fit；test 仅最终 transform。
9. **Proposal 写什么：**采用哪些预处理、为何必要、全部学习发生于训练折内，以及省略哪些不需要的处理。

当前推荐结构：

```text
16 numerical inputs → StandardScaler → L2 multinomial LogisticRegression
```

`Class` 可保持字符串标签。不要把它编码为 0–6 后作为连续 target 做回归，也不要把 target one-hot 后混入 X。

**完成标准：**能解释每个 `fit` 具体看到了哪些行；当前 baseline 没有无必要的特征工程。

### 步骤 10：配置 regularized logistic baseline【必做】

1. **具体做什么：**使用 L2-regularized multinomial LogisticRegression，保留全部 16 个输入。
2. **为什么需要：**它直接满足 rubric，适合无序七分类和概率输出，也能作为后续方法的清楚参照。
3. **对应 rubric：**P4，衔接 P3。
4. **Library / function：**`sklearn.linear_model.LogisticRegression`；建议兼容 multinomial 的 `lbfgs` solver。
5. **检查或代码类型：**核对安装版本、L2 参数、solver、`max_iter`、收敛警告、`classes_` 和 `predict_proba`。
6. **重点观察：**是否真正使用多分类 logistic 目标；是否收敛；是否无意启用 class weights 或不兼容的 solver。
7. **不同结果如何处理：**不收敛先检查 scaling、迭代数和 tolerance；当前不因类别数量不同自动启用 `class_weight='balanced'`。
8. **数据范围：**development 内各训练折；不 fit final test。
9. **Proposal 写什么：**模型形式、L2 正则化理由、预处理、solver，以及后续步骤得到的参数选择与结果。

| 模型 | 优点 | 局限与建议 |
|---|---|---|
| L2 multinomial LogisticRegression | 直接输出七类概率，适合保留相关输入，容易解释 | 推荐当前 baseline |
| L1 / Elastic Net logistic | 可产生稀疏表示 | 增加选择工作；当前无明确稀疏化需求 |
| Ridge regression | 简单 regularized regression | 当前 target 是无序类别，不能直接对类别编号回归 |

模型解释可写为：

\[
p_k(x)=\frac{\exp(w_k^\top z+b_k)}{\sum_{j=1}^{7}\exp(w_j^\top z+b_j)},
\]

其中 z 为标准化输入。课程形式的训练目标是平均 cross-entropy 加 L2 penalty：

\[
\frac{1}{n}\sum_i -\log p_{y_i}(x_i)+\lambda\sum_k\|w_k\|_2^2.
\]

Softmax 概率并非输入的线性函数；这里的线性结构指类别之间的 log-odds 对当前输入表示为线性，类别决策边界具有线性形式。

`C` 越小，正则化越强。不要把 `C` 直接当作课程中的 λ，也不要忽略求和/平均与 penalty 常数的区别。实际参数写法以安装版本为准：当前官方文档建议用 `l1_ratio=0` 表示 L2，较早版本常见 `penalty='l2'`。参见 [LogisticRegression 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。

**完成标准：**能说明模型优化什么、L2 控制什么、为什么需要 scaling，以及 solver 是否收敛。

### 步骤 11：CV 与 regularization penalty selection【必做；nested 是推荐实现】

1. **具体做什么：**预先固定 C 网格和主要指标，在 development 内选择 penalty；用外层 resampling 评估这套选择过程。
2. **为什么需要：**区分模型拟合、参数选择与性能估计，避免把最高内层 CV 分数当成独立评估。
3. **对应 rubric：**P4、P5。
4. **Library / function：**`StratifiedKFold`、`GridSearchCV`、`cross_validate` 或显式外层循环、`np.logspace`、`cv_results_`。
5. **检查或代码类型：**建议外层 5 folds、内层 3 folds；小网格如 `np.logspace(-3, 3, 7)`；`scoring='neg_log_loss'`；Pipeline 作为搜索对象。
6. **重点观察：**内层各 C 的分数、边界最优、收敛情况；外层验证是否完全未参与该折的拟合与调参。
7. **不同结果如何处理：**网格调整只根据训练内证据并记录；若据外层结果修改网格或方法，承认这是探索性迭代；不能换 seed 追求好看分数。
8. **数据范围：**全部发生于 development 内部；每层都保持 training/validation 的角色，final test 不参与。
9. **Proposal 写什么：**内外层 fold 数、stratification、seed、C 网格、selection metric、Pipeline 及报告哪一层结果。

```text
Final test：封存

Development：外层 5-fold
  每次外层迭代：
    外层训练部分
      → 内层 3-fold：每个 C 都在内层训练折 fit Pipeline
      → 以内层平均 log loss 选 C
      → 用所选 C 在整个外层训练部分 refit
    外层验证部分
      → 仅预测与评分
  汇总 5 次外层验证结果
```

| 协议 | 优点 | 局限 |
|---|---|---|
| 单层 CV 调参并报告最佳均值 | 最简单 | 是选择结果，有选择乐观性；不能称独立最终评估 |
| development 中另外保留 validation | 直观，能分离选择与评价 | 依赖一次划分，训练数据更少 |
| 小规模 nested CV | 将调参与评价分开，适合 test 持续封存 | 代码和计算略多 |

**推荐小规模 nested CV，但 rubric 未把它规定为 proposal 唯一合法方案。** 关键是解释清楚 protocol，并如实描述结果性质。可参照 [scikit-learn nested CV 说明](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。

`GridSearchCV` 最大化负的 log loss；报告时转回正的 log loss，越小越好。若内层分数近似相同，可预先规定偏好较小 C；不要在看结果后随意改准则。

**完成标准：**每个评估样本未参与对应模型的拟合/内层选择；能解释内外层各自负责什么。

### 步骤 12：Baseline evaluation 与类别诊断【主要结果必做，详细图可选】

1. **具体做什么：**汇总外层 log loss、accuracy、macro-F1；需要时保存外层验证概率用于混淆矩阵和分类别 recall。
2. **为什么需要：**总体分数可能掩盖某一类别失败；评价必须使用适合 loss 的预测形式。
3. **对应 rubric：**P4、P5，支持 P6。
4. **Library / function：**`log_loss`、`accuracy_score`、`f1_score`、`classification_report`、`confusion_matrix`、`predict_proba`。
5. **检查或代码类型：**概率 shape、有限性、非负性、每行概率和、`classes_` 对齐；每个外层验证样本预测一次；计算各 fold 指标。
6. **重点观察：**外层均值与波动、小类别 recall、混淆方向、高置信度错误是否导致大 log loss。
7. **不同结果如何处理：**表现差先排查 bug、收敛和数据问题；真实困难如实报告；小类差时先诊断，不直接重采样或更换主指标。
8. **数据范围：**development 的外层验证预测；final test 不使用。
9. **Proposal 写什么：**主要外层结果、fold 标准差、必要辅助指标和一条最重要的诊断；明确 test 尚未评估。

结果记录模板（运行后填写，不能预填猜测值）：

| Model | Protocol | Log loss | Accuracy | Macro-F1 | Final test |
|---|---|---|---|---|---|
| Scaler + L2 multinomial logistic | Development outer 5 / inner 3 CV | 待运行 | 待运行 | 待运行 | 未使用 |

可报告外层 fold 的均值 ± 标准差，但各 fold 不是统计独立重复，fold SD 不等于 95% confidence interval。如果同时报告汇总 OOF 指标，需标明它与 fold 平均的区别，尤其 macro-F1 不一定一致。

可选的 `DummyClassifier(strategy='prior')` 提供不使用 X 的概率参照。它应只用相应训练折估计类别先验，并沿用同一评价划分；不能替代 rubric 要求的 regularized baseline。

**完成标准：**所有数字可由代码重现；没有把 training accuracy 或最佳内层 CV 分数冒充外层评估。

### 步骤 13：解释 baseline，确定最终 development 模型【必做】

1. **具体做什么：**解释外层结果与局限；再按同一内层规则在全部 development 上选择 C，并 refit 最终 baseline Pipeline。
2. **为什么需要：**外层 CV 给出流程的性能估计；全部 development 上的选择与 refit 给出之后保存/使用的具体 baseline，两者角色不同。
3. **对应 rubric：**P4、P5、P6。
4. **Library / function：**`GridSearchCV(refit=True)`、`best_params_`、`best_estimator_`、结果表；系数查看可选。
5. **检查或代码类型：**记录最终 C、预处理和模型参数、收敛情况、版本、seed；区分外层每折 C 与全部 development 选出的 C。
6. **重点观察：**结果是否稳定、哪些类别困难、线性限制是否与观察吻合；高分是否可能来自错误协议。
7. **不同结果如何处理：**不能平均外层各折 C 作为最终 C；若诊断后修改流程，重新记录该实验及其探索性质；不因 baseline 高分就停止检查泄漏。
8. **数据范围：**development 的评估结果和全 development refit；仍不预测 final test。
9. **Proposal 写什么：**外层性能估计、最终 C、基于证据的解释、局限，以及 final test 封存状态。

普通 log loss 的目标是原分布下的概率质量。`class_weight='balanced'` 会改变训练目标及概率解释；是否采用需有额外理由，当前先保留无 class weights 的 baseline，用 macro-F1 与分类别 recall 补充诊断。

系数探索属于可选：相关变量会共享信息，系数大小不等于因果影响，也不是无条件的变量重要性。不能凭一个好分数声称已经达到 Bayes risk。

**完成标准：**能区分「评估整个选择过程的结果」与「在全部 development 上拟合的最终 baseline」，并解释主要局限。

### 步骤 14：形成非线性方法的预期【必做文字，暂不实现模型】

1. **具体做什么：**根据 development EDA 与 baseline，提出之后可检验的非线性改进假设。
2. **为什么需要：**解释一个问题如何贯穿课程各方法家族，而不是简单罗列算法名称。
3. **对应 rubric：**P6。
4. **Library / function：**当前不需要 nonlinear library；使用已有结果、图和阅读笔记。
5. **检查或代码类型：**把每条假设写成「观察/模型限制 → 方法可能增加什么 → 后续如何用同一协议检验」。
6. **重点观察：**错误是否可能来自复杂边界或输入交互；已有几何描述量是否已提供丰富表示；baseline 是否已很强。
7. **不同结果如何处理：**baseline 强则预期改进可能有限；重叠复杂则提出灵活边界假设；不能先声称非线性一定更好。
8. **数据范围：**development 数据与评估结果；不使用 final test 形成或筛选假设。
9. **Proposal 写什么：**一个简短 expectation 段落、可检验理由、潜在代价和预期学习内容。

| 方法家族 | 之后可能研究的问题 | 实施阶段 |
|---|---|---|
| Kernel methods | 更灵活的边界能否改善概率/类别预测？计算代价如何？ | Milestone 1 |
| Neural methods | 学习输入组合是否优于固定描述量上的线性结构？ | Milestone 2 |
| Hybrid / ensemble methods | 组合表示或模型是否带来互补收益？ | Milestone 3 |
| 综合比较 | 相同协议下，性能、复杂度、解释性和成本如何取舍？ | Final integrative report |

数据 n 略高于 guidelines 建议的约 1,000–10,000，但不是自动不合格；后续 kernel 方法需考虑内存和训练时间。若必须下采样，应提前说明仅在训练部分实施并设计公平比较，不能根据 test 结果决定样本。

当前主 loss 是 log loss，所以后续方法也必须产生适当概率。尤其 SVM 的 decision score 不是概率；如需校准，校准步骤也必须留在训练内协议。这是后续设计事项，proposal 无需现在实现。

**完成标准：**清楚区分已观察结果、待检验假设与未来工作；没有运行额外 nonlinear models 来凑 proposal。

### 步骤 15：阅读并选择 1–2 篇 references【必做】

1. **具体做什么：**选择与数据、研究问题或方法依据直接相关的文献，核验书目信息并实际阅读相关部分。
2. **为什么需要：**支持数据来源和研究背景，避免把未核对的文献结果当成自己的证据。
3. **对应 rubric：**P6，数据归属支持 P2。
4. **Library / function：**不要求 Python library；浏览器、PDF 阅读器、BibTeX 均可。
5. **检查或代码类型：**核对作者、年份、题名、DOI；记录所用样本、特征、split/CV、评价指标及与本项目的差异。
6. **重点观察：**文献支持哪句话；其结果能否与当前实验直接比较；是否研究同一份数据。
7. **不同结果如何处理：**协议不同只作为背景，不直接宣称优劣；无法核验或未读到的内容不写成已确认事实；一篇高度相关文献也满足 1–2 篇数量要求。
8. **数据范围：**外部文献与来源资料；不需要 final test。
9. **Proposal 写什么：**1–2 篇有实际用途的引用；数据 DOI 和许可证另行注明；引用应支撑正文论点。

优先阅读：

> Koklu, M., & Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques*. Computers and Electronics in Agriculture, 174, 105507. [DOI](https://doi.org/10.1016/j.compag.2020.105507).

它是当前数据对应的原始研究，优先看任务背景、特征提取和评价协议。无需为了达到「两篇」而增加不相关文献。

数据来源：

> Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C50S4B). License: CC BY 4.0.

**完成标准：**每个引用都已打开核对，并能解释它支持了自己的哪一项选择或背景陈述。

### 步骤 16：整理 proposal 与可复现提交包【必做】

1. **具体做什么：**把已完成分析压缩进课程模板，形成最多两页 proposal，并准备代码、数据来源和复现说明。
2. **为什么需要：**正式 PDF 必须独立可读；notebook 是代码与证据，不能替代所要求的文档。
3. **对应 rubric：**P1–P6；符合 guidelines 的提交规范。
4. **Library / function：**课程 LaTeX template、Jupyter、README、`importlib.metadata.version`；具体排版按模板完成。
5. **检查或代码类型：**重启 kernel 后顺序运行；核对每个正文数字；检查页数、作者、project card、results ledger、引用、版本、seed、split 和 AI 使用说明。
6. **重点观察：**有无真实 baseline 结果；penalty 如何选择是否明确；是否写出 loss 的最优预测对象；是否误把计划写成已完成。
7. **不同结果如何处理：**数字不一致就重跑修复，不能手工改数；模板缺失从 Racó 获取；材料缺项按清单补齐。
8. **数据范围：**汇总审计及 development 实验；final test 继续封存，等最终方案锁定后使用。
9. **Proposal 写什么：**下方内容清单。正文由你结合实际运行结果组织，本文不提供代填的最终结果或讨论。

**完成标准：**两页内覆盖六项 rubric，任何结果都有可运行代码来源，按要求打包提交。

## 6. Proposal 应包含什么

| 内容 | 最小必要信息 |
|---|---|
| 标题、成员 | 初步标题及两位成员姓名；若有例外，需符合课程已批准安排 |
| Question | 一个精确问题、实际意义、适用范围 |
| Data | 来源、许可证、观测单位、原始与分析 n、d、每个变量的类型、target、missing 与 duplicate 决策 |
| Loss | 名称、理由、最优预测对象、辅助指标 |
| Baseline | Pipeline、L2 multinomial logistic、C 如何选择、实际评估结果 |
| Protocol | held-out、stratification、CV、预处理 fit 边界、test 封存 |
| Expectation | 非线性可能增加什么、依据、预期学习内容 |
| References | 1–2 篇相关文献；数据来源和工具按要求注明 |
| 模板要求 | project card 和 results ledger；按课程模板填写并持续保留实验记录 |
| AI 使用说明 | 按 handout 在相关代码和文本中说明实际使用位置与内容 |

两页空间有限，不需要把 notebook 的所有检查表和图都放进去。正文保留支持六项要求的关键数字、理由和结果；完整审计、EDA、搜索记录留在代码材料中。

课程要求一个压缩包，包含：

- PDF 文档及 `.tex` 源文件，使用课程指定模板。
- 复现结果所需的 Python/R 代码文件，不能只给仓库链接。
- 数据，或在数据较大/许可要求下提供来源链接。
- 纯文本 README：准确执行步骤、随机种子、依赖版本，以及如何生成文中的每个数字。

当前材料缺口：文件夹未发现 `aml-project-template.zip`；notebook 目前只有一位作者。它们是提交前清单事项，不妨碍现在完成数据审计。

## 7. 哪些现在做，哪些以后做

| Proposal 必做 | 有帮助但非必须 | 后续 milestones / final 按需开展 |
|---|---|---|
| Schema、变量语义与实际 n/d | 更多 EDA 图 | 各 milestone 对应的 nonlinear models |
| Missing、duplicate、明显不合理值审计 | Dummy 概率参照 | 系统模型比较与计算成本研究 |
| Target、ID、leakage 检查 | 完整分类别报告 | 大规模超参数搜索 |
| 问题与主要 loss | 系数探索 | 有明确理由的 feature engineering / selection |
| 固定 held-out protocol | 更详细的不确定性分析 | 有研究需求的 PCA、重采样、class weighting |
| 最小 preprocessing Pipeline | 少量敏感性分析 | 有概率需求的方法校准 |
| Regularized baseline、训练内 penalty selection | 更丰富的错误分析 | 最终方案锁定后的 test 评估 |
| 合理协议下的真实 baseline 结果 | — | 综合结论、局限与整个课程的比较 |
| Expectation、references、复现材料 | — | 有额外数据时的跨域/跨批次评估 |

上表的后续可选技术不是 final 必须全部实现的清单。Kernel、neural、hybrid/ensemble 应按各自 milestone 开展，不应全部留到最终报告才开始。

## 8. 执行顺序与阶段出口

| 阶段 | 对应步骤 | 离开本阶段前应得到什么 |
|---|---|---|
| A. 数据审计 | 1–3 | Schema 表、quality 表、target 表、重复处理与泄漏判断的 Markdown |
| B. 问题与实验边界 | 4–6 | 明确问题与 loss；分析 X/y；固定 split；封存 test |
| C. 理解训练数据 | 7–8 | 数值汇总、少量 EDA、最小 preprocessing 理由 |
| D. Baseline 与验证 | 9–13 | Pipeline、CV 搜索记录、外层评估、最终 development C/refit、结果解释 |
| E. 完成 proposal | 14–16 | 非线性预期、已读 references、两页文档、完整复现包 |

### 8.1 现在先完成阶段 A

- [ ] 解释 n 与 d，并核对 `(13611, 17)` 与 16 个输入。
- [ ] 建立全部列的 dtype / 语义类型 / 角色表。
- [ ] 区分数值测量、类别、ID；不根据 dtype 单独判断含义。
- [ ] 复现缺失值、非有限值检查。
- [ ] 复现 68 条完全重复的多余记录和 136 个重复成员。
- [ ] 复现相同输入是否存在不同标签的检查。
- [ ] 运行有文档依据的基本合理性规则。
- [ ] 记录原始类别数量与不均衡程度。
- [ ] 审查 target、行索引及潜在泄漏。
- [ ] 写下重复处理方案及其局限，保留 raw data。

完成阶段 A 后，推进到问题/loss 和固定 split。此时再开始正式 training EDA，不需要先把所有分析做完才划分。

### 8.2 Proposal 提交前自检

- [ ] 问题、target、主 loss 一致。
- [ ] 原始与分析 n 区分清楚；d 不包含 target。
- [ ] 每个变量类型有依据；缺失和重复在实际文件上核验。
- [ ] 固定 held-out，所有后续模型沿用同一份。
- [ ] 所有 learned preprocessing 在训练折内 fit。
- [ ] Baseline 符合 regularized linear/logistic 要求。
- [ ] 写清 C 网格、选择指标和 resampling。
- [ ] 评估结果与调参分数明确区分。
- [ ] 最终 C 来自全部 development 上的同规则选择，没有平均外层 C。
- [ ] 主结果来自真实运行；没有使用 final test。
- [ ] Nonlinear expectation 有依据且未被写成已验证结论。
- [ ] References 已核验并阅读。
- [ ] 使用课程模板，不超过两页；成员与模板字段齐全。
- [ ] 重启 kernel 后按顺序可重现所有数字。
- [ ] 代码、数据/链接、README、版本与 AI 使用说明完整。

## 9. 可复制到 notebook 的 Markdown 记录模板

以下是记录结构，不是已经完成的分析结论。每个重要检查后，根据自己的实际输出填写。

```markdown
## Data inspection and quality audit

### Purpose
本阶段要确认什么？对应哪项 rubric？

### Data used
本次检查使用 full dataset 还是 training/development set？
为什么这个范围适合当前检查？

### Observation
代码实际显示了什么？记录数量、比例或其他必要证据。

### Interpretation
结合 dataset documentation，这些结果意味着什么？
哪些含义尚不确定？

### Decision
采取什么处理？为什么？有什么局限？
如果不处理，也说明原因。

### Implication for the modelling protocol
这是否影响样本量、split、preprocessing 或 evaluation？
是否有任何参数需要在 training folds 内学习？

### Proposal note
这一步最后应在 proposal 中保留哪一句事实或哪项方法说明？
```

Baseline 阶段另加一个简短记录模板：

```markdown
## Baseline experiment

- Question and primary loss:
- Data version and split identifier:
- Preprocessing Pipeline:
- Model and regularization:
- Candidate C values and inner CV:
- Outer evaluation protocol:
- Outer results and variability:
- Final C selected on all development data:
- Main observation and limitation:
- Final held-out test status: not evaluated
```

本路线图由 AI 辅助整理，包含实施建议而非课程指定的唯一方案。正式提交时按 handout 如实记录实际协助，并确保每项选择、每段代码与每个结论都能由你解释和核验。
