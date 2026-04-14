---
# 核心元数据
author: lanshi
date: "2026-04-14T12:00:00+08:00"
lastmod: "2026-04-14T22:00:00+08:00"
title: "用 LLM 给聚类调参，但不让它决定真相：一次汽车购买数据的无监督实验复盘"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: "本文基于汽车购买属性数据，比较 KMeans、DBSCAN 与 Agglomerative Clustering 三种聚类方法，并引入 LLM 作为参数空间生成器，在保证评分与最终裁决完全由确定性代码执行的前提下，讨论无监督学习调参、聚类评价失真和工程化流程设计。"

# 内容分类
series:
  - "机器学习实验"
tags:
  - "LLM"
  - "聚类分析"
  - "KMeans"
  - "DBSCAN"
  - "Agglomerative Clustering"
  - "无监督学习"
  - "Silhouette Score"
  - "LangGraph"
  - "LangChain"
  - "自动调参"
categories:
  - "机器学习"
  - "无监督学习"
  - "实验复盘"

# SEO优化
description: "本文复盘一次汽车购买数据的无监督聚类实验，比较 KMeans、DBSCAN 与分层聚类在 One-Hot 编码特征上的表现，并分析如何让 LLM 只参与参数搜索建议，而不参与真值判断与最终裁决。"
keywords:
  - "LLM聚类调参"
  - "KMeans"
  - "DBSCAN"
  - "Agglomerative Clustering"
  - "无监督学习"
  - "Silhouette Score"
  - "LangGraph"
  - "LangChain"
  - "聚类评价"
  - "机器学习实验"

# 主题集成
math: false
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "database-cover.png"
  alt: "LLM辅助聚类调参实验封面"
  caption: "LLM 负责提议，代码负责求证"
  relative: true

# 版权声明
copyright: true
---

无监督学习最容易踩的坑，不是模型跑不起来，而是你拿到一个“看起来很漂亮”的指标，却得到一堆完全不可解释、不可落地的簇。

这次我用一个汽车属性数据集做了一个完整实验：同时比较 `KMeans`、`DBSCAN` 和 `Agglomerative Clustering`，再让 LLM 只负责一件事：为每一轮搜索生成更有希望的参数空间。真正的聚类、评分、比较和最终裁决，全部交给确定性代码完成。

这篇文章想回答三个问题：

1. LLM 适不适合参与无监督聚类调参。
2. 为什么 `Silhouette Score = 1.0` 也不一定代表结果最好。
3. 怎样把“LLM + 传统机器学习”做成一个工程上靠谱的结构。

---

## 一、任务背景

实验数据是经典汽车购买评价数据，共 `1728` 条样本。

参与聚类训练的字段有 5 个：

- `buying`：购买费用
- `maint`：维修费用
- `doors`：车门数量
- `person`：乘坐人数
- `lug_boot`：行李箱容量

` safety ` 字段没有参与训练，只用于聚类完成后的后验解释。这一点非常关键，因为如果把 `safety` 直接放进训练特征，聚类结果会被目标语义“提前泄漏”，实验就不再纯粹。

---

## 二、预处理策略：训练空间和评分空间必须一致

这个实验里，我坚持了一个很重要的原则：

> 聚类训练在哪个空间完成，评价就必须在哪个空间完成。

具体做法是：

- 对 5 个离散特征做 `OneHotEncoder`
- 再做 `StandardScaler`
- 得到最终训练空间 `X=(1728, 17)`
- 额外用 `PCA` 压到 2 维，仅用于可视化，不参与调参和评分

这样做可以避免一个常见错误：模型在高维编码空间里训练，却在 PCA 压缩空间里算轮廓系数。那样得到的分数，优化目标和真实训练目标并不一致。

---

## 三、系统设计：让 LLM 当“提议者”，不要当“裁判”

整个设计有 5 个核心原则：

1. 预处理与聚类彻底分离。
2. 训练空间与评分空间保持一致。
3. LLM 只负责生成搜索空间，不负责计算真值。
4. 聚类搜索函数保持纯函数化。
5. LangGraph 只负责编排流程，不承载核心算法逻辑。

也就是说，LLM 可以说：

- “下一轮 KMeans 试试 `3~15` 个簇”
- “下一轮 DBSCAN 缩小 `eps` 搜索范围”
- “层次聚类优先试 `complete` linkage`

但它不能说：

- “这个聚类更好”
- “这个轮廓系数是真的”
- “这个结果应该直接上线”

真值判断必须留给代码。

---

## 四、自动调参框架怎么搭

整个自动调参控制器其实非常简单，本质上是一个高阶函数：

```python
def autotune(algorithm, generator, searcher, target_score, max_rounds):
    history = []
    best_result = {"best_score": float("-inf")}

    for round_index in range(1, max_rounds + 1):
        search_space = generator(tuple(history))
        round_result = searcher(**search_space)
        history.append({"round": round_index, "space": search_space, **round_result})

        if round_result["best_score"] > best_result["best_score"]:
            best_result = round_result

        if best_result["best_score"] >= target_score:
            break

    return best_result
```

这里最重要的不是代码量，而是职责切分：

- `generator(history)` 负责提议下一轮搜索空间
- `searcher(**search_space)` 负责真正执行搜索
- `history` 负责记录每一轮的结果
- `target_score` 决定是否提前停止

这个结构的好处是，LLM 可以替换，也可以随时拿掉。没有 LLM 时，系统会回退到默认参数空间，整个流程仍然可运行。

---

## 五、三种聚类算法的评分策略

三种算法的评分并不完全相同。

### 1. KMeans / Agglomerative

这两类算法直接在训练空间 `X` 上计算轮廓系数。

### 2. DBSCAN

DBSCAN 会产生 `-1` 噪声标签，因此评分时只对非噪声点计算：

- 过滤掉 `label == -1` 的样本
- 如果剩余样本不足以形成至少 2 个簇，则判为无效结果
- 否则再计算轮廓系数

这比“直接把噪声当作普通簇”更合理。

---

## 六、为什么我又额外设计了一个 `practical_score`

只用原始 `Silhouette Score` 做比较，在这个实验里会得出一个明显错误的结论。

因为最终结果是这样的：

| 算法 | 原始轮廓系数 | 实用性评分 | 最优参数 | 簇数 | 噪声比例 | 最大簇占比 | 轮数 |
|---|---:|---:|---|---:|---:|---:|---:|
| Agglomerative | 0.2075 | 0.1175 | `{'linkage': 'complete', 'n_clusters': 15}` | 15 | 0.00% | 12.50% | 6 |
| KMeans | 0.1978 | 0.1078 | `{'n_clusters': 15}` | 15 | 0.00% | 8.33% | 6 |
| DBSCAN | 1.0000 | 0.1000 | `{'eps': 0.1, 'min_samples': 2}` | 432 | 0.00% | 0.23% | 1 |

如果只看原始轮廓系数，`DBSCAN` 是绝对第一，而且一轮就满足目标分数 `0.53`。

但问题在于，它把 `1728` 条样本切成了 `432` 个簇。最大簇只占 `0.23%`，也就是单个簇只有极少样本。这样的结果在数学上可能“分得很开”，但在业务上几乎没有解释价值。

所以我额外定义了一个实用性评分：

```text
practical_score
= raw_silhouette
- cluster_penalty
- noise_penalty
- imbalance_penalty
- singleton_penalty
```

它主要惩罚四类问题：

- 簇数量过多
- 噪声比例过高
- 最大簇过于失衡
- 单点簇比例过高

在这次实验里：

- `Agglomerative` 的簇数是 15，结构可接受，只被扣了 `0.09`
- `KMeans` 同样是 15 个簇，也被扣了 `0.09`
- `DBSCAN` 因为簇数高达 432，被直接扣了 `0.9`

于是原始得分最高的 `DBSCAN`，在实用性排序里反而落到最后。

---

## 七、最终为什么选了 Agglomerative

LLM 在最终裁决阶段没有盲从最高分，而是基于结构质量选择了 `Agglomerative`。它的理由可以概括为：

- 簇数量适中
- 没有噪声点
- 最大簇占比不高，分布比较平衡
- 相比 `KMeans`，分层聚类在当前数据上的原始得分略高
- 相比 `DBSCAN`，结果没有出现“过度碎片化”

最终最优配置是：

```python
{'linkage': 'complete', 'n_clusters': 15}
```

这也说明一个很现实的问题：

> 在聚类任务里，“最优”往往不是某个单一指标的最大值，而是多个约束下的平衡结果。

---

## 八、调参历史也很有意思

### KMeans

KMeans 在 6 轮里反复收敛到 `n_clusters = 15`，最佳分数 `0.1978`。这说明它对当前 one-hot 编码后的类别结构有一定适配性，但上限不高。

### Agglomerative

Agglomerative 第一轮还能得到 `ward + 4 簇` 的 `0.1656`，但从第二轮开始就稳定找到：

```python
{'linkage': 'complete', 'n_clusters': 15}
```

最佳分数 `0.2075`，说明层次结构在这组离散特征上比 KMeans 稍微更合适。

### DBSCAN

DBSCAN 第一轮就找到了：

```python
{'eps': 0.1, 'min_samples': 2}
```

并且轮廓系数直接达到 `1.0`。这不是“模型神了”，而是参数把数据切得过于零碎，导致内部指标被异常放大。

这类结果如果没有后处理惩罚，很容易误导后续决策。

---

## 九、从 `safety` 的簇内分布看，聚类并没有自然恢复安全等级

因为 `safety` 没参与训练，所以我们可以用它来做后验解释。

最优算法 `Agglomerative` 的 15 个簇中，`unacc` 都占主导，大致在 `59% ~ 85%` 之间；`vgood` 占比普遍很低，最高也只有 `7.41%`。

这说明：

1. 当前用于训练的 5 个属性，确实能形成某种结构。
2. 但这种结构和 `safety` 标签并不是强一一对应关系。
3. 聚类更像是在按“成本、空间、承载能力”分组，而不是直接恢复“安全等级”。

这也是无监督学习特别值得警惕的一点：

> 找到结构，不等于找到你真正关心的语义。

---

## 十、工程实现里几个值得复用的点

### 1. LLM 不可用时自动回退

如果没有配置 API Key，系统会自动使用默认参数空间，而不是整体崩掉。

### 2. JSON 安全解析

LLM 输出统一走 `safe_json_loads`，会自动剥离 Markdown 代码块和无关文本，再提取 JSON 对象，减少解析失败。

### 3. 流式回退

`llm_invoke` 里对某些必须开启 streaming 的模型做了兼容，如果普通 `invoke` 失败，会自动切到 stream 聚合模式。

### 4. Tool 层很薄

LangChain Tool 只是纯函数的轻量包装，核心逻辑不依赖 Agent 框架。这意味着后续你完全可以把 LangGraph 拿掉，直接在普通 Python 程序里复用这些搜索器。

### 5. LangGraph 只做流程编排

整个工作流非常清晰：

```text
planner -> execute_selected -> aggregate -> reply
```

这让系统行为既可解释，也容易调试。

---

## 十一、这次实验给我的几个结论

### 1. LLM 适合做“搜索导航”，不适合做“指标真值”
让 LLM 生成下一轮参数空间是有价值的，因为它能利用历史结果缩小搜索范围。但最终评分必须交给确定性算法。

### 2. 无监督任务里，单一内部指标很危险
`DBSCAN = 1.0` 就是一个典型案例。指标越漂亮，越要警惕它是否在钻评价函数的空子。

### 3. 聚类结果必须加入结构约束
仅比较原始轮廓系数远远不够。簇数、噪声比例、簇平衡性、单点簇比例，都应该纳入评价。

### 4. 后验解释比“直接把标签放进训练”更有价值
把 `safety` 留到最后分析，才能真正判断聚类学到的结构是否接近业务语义。

### 5. LangGraph 最适合承担 orchestration，而不是算法实现
把聚类逻辑写成纯函数，再由图结构负责编排，是更稳妥的工程方案。

---

## 十二、下一步可以怎么改进

如果继续优化这个实验，我会优先做三件事：

1. 把 `practical_score` 设计成可配置规则，而不是写死在代码里。
2. 为不同算法增加更细粒度的搜索约束，例如限制 DBSCAN 的最大簇数或最小平均簇规模。
3. 在最终比较时加入更多内部指标，例如 `Calinski-Harabasz`、`Davies-Bouldin`，避免单指标偏置。

---

## 结语

这次实验最有意思的地方，不是哪个算法赢了，而是它清楚展示了一个现实：

> 当 LLM 进入传统机器学习流程时，最好的位置不是“替代算法”，而是“辅助搜索与编排”。

它可以帮助我们更快探索参数空间，更灵活地组织流程，但不能替代严谨的评价体系。

在这个实验里，`DBSCAN` 给出了最耀眼的数字，`Agglomerative` 给出了最稳妥的结果。而真正让系统变得可靠的，不是 LLM，而是那条始终没变的边界：

**LLM 负责提议，代码负责求证。**
