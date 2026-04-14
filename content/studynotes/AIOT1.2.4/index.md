---
# 核心元数据
author: lanshi
date: "2026-04-13T12:00:00+08:00"
lastmod: "2025-04-13T22:00:00+08:00"
title: "一次汽车购买价值聚类实验：KMeans、DBSCAN 与分层聚类的结果对比"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: "本文基于一份汽车购买相关离散特征数据，使用 KMeans、DBSCAN 与分层聚类三种无监督学习方法进行对比实验，分析不同聚类算法在类别编码数据上的表现差异，并讨论轮廓系数、簇结构稳定性与特征表示方式对结果的影响。"

# 内容分类
series:
  - "机器学习实验"
tags:
  - "聚类分析"
  - "KMeans"
  - "DBSCAN"
  - "分层聚类"
  - "无监督学习"
  - "Silhouette Score"
  - "One-Hot Encoding"
  - "数据预处理"
  - "Scikit-learn"
categories:
  - "机器学习"
  - "聚类实验"

# SEO优化
description: "本文对汽车购买相关离散特征数据进行聚类实验，比较 KMeans、DBSCAN 与分层聚类的结果差异，重点讨论类别变量数值映射、轮廓系数解释、簇结构稳定性以及无监督学习中的数据表示问题。"
keywords:
  - "KMeans"
  - "DBSCAN"
  - "分层聚类"
  - "聚类分析"
  - "无监督学习"
  - "轮廓系数"
  - "One-Hot Encoding"
  - "Scikit-learn"
  - "类别特征聚类"
  - "机器学习实验"

# 主题集成
math: false
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "database-cover.png"
  alt: "汽车购买价值聚类实验封面"
  caption: "KMeans、DBSCAN 与分层聚类结果对比"
  relative: true

# 版权声明
copyright: true
---

## 摘要

这篇文章基于 `cluster_analysis_01.ipynb` 的实验过程，使用 `KMeans`、`DBSCAN` 和分层聚类三种方法，对一份汽车购买相关的离散特征数据进行无监督分析。整个实验的目标不是做分类预测，而是观察样本是否会自然形成若干“购买模式”，并比较不同聚类算法在这类数据上的表现。

从结果来看，`DBSCAN` 的轮廓系数最高，但它同时产生了大量微小簇和噪声点；`KMeans` 与分层聚类的整体结构更稳定，但结果也暴露出一个典型问题：把类别变量简单映射成数字后直接使用欧式距离，会让某些字段在聚类中被过度放大。

---

## 1. 数据概览

Notebook 中加载的数据共有 `1728` 条记录、`6` 个字段，且没有缺失值。

```python
raw_data = pd.read_csv("./data/inputs/car_data.csv")
raw_data.info()
print("数据维度：", raw_data.shape)
print("null vlues：", raw_data.isnull().sum())
```

从实验输出来看：

- 数据规模：`(1728, 6)`
- 所有字段均为 `object`
- 不存在缺失值

Notebook 顶部对字段做了如下业务解释：

- `buying`：购买费用
- `maint`：维修费用
- `doors`：车门数量
- `person`：乘坐人数
- `lug_boot`：行李箱容量
- `safety`：安全性

不过从实际取值分布看，字段命名和业务语义之间存在一定偏差风险，因此这次实验更适合被理解为“对离散类别特征做编码后聚类”，而不是对强业务含义字段做精细解释。

---

## 2. 数据预处理：把类别映射成数值

由于三种聚类算法都不能直接处理字符串，Notebook 先把类别变量映射成整数：

```python
data["buying"] = data["buying"].map({"vhigh":1, "high":2, "med":3, "low":4})
data["maint"] = data["maint"].map({"2":2, "3":3, "4":4, "5more":5})
data["doors"] = data["doors"].map({"2":2, "4":4, "more":5})
data["person"] = data["person"].map({"small":2, "med":5, "big":7})
data["lug_boot"] = data["lug_boot"].map({"low":1, "med":2, "high":3})
data["safety"] = data["safety"].map({"unacc":1, "acc":2, "good":3, "vgood":4})
```

这一步虽然简单直接，但也埋下了一个关键问题：

**数值映射会人为引入“距离关系”。**

例如：

- `small -> 2`
- `med -> 5`
- `big -> 7`

这意味着模型会认为 `small` 和 `big` 的距离远大于 `med` 和 `big`，而这未必符合真实业务语义。对于基于距离的聚类算法来说，这种编码方式会直接影响最终簇结构。

---

## 3. KMeans：最直观，但不一定最合理

Notebook 先用 `KMeans` 在 `k=2~7` 之间做了尝试，并使用轮廓系数评估聚类质量：

```python
scores = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k)
    y_ = kmeans.fit_predict(X)
    score = silhouette_score(X, y_)
    scores.append(score)
```

随后，Notebook 手动选择了 `k=6`：

```python
kmeans = KMeans(n_clusters=6)
y_ = kmeans.fit_predict(X)
print('得分是：', silhouette_score(X, y_))
```

保存下来的结果是：

- `KMeans(n_clusters=6)`
- 轮廓系数：`0.2327`

但如果把 `k=2~7` 系统性比较并固定随机种子后重新检查，会发现更高的轮廓系数实际上出现在 `k=2`：

| k | 轮廓系数 |
|---|---:|
| 2 | 0.3484 |
| 3 | 0.2869 |
| 4 | 0.2657 |
| 5 | 0.2427 |
| 6 | 0.2390 |
| 7 | 0.2337 |

这说明一个很实际的问题：

**在聚类任务里，手动指定簇数很容易偏离最佳结构，最好用评估指标辅助决策。**

进一步看 `k=2` 的聚类中心，可以发现两类样本最主要的区别几乎集中在 `person` 这个字段上。换句话说，当前编码下，模型更像是在按“乘坐人数对应的数值大小”分组，而不是综合多维属性形成更自然的购买模式。

---

## 4. DBSCAN：分数最高，但解释性最差

接下来，Notebook 对 `DBSCAN` 做了参数搜索：

```python
eps = np.linspace(0.1, 1, 10).tolist()
min_samples = list(range(3, 10, 2))
```

最终找到的最佳参数是：

- `eps = 0.1`
- `min_samples = 3`

对应结果：

- 轮廓系数：`0.4958`

单看分数，这似乎是三种算法里表现最好的。但如果继续分析簇结构，会发现它的问题也最明显：

- 生成了 `314` 个簇
- 噪声点数量达到 `528`
- 大量簇的样本量只有 `4` 个左右

这说明 `DBSCAN` 虽然拿到了较高的轮廓系数，但它并没有形成一个“可解释、可用于业务理解”的聚类结果，而是把数据切得非常碎。

这也是聚类评估里经常被忽视的一点：

**高分不一定等于好结果。**

如果一个算法把数据分成数百个小簇，即使轮廓系数不错，这样的聚类通常也很难支持后续分析、画像或决策。

---

## 5. 分层聚类：思路正确，但代码里有个小坑

Notebook 最后使用了分层聚类，并在不同 `n_clusters` 和 `linkage` 之间尝试搜索最佳组合：

```python
for i in n_clusters:
    for j in linkage:
        agg = AgglomerativeClustering(n_clusters=i, linkage=j)
        y_ = agg.fit_predict(X)
        if score > score_high:
            score_high = score
            best_params['best'] = {'n_clusters': i, 'linkage': j}
```

这里有一个容易忽略的问题：

**循环里没有重新计算 `score`。**

也就是说，这段代码在比较参数时使用的是一个旧变量，导致打印出的最佳分数并不可靠。  
不过如果把这段逻辑补完整重新计算，最佳参数仍然是：

- `n_clusters = 2`
- `linkage = 'ward'`

对应轮廓系数大约为：

- `0.3484`

这个结果和 `KMeans(k=2)` 非常接近，说明在当前数据编码方式下，数据的主要结构更偏向“二分”，而不是更细粒度的多簇分布。

---

## 6. 这次实验最值得关注的三个结论

### 结论一：这份数据更像“弱聚类结构”

无论是 `KMeans` 还是分层聚类，最佳结果都集中在 `2` 类附近，而且轮廓系数并不算特别高。这说明样本之间存在一定分组趋势，但并没有特别清晰、特别强烈的天然簇结构。

### 结论二：`DBSCAN` 的高分具有误导性

`DBSCAN` 的分数最高，但代价是簇数量过多、噪声点过多。对于实际业务分析而言，这样的结果通常不如一个稍低分但更稳定、更容易解释的两类结构。

### 结论三：编码方式决定了聚类结果的上限

这次实验最核心的技术启发，不是“哪种算法更强”，而是：

**如果输入给聚类模型的距离关系本身就不合理，那么再换算法也很难得到真正高质量的簇。**

---

## 7. 如果继续优化，这个实验可以怎么做？

如果要把这份 Notebook 进一步打磨成更严谨的聚类分析，我会优先做下面几件事：

1. 使用 `One-Hot Encoding` 替代手工整数映射，避免人为制造虚假的大小关系。
2. 对距离度量更谨慎，如果特征本质上是类别型数据，可以考虑 `k-modes`、`k-prototypes` 或基于 `Gower distance` 的聚类方法。
3. 对 `KMeans` 固定 `random_state`，保证结果可复现。
4. 在分层聚类参数搜索里重新计算 `silhouette_score`，避免旧变量污染结果。
5. 不只看轮廓系数，还要同时看簇数量、簇规模分布和业务可解释性。

---

## 8. 总结

这次聚类实验非常适合作为一个入门案例：流程完整，涵盖了数据读取、类别编码、参数搜索和多算法比较，也能直观看到不同聚类方法在同一份数据上的行为差异。

但它更有价值的地方在于提醒我们：

**聚类的难点往往不在模型，而在数据表达。**

当类别特征被简单映射为数字后，算法看到的已经不是原始业务世界，而是一个被“距离规则”重塑过的空间。在这样的前提下，`KMeans`、`DBSCAN` 和分层聚类的表现差异，某种程度上也是输入表示方式差异的放大结果。

所以，真正高质量的无监督学习，往往不是从“换一个算法”开始，而是从“重新定义数据如何被表示”开始。
