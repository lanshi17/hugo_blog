---
# 核心元数据
author: lanshi
date: "2026-04-04T12:00:00+08:00"
lastmod: "2025-04-04T22:00:00+08:00"
title: "工业蒸汽量预测实战：SVR、Lasso、ElasticNet 与加权融合"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: "基于天池工业蒸汽量预测任务，对比 SVR、Lasso、ElasticNet 等回归模型表现，并通过加权平均融合与权重搜索进一步提升预测效果。"

# 内容分类
series:
  - 机器学习实战
tags:
  - 回归预测
  - SVR
  - Lasso
  - ElasticNet
  - 模型融合
  - 加权平均
  - 工业蒸汽量预测
  - 天池
  - 数据挖掘
categories:
  - 机器学习

# SEO优化
description: "本文围绕天池工业蒸汽量预测任务，使用 SVR、Lasso、ElasticNet 等回归模型进行建模，并结合加权平均融合、最优权重搜索与结果可视化，系统分析回归预测与模型融合实践。"
keywords:
  - 回归预测
  - SVR
  - Lasso回归
  - ElasticNet
  - 模型融合
  - 加权平均
  - 工业蒸汽量预测
  - 天池数据挖掘
  - sklearn
  - 回归建模

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "steam-prediction-cover.png"
  alt: "工业蒸汽量预测与模型融合封面"
  caption: "SVR、Lasso、ElasticNet 与加权融合"
  relative: true

# 版权声明
copyright: true
---
# 1. 项目背景

这次任务是对 **天池工业蒸汽量** 进行预测，本质上属于一个典型的 **回归问题**。
目标是根据训练集中的多维特征，学习目标变量 `target` 的变化规律，并对测试集进行预测。

本次实验主要完成了以下几类模型：

- **SVR 支持向量回归**

  - linear 核
  - rbf 核
  - poly 核
  - sigmoid 核
- **Lasso 回归**
- **ElasticNet 弹性网络**
- **多模型加权平均融合**

同时，还需要对比：

1. 不同算法单独建模的效果差异
2. 模型融合前后的得分变化
3. 不同权重分配对融合效果的影响

---

# 2. 实验目标

本实验希望回答以下几个问题：

- 哪一种单模型在工业蒸汽量预测任务上效果更好？
- 线性模型和非线性模型的表现差异如何？
- 使用多个模型的加权平均，能否进一步提升预测效果？
- 如果进行双模型融合，权重如何选择更合理？

---

# 3. 方法概览

本次实验采用以下技术路线：

## 3.1 单模型建模

使用三类典型回归模型：

### 1）Lasso

Lasso 是带有 **L1 正则化** 的线性回归模型。
它的特点是：

- 可以抑制过拟合
- 可以让部分特征系数变成 0
- 具有一定的特征选择作用

### 2）ElasticNet

ElasticNet 是 **L1 + L2 正则化** 的结合体。
相对于 Lasso，它在特征相关性较强时通常更加稳定。

### 3）SVR（Support Vector Regression）

支持向量回归是支持向量机在回归任务中的扩展。
为了探索不同核函数的效果，本实验尝试了：

- `linear`
- `rbf`
- `poly`
- `sigmoid`

其中，`rbf` 核通常更适合拟合非线性关系。

---

## 3.2 模型融合

为了进一步提升预测效果，本实验将多个模型的预测结果进行 **加权平均融合**。

典型形式如下：

### 双模型融合

$$
\hat{y} = 0.8 \times \hat{y}_{ElasticNet} + 0.2 \times \hat{y}_{SVR-rbf}
$$

### 三模型融合

$$
\hat{y} = 0.2 \times \hat{y}_{Lasso} + 0.5 \times \hat{y}_{ElasticNet} + 0.3 \times \hat{y}_{SVR-rbf}
$$

### 简单平均

$$
\hat{y} = \frac{\hat{y}_{Lasso} + \hat{y}_{ElasticNet} + \hat{y}_{SVR-rbf}}{3}
$$

此外，还通过遍历权重的方式，自动寻找双模型最优融合比例。

---

# 4. 数据读取与预处理

首先读取训练集与测试集，并将训练集划分为训练子集和验证子集，用于模型评估。

```python
import os
import numpy as np
import pandas as pd
import joblib

from loguru import logger
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("./outputs", exist_ok=True)

train_data = pd.read_csv("./data/zhengqi_train.txt", sep="\t")
X_test = pd.read_csv("./data/zhengqi_test.txt", sep="\t")

X = train_data.drop(columns="target")
y = train_data["target"]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 为什么要做标准化？

在本实验中，所有模型都统一放入了 `Pipeline`，并配合 `StandardScaler()` 进行标准化，原因主要有两点：

1. **SVR 对特征尺度非常敏感**
2. **Lasso / ElasticNet 带正则项，特征量纲不一致时会影响模型训练**

所以，统一标准化是非常有必要的。

---

# 5. 模型评估函数设计

为了便于统一比较不同模型的效果，我封装了一个模型评估函数，输出以下指标：

- MSE
- RMSE
- MAE
- R²

```python
def evaluate_model(name, model, X_tr, y_tr, X_val, y_val):
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)

    logger.info(f"{name} -> MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")

    return {
        "model": name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "val_pred": val_pred
    }
```

---

# 6. 构建模型

本次实验共构建了 6 个单模型：

```python
models = {
    'Lasso': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.01, random_state=42, max_iter=10000))
    ]),
    'ElasticNet': Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000))
    ]),
    'SVR-linear': Pipeline([
        ('scaler', StandardScaler()),
        ('model', svm.SVR(kernel='linear', C=10, epsilon=0.1))
    ]),
    'SVR-rbf': Pipeline([
        ('scaler', StandardScaler()),
        ('model', svm.SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1))
    ]),
    'SVR-poly': Pipeline([
        ('scaler', StandardScaler()),
        ('model', svm.SVR(kernel='poly', degree=3, C=10, epsilon=0.1, gamma='scale'))
    ]),
    'SVR-sigmoid': Pipeline([
        ('scaler', StandardScaler()),
        ('model', svm.SVR(kernel='sigmoid', C=10, epsilon=0.1, gamma='scale'))
    ])
}
```

---

# 7. baseline：单模型效果对比

接下来在验证集上评估所有单模型：

```python
results = []
val_pred_dict = {}

for name, model in models.items():
    res = evaluate_model(name, model, X_tr, y_tr, X_val, y_val)
    results.append({
        "model": res["model"],
        "mse": res["mse"],
        "rmse": res["rmse"],
        "mae": res["mae"],
        "r2": res["r2"]
    })
    val_pred_dict[name] = res["val_pred"]

results_df = pd.DataFrame(results).sort_values(by="rmse")
print("单模型结果：")
print(results_df)
```

---

## 单模型分析思路

在这一步，重点关注以下几点：

### 1）RMSE 越小越好

RMSE 是回归任务中非常常见的评价指标，对误差较大的样本更敏感。

### 2）R² 越接近 1 越好

说明模型对数据变化规律的解释能力越强。

### 3）观察不同模型的偏好

通常可以预期：

- **Lasso / ElasticNet**：更稳定，偏线性
- **SVR-rbf**：通常更擅长非线性关系拟合
- **SVR-poly / sigmoid**：有时不如 rbf 稳定

---

# 8. 模型加权平均融合

为了进一步提升预测效果，对多个模型的输出进行融合。

---

## 8.1 固定权重融合

### 方案一：ElasticNet + SVR-rbf

```python
fusion_pred_1 = 0.8 * val_pred_dict["ElasticNet"] + 0.2 * val_pred_dict["SVR-rbf"]
```

### 方案二：Lasso + ElasticNet + SVR-rbf

```python
fusion_pred_2 = (
    0.2 * val_pred_dict["Lasso"] +
    0.5 * val_pred_dict["ElasticNet"] +
    0.3 * val_pred_dict["SVR-rbf"]
)
```

### 方案三：简单平均

```python
fusion_pred_3 = (
    val_pred_dict["Lasso"] +
    val_pred_dict["ElasticNet"] +
    val_pred_dict["SVR-rbf"]
) / 3
```

然后统一评估：

```python
fusion_results = []

for name, pred in {
    "Fusion1_ElasticNet0.8_SVRrbf0.2": fusion_pred_1,
    "Fusion2_Lasso0.2_ElasticNet0.5_SVRrbf0.3": fusion_pred_2,
    "Fusion3_Average": fusion_pred_3
}.items():
    mse = mean_squared_error(y_val, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)

    logger.info(f"{name} -> MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")

    fusion_results.append({
        "model": name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })

fusion_df = pd.DataFrame(fusion_results)
print("融合模型结果：")
print(fusion_df)
```

---

## 8.2 融合的意义

为什么加权平均有时会更好？

因为不同模型的误差模式并不完全一致。

比如：

- 线性模型在某些样本上预测偏高
- 非线性模型在另一些样本上预测偏低

这时通过合理加权，模型之间就可能产生 **误差互补效应**，从而让整体预测更稳定。

---

# 9. 自动搜索最优融合权重

除了手工设定权重，还可以遍历权重寻找更优解：

```python
best_rmse = float("inf")
best_weight = None

for w in np.arange(0.0, 1.01, 0.1):
    pred = w * val_pred_dict["ElasticNet"] + (1 - w) * val_pred_dict["SVR-rbf"]
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w

print(f"最佳双模型权重: ElasticNet={best_weight:.1f}, SVR-rbf={1-best_weight:.1f}")
print(f"最佳双模型RMSE: {best_rmse:.6f}")
```

---

## 这一步说明了什么？

这一步很适合写进实验报告，因为它体现了：

- 模型融合不是随便平均
- 权重会直接影响最终效果
- 表现更强的模型通常应分配更高权重
- 但不能完全压制另一个模型，否则就失去了融合的意义

---

# 10. 全量数据训练与测试集预测

在验证集上比较完成后，选定最终模型，并在完整训练集 `X, y` 上重新训练，然后预测测试集。

```python
final_models = {
    'Lasso': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.01, random_state=42, max_iter=10000))
    ]),
    'ElasticNet': Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000))
    ]),
    'SVR-rbf': Pipeline([
        ('scaler', StandardScaler()),
        ('model', svm.SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1))
    ])
}

for name, model in final_models.items():
    model.fit(X, y)
    joblib.dump(model, f"./outputs/{name}.pkl")

lasso_test_pred = final_models["Lasso"].predict(X_test)
elastic_test_pred = final_models["ElasticNet"].predict(X_test)
svr_rbf_test_pred = final_models["SVR-rbf"].predict(X_test)

final_pred = (
    0.2 * lasso_test_pred +
    0.5 * elastic_test_pred +
    0.3 * svr_rbf_test_pred
)

np.savetxt("./outputs/fusion_pred.txt", final_pred)
print("最终融合预测已保存到 ./outputs/fusion_pred.txt")
```

---

## 为什么最后要用全量训练集重训？

因为验证集只是为了比较模型优劣。
真正预测测试集时，应尽量利用全部已知训练样本，让模型学习到更多信息，从而提升泛化能力。

---

# 11. 结果可视化分析

虽然测试集没有真实标签，无法直接绘制“真实值 vs 预测值”对比图，但仍然可以通过以下方式观察模型输出特征。

---

## 11.1 构造预测结果表

```python
pred_df = pd.DataFrame({
    "Lasso": lasso_test_pred,
    "ElasticNet": elastic_test_pred,
    "SVR-rbf": svr_rbf_test_pred,
    "Fusion": final_pred
})

pred_df.head()
```

---

## 11.2 预测分布直方图

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.hist(pred_df["Lasso"], bins=40, alpha=0.5, label="Lasso")
plt.hist(pred_df["ElasticNet"], bins=40, alpha=0.5, label="ElasticNet")
plt.hist(pred_df["SVR-rbf"], bins=40, alpha=0.5, label="SVR-rbf")
plt.hist(pred_df["Fusion"], bins=40, alpha=0.5, label="Fusion")

plt.xlabel("Predicted Value")
plt.ylabel("Frequency")
plt.title("Distribution of Predictions from Different Models")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 图像解读

这张图主要用于观察：

- 不同模型预测值的整体分布是否相近
- 融合后预测是否更加平滑
- 是否存在某个模型预测明显偏离其他模型的情况

---

## 11.3 融合前后对比图

```python
plt.figure(figsize=(14, 6))

n_show = 120
plt.plot(pred_df["ElasticNet"][:n_show].values, label="ElasticNet", alpha=0.8)
plt.plot(pred_df["SVR-rbf"][:n_show].values, label="SVR-rbf", alpha=0.8)
plt.plot(pred_df["Fusion"][:n_show].values, label="Fusion", color="red", linewidth=2)

plt.title("Comparison Before and After Fusion")
plt.xlabel("Sample Index")
plt.ylabel("Predicted Value")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 图像解读

这张图非常适合放在作业报告中，因为它能直观展示：

- 融合结果整体趋势与单模型接近
- 融合曲线常常位于多个模型之间
- 与单一模型相比，融合结果通常更平滑、更稳定

---

# 12. 实验结论

结合本次实验，可以总结出以下几点：

## 1）不同算法表现存在明显差异

- Lasso 和 ElasticNet 属于线性模型，整体比较稳定；
- ElasticNet 通常比 Lasso 更灵活，在特征存在相关性时更有优势；
- SVR 中不同核函数效果差异明显，其中 **RBF 核** 通常更适合处理复杂非线性关系。

## 2）模型融合通常优于单模型

- 单模型往往只擅长拟合某一类模式；
- 加权融合能够结合多个模型的优势，减少单模型误差带来的波动；
- 当权重设置合理时，融合模型通常在验证集上能获得更低的 RMSE。

## 3）权重分配很关键

- 不是简单平均就一定最好；
- 也不是某个强模型权重越高越好；
- 最优权重需要根据验证集实验结果来确定。

---

# 13. 本实验的优点与不足

## 优点

- 同时比较了多种单模型
- 引入了融合策略，结果更完整
- 通过验证集评估避免了只看测试集预测的盲目性
- 加入可视化分析，结果更直观

## 不足

- 目前超参数主要是手工设置，仍有进一步优化空间
- 融合方式仍然较简单，仅采用加权平均
- 验证方式采用单次划分，结果可能受随机划分影响

---

# 14. 后续优化方向

如果继续提升模型效果，可以尝试以下方向：

## 1）引入 GridSearchCV 自动调参

对以下参数进行搜索：

- Lasso 的 `alpha`
- ElasticNet 的 `alpha` 和 `l1_ratio`
- SVR 的 `C`、`epsilon`、`gamma`、`kernel`

## 2）使用 K 折交叉验证

相比单次 train/val split，KFold 更稳定，更适合做模型选择。

## 3）尝试更复杂的融合方式

例如：

- Stacking
- Blending
- Boosting 类模型再融合

## 4）做特征工程

包括：

- 异常值处理
- 特征筛选
- 特征组合
- 降维处理

---

# 15. 完整代码思路总结

本项目的完整建模流程可以总结为：

1. 读取训练集和测试集
2. 划分训练集 / 验证集
3. 构建 Lasso、ElasticNet、SVR 多个模型
4. 统一评估各模型在验证集上的表现
5. 对多个模型预测结果做加权平均融合
6. 搜索更优融合权重
7. 在全量训练集上重新训练最终模型
8. 对测试集输出最终预测结果
9. 通过可视化分析模型输出分布与融合效果

---

> 本实验围绕天池工业蒸汽量预测任务，分别使用了 Lasso、ElasticNet 和多核函数 SVR 进行建模，并进一步尝试了加权平均融合策略。实验结果表明，单模型之间存在明显性能差异，其中 ElasticNet 与 SVR-rbf 往往具有较好的表现；进一步将多个模型进行融合后，能够在一定程度上提升预测稳定性和泛化能力。整体来看，模型融合是一种简单但有效的性能优化手段，对于回归类任务尤其值得尝试。
