---
# 核心元数据
author: lanshi
date: "2026-03-31T12:00:00+08:00"
lastmod: "2026-03-31T22:00:00+08:00"
title: "模型融合实战笔记：从单模型到多模型 Bagging"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: "从平均法、加权平均法到 Stacking，系统梳理机器学习中的模型融合思路，并结合 Bagging 平均融合给出完整实战流程。"

# 内容分类
series:
  - 机器学习实战
tags:
  - 机器学习
  - 模型融合
  - Bagging
  - 回归
  - LightGBM
  - XGBoost
  - sklearn
categories:
  - 机器学习

# SEO优化
description: "本文系统介绍模型融合的常见方法，包括平均法、加权平均法与 Stacking，并结合多个回归模型演示 Bagging 平均融合的完整实战流程。"
keywords:
  - 模型融合
  - Bagging
  - Averaging
  - Stacking
  - 机器学习
  - LightGBM
  - XGBoost
  - 回归建模

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "model-ensemble-cover.png"
  alt: "模型融合实战笔记封面"
  caption: "从单模型到多模型 Bagging"
  relative: true

# 版权声明
copyright: true
---
# 模型融合实战笔记：从单模型到多模型 Bagging

## 1. 为什么要做模型融合？

在机器学习竞赛和实际业务建模中，单一模型往往很难同时兼顾稳定性、泛化能力、非线性拟合能力以及对噪声的鲁棒性。不同模型擅长的方向也不一样：

- **线性模型**：稳定、可解释性强
- **SVR**：适合处理中小规模复杂关系
- **随机森林**：抗过拟合能力较强
- **GBDT / XGBoost / LightGBM**：对表格数据通常表现优秀

因此，一个很自然的思路就是：

> 让多个模型分别学习，再把它们的预测结果融合起来。

这就是模型融合。

---

## 2. 常见模型融合方式

### 2.1 平均法（Averaging / Bagging）

多个模型预测后直接求平均：

\[
\hat y = \frac{1}{M}\sum_{i=1}^{M}\hat y_i
\]

优点：

- 简单
- 稳定
- 很适合快速提升成绩

缺点：

- 所有模型权重一样
- 差模型会拉低整体表现

---

### 2.2 加权平均法（Weighted Averaging）

给更好的模型更高权重：

\[
\hat y = \sum_{i=1}^{M} w_i \hat y_i,\quad \sum w_i = 1
\]

优点：

- 比平均法更灵活
- 可以利用交叉验证结果给权重

缺点：

- 权重设计麻烦
- 如果验证集不稳定，容易过拟合权重

---

### 2.3 Stacking

第一层多个基模型输出预测结果，第二层模型再学习这些结果。

优点：

- 表达能力强
- 通常比简单平均更进一步

缺点：

- 流程复杂
- 容易数据泄露
- 实现成本高

---

## 3. 本文采用的融合策略

本文采用最基础、但非常有效的一种方式：

> **Bagging 平均融合**

即：

- 训练多个不同回归模型
- 每个模型分别预测测试集
- 最后对结果做平均

这种方式非常适合作为竞赛 baseline，也适合在项目初期快速验证融合是否有效。

---

## 4. 项目数据说明

本文使用两类数据：

### 4.1 未降维数据
经过特征工程处理后的原始特征数据。

### 4.2 PCA 降维数据
对特征做主成分降维后得到的数据。

虽然本文主要基于未降维数据完成模型调参与融合，但保留 PCA 数据读取逻辑，方便后续扩展实验。

---

## 5. 整体流程

整个建模流程可以概括为：

```text
读取数据
→ 划分训练/验证
→ 构建多个候选模型
→ GridSearchCV 搜索最优参数
→ 比较交叉验证误差
→ 多模型平均融合
→ 输出最终结果
```

---

## 6. 导入依赖库

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error

from xgboost import XGBRFRegressor
import lightgbm as lgb
```

---

## 7. 数据加载与划分

### 7.1 读取原始特征数据

```python
all_data = pd.read_csv('./data/processed_zhengqi_data2.csv')

train_data = all_data[all_data['label'] == 'train'].copy()
train_data = train_data.drop(columns=['label'])

X = train_data.drop(columns=['target'])
y = train_data['target']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

submit_test = all_data[all_data['label'] == 'test'].copy()
submit_test = submit_test.drop(columns=['label', 'target'])
```

### 7.2 读取 PCA 数据

```python
train_pca = np.load('./data/train_data_pca.npz')
test_pca = np.load('./data/test_data_pca.npz')

X_pca = train_pca['X_train']
y_pca = train_pca['y_train']
submit_test_pca = test_pca['X_test']
```

---

## 8. 封装统一训练函数

为了方便对不同模型执行统一的调参流程，可以封装一个 `train_model()` 函数。这个函数主要负责：

1. 构建 `RepeatedKFold`
2. 使用 `GridSearchCV` 搜索最优参数
3. 输出最优模型、训练集拟合效果和交叉验证结果
4. 绘制预测值与残差分布图

```python
def train_model(model, param_grid=None, X=None, y=None, splits=5, repeats=5):
    if param_grid is None:
        param_grid = {}

    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=42)

    gsearch = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=rkfold,
        scoring='neg_mean_squared_error',
        verbose=1,
        return_train_score=True,
        n_jobs=-1
    )

    gsearch.fit(X, y)

    best_model = gsearch.best_estimator_
    best_idx = gsearch.best_index_
    grid_results = pd.DataFrame(gsearch.cv_results_)

    cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
    cv_std = grid_results.loc[best_idx, 'std_test_score']
    cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

    y_pred = best_model.predict(X)

    print('----------------------')
    print(best_model)
    print('----------------------')
    print('score =', best_model.score(X, y))
    print('mse =', mean_squared_error(y, y_pred))
    print('cross_val: mean =', cv_mean, ', std =', cv_std)

    y_pred = pd.Series(y_pred, index=y.index if hasattr(y, 'index') else None)
    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y

    resid = y_series.reset_index(drop=True) - y_pred.reset_index(drop=True)
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    n_outliers = sum(abs(z) > 3)

    plt.figure(figsize=(15, 5))

    ax1 = plt.subplot(1, 3, 1)
    plt.plot(y_series, y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.title('corr = {:.3f}'.format(np.corrcoef(y_series, y_pred)[0][1]))

    ax2 = plt.subplot(1, 3, 2)
    plt.plot(y_series, y_series - y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y - y_pred')
    plt.title('std resid = {:.3f}'.format(std_resid))

    ax3 = plt.subplot(1, 3, 3)
    pd.Series(z).plot.hist(bins=50, ax=ax3)
    plt.xlabel('z')
    plt.title('{:.0f} samples with |z| > 3'.format(n_outliers))

    plt.tight_layout()
    plt.show()

    return best_model, cv_score, grid_results
```

---

## 9. 各模型调参与分析

---

## 9.1 Ridge 岭回归

### 原理
Ridge 在线性回归基础上加入 \(L_2\) 正则化，能够缓解多重共线性问题。

### 调参重点
- `alpha`：正则化强度

```python
model = Ridge()
alphas = np.arange(0.1, 5.0, 0.2)
param_grid = {'alpha': alphas}
```

### 特点总结
- 表现稳定
- 对线性关系建模较好
- 适合作为基线模型

---

## 9.2 Lasso 套索回归

### 原理
Lasso 使用 \(L_1\) 正则化，除了控制过拟合，还能做一定的特征压缩。

### 调参重点
- `alpha`

```python
model = Lasso(max_iter=10000)
alphas = np.arange(1e-4, 1e-3, 4e-5)
param_grid = {'alpha': alphas}
```

### 特点总结
- 有一定特征选择能力
- 更容易得到稀疏解
- 在高维场景中常有效

---

## 9.3 ElasticNet 弹性网络

### 原理
ElasticNet 综合了 Ridge 和 Lasso：

- \(L_1\) 用于稀疏化
- \(L_2\) 用于稳定训练

### 调参重点
- `alpha`
- `l1_ratio`

```python
model = ElasticNet(max_iter=10000)
param_grid = {
    'alpha': np.arange(1e-4, 1e-3, 1e-4),
    'l1_ratio': np.arange(0.1, 1.0, 0.1)
}
```

### 特点总结
- 比 Lasso 更稳定
- 适合特征间相关性较强的任务

---

## 9.4 SVR 支持向量回归

### 原理
SVR 通过核方法拟合复杂非线性关系。

### 调参重点
- `C`
- `epsilon`

```python
model = SVR()
param_grid = {
    'C': np.arange(0.1, 1.0, 0.2),
    'epsilon': np.arange(0.1, 1.0, 0.2)
}
```

### 特点总结
- 适合中小规模数据
- 对非线性关系有较好表达能力
- 训练速度通常比线性模型慢

---

## 9.5 GBDT

### 原理
GBDT 通过不断拟合残差提升模型性能，是表格数据常见强模型。

### 调参重点
- `n_estimators`
- `max_depth`
- `min_samples_split`

```python
model = GradientBoostingRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 6, 7]
}
```

### 特点总结
- 对结构化数据效果通常较强
- 能较好处理非线性关系
- 是融合中的常见核心模型

---

## 9.6 Random Forest 随机森林

### 原理
随机森林通过多棵树做平均，降低方差，提高稳定性。

### 调参重点
- `n_estimators`
- `max_features`
- `min_samples_split`

```python
model = RandomForestRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': [12, 16, 20, 24],
    'min_samples_split': [5, 7, 9]
}
```

### 特点总结
- 稳定
- 抗过拟合能力较强
- 非常适合作为 baseline

---

## 9.7 XGBoost

本文沿用原始 notebook 中的 `XGBRFRegressor`。

### 调参重点
- `n_estimators`
- `max_depth`
- `reg_lambda`

```python
model = XGBRFRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5],
    'reg_lambda': np.arange(1e-5, 1e-3, 1e-4)
}
```

### 特点总结
- 拟合能力强
- 对复杂模式建模效果通常不错
- 是融合中的重要候选模型

---

## 9.8 LightGBM

### 调参重点
- `n_estimators`
- `max_depth`
- `min_child_weight`
- `reg_alpha`
- `reg_lambda`

```python
model = lgb.LGBMRegressor(random_state=42)
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5],
    'min_child_weight': [3, 4, 5],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1],
    'reg_lambda': [1e-5, 1e-2, 0.1, 1]
}
```

### 特点总结
- 在表格数据任务中常表现优异
- 训练效率较高
- 非常适合作为主力模型

---

## 10. 模型结果汇总

调参完成后，可以统一查看交叉验证表现：

```python
score_models.sort_values(by='mean')
```

其中：

- `mean` 越小越好（这里采用的是 MSE）
- `std` 越小表示模型越稳定

一般建议：

- 优先保留表现好的模型进入融合
- 不建议把明显表现较差的模型也一起平均

---

## 11. 多模型 Bagging 融合

定义一个简单的融合函数，对多个模型的预测结果直接平均：

```python
def model_predict(submit_test):
    i = 0
    y_predict_total = np.zeros(submit_test.shape[0])

    for model_name in opt_models.keys():
        print(f'使用模型预测：{model_name}')
        y_predict = opt_models[model_name].predict(submit_test)
        y_predict_total += y_predict
        i += 1

    y_predict_mean = np.round(y_predict_total / i, 3)
    return y_predict_mean
```

保存结果：

```python
result = model_predict(submit_test)
np.savetxt('./export_data/bagging_result.txt', result)
```

---

## 12. 为什么简单平均往往有效？

因为不同模型的误差方向往往不同。

例如：

- Ridge 偏保守
- GBDT 偏强拟合
- RandomForest 偏稳健
- XGBoost 更善于学习复杂非线性关系

如果把这些模型直接平均：

- 某些模型高估一点
- 某些模型低估一点

平均后，整体误差往往会被抵消一部分。

本质上可以理解为：

> 用多个模型互相“纠偏”。

---

## 13. 实战经验总结

### 13.1 融合不是模型越多越好
如果某个模型明显表现很差，加入平均反而可能拉低整体效果。

### 13.2 先看交叉验证，再决定保留哪些模型
不要机械地把所有模型都放进去平均。

### 13.3 简单平均是起点，不是终点
当简单平均有效之后，可以继续尝试：

- 加权平均
- Stacking
- Blending

### 13.4 线性模型和树模型通常互补
这是最常见、也最容易产生收益的一种组合。

---

## 14. 后续优化方向

如果继续向上优化，可以考虑以下几个方向。

### 方向一：加权融合

根据交叉验证表现设置权重：

```python
final_pred = 0.1 * ridge_pred + 0.2 * rf_pred + 0.3 * gbdt_pred + 0.4 * lgb_pred
```

---

### 方向二：不同特征空间分别建模

例如：

- 原始特征训练一组模型
- PCA 特征训练一组模型
- 最后再做二次融合

---

### 方向三：Stacking

使用第一层模型预测结果作为第二层模型输入，进一步挖掘模型间互补信息。

---

### 方向四：引入标准化 Pipeline

尤其对这些模型更重要：

- Lasso
- ElasticNet
- SVR

---

## 15. 一句话总结

模型融合最核心的思想就是：

> **不要把希望压在一个模型上，而是让多个模型共同投票。**

对于结构化数据任务来说，哪怕只是最简单的平均融合，往往都能带来稳定收益。
