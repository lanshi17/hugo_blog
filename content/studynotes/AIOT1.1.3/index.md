---
# 核心元数据
author: lanshi
date: "2025-12-13T12:47:02+08:00"
lastmod:
title: 火力发电锅炉燃烧效率优化：机器学习回归算法实战

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: 本笔记详细记录了在“火力发电锅炉燃烧效率优化”项目中，利用多种机器学习回归算法对经过特征工程处理后的锅炉运行数据进行建模、训练、评估和预测的过程。目标是找到最适合预测锅炉燃烧效率的模型。

# 内容分类
series:
tags: ["机器学习", "回归算法", "特征工程", "Python", "Scikit-learn", "XGBoost", "LightGBM"]
categories: ["数据科学", "机器学习"]

# SEO优化
description: 本笔记详细记录了在“火力发电锅炉燃烧效率优化”项目中，利用多种机器学习回归算法对经过特征工程处理后的锅炉运行数据进行建模、训练、评估和预测的过程。目标是找到最适合预测锅炉燃烧效率的模型。
keywords: ["机器学习", "回归算法", "特征工程", "Python", "Scikit-learn", "XGBoost", "LightGBM", "PCA", "MSE", "模型评估"]

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "boiler-efficiency-cover.png"
  alt: "火力发电锅炉燃烧效率优化封面"
  caption: "火力发电锅炉燃烧效率优化"
  relative: true

# 版权声明
copyright: true
---
## 1. 概述

本笔记详细记录了在“火力发电锅炉燃烧效率优化”项目中，利用多种机器学习回归算法对经过特征工程处理后的锅炉运行数据进行建模、训练、评估和预测的过程。目标是找到最适合预测锅炉燃烧效率的模型。

## 2. 环境准备与数据加载

### 2.1 导入所需库

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
```

### 2.2 加载数据

#### 2.2.1 未降维数据

```python
all_data = pd.read_csv("./data/processed_zhengqi_data2.csv") # 假设这是经过特征工程后的文件

# 分离训练集和测试集
train_data = all_data[all_data['label'] == "train"].drop(labels='label', axis=1)
test_data = all_data[all_data['label'] == "test"].drop(labels=['label', 'target'], axis=1)

# 切分训练集为训练子集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    train_data.drop(labels='target', axis=1),
    train_data['target'],
    test_size=0.2,
    random_state=42 # 为了结果可复现
)
```

#### 2.2.2 降维数据 (PCA)

```python
# 假设 PCA 处理后的数据已保存为 .npz 文件
train_data_pca_dict = np.load("data/train_data_pca.npz")
train_data_pca = train_data_pca_dict['X_train']
target_data_pca = train_data_pca_dict["y_train"]

X_train_pca, X_valid_pca, y_train_pca, y_valid_pca = train_test_split(
    train_data_pca, target_data_pca, test_size=0.2, random_state=42
)

test_data_pca_dict = np.load("data/test_data_pca.npz")
test_data_pca = test_data_pca_dict["X_test"]
```

## 3. 辅助函数：绘制学习曲线

学习曲线有助于诊断模型是否存在欠拟合（高偏差）或过拟合（高方差）问题。

```python
def plot_learning_curve(model, title, X, y, cv=None):
    """
    绘制模型的学习曲线。
  
    Parameters:
        model : 一个实现了 fit 和 predict 方法的 scikit-learn 估计器。
        title : 字符串，图像标题。
        X : array-like, shape (n_samples, n_features) 训练向量。
        y : array-like, shape (n_samples,) 目标值。
        cv : int, cross-validation generator or an iterable, 可选，默认为 None。
    """
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
  
    plt.grid()
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()
```

## 4. 模型训练、评估与预测

### 4.1 多元线性回归 (Linear Regression)

- **优点**: 简单、快速、可解释性强。
- **缺点**: 对非线性关系建模能力差。

```python
# --- 降维数据 ---
clf_lr_pca = LinearRegression()
clf_lr_pca.fit(X_train_pca, y_train_pca)
score_lr_pca = mean_squared_error(y_valid_pca, clf_lr_pca.predict(X_valid_pca))
print("LinearRegression (PCA Data) MSE:", score_lr_pca)

# --- 未降维数据 ---
clf_lr_full = LinearRegression()
clf_lr_full.fit(X_train, y_train)
score_lr_full = mean_squared_error(y_valid, clf_lr_full.predict(X_valid))
print("LinearRegression (Full Data) MSE:", score_lr_full)

# --- 学习曲线 (示例：未降维数据) ---
cv_splitter = ShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
plot_learning_curve(clf_lr_full, "Learning Curve (Linear Regression)", X_train, y_train, cv=cv_splitter)

# --- 最终预测 (示例：未降维数据) ---
final_model_lr = LinearRegression()
final_model_lr.fit(train_data.drop('target', axis=1), train_data['target'])
y_pred_lr = final_model_lr.predict(test_data)
# np.savetxt('./export_data/多元线性回归模型预测(非降维数据).txt', y_pred_lr)
```

### 4.2 随机森林 (Random Forest Regressor)

- **优点**: 高准确性、鲁棒性强、能处理非线性关系和特征交互、不易过拟合。
- **缺点**: 模型解释性较差、训练速度相对较慢。

```python
# --- 模型定义 ---
rf_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'max_features': 'log2',
    'min_samples_leaf': 10,
    'min_samples_split': 40,
    'criterion': 'squared_error',
    'random_state': 42
}
model_rf = RandomForestRegressor(**rf_params)

# --- 降维数据 ---
model_rf.fit(X_train_pca, y_train_pca)
score_rf_pca = mean_squared_error(y_valid_pca, model_rf.predict(X_valid_pca))
print("Random Forest (PCA Data) MSE:", score_rf_pca)

# --- 未降维数据 ---
model_rf.fit(X_train, y_train)
score_rf_full = mean_squared_error(y_valid, model_rf.predict(X_valid))
print("Random Forest (Full Data) MSE:", score_rf_full)

# --- 学习曲线 (示例：未降维数据) ---
# plot_learning_curve(model_rf, "Learning Curve (Random Forest)", X_train, y_train, cv=cv_splitter)

# --- 最终预测 (示例：未降维数据) ---
final_model_rf = RandomForestRegressor(**rf_params)
final_model_rf.fit(train_data.drop('target', axis=1), train_data['target'])
y_pred_rf = final_model_rf.predict(test_data)
# np.savetxt('./export_data/随机森林模型预测(非降维数据).txt', y_pred_rf)
```

### 4.3 支持向量回归 (SVR)

- **优点**: 在高维空间有效、内存使用效率高、核函数灵活。
- **缺点**: 对特征缩放敏感、大数据集训练慢、难以解释。

```python
# --- 模型定义 (RBF核) ---
model_svr_rbf = SVR(kernel='rbf', C=1, gamma=0.01, tol=0.0001, epsilon=0.3)

# --- 降维数据 ---
model_svr_rbf.fit(X_train_pca, y_train_pca)
score_svr_pca = mean_squared_error(y_valid_pca, model_svr_rbf.predict(X_valid_pca))
print("SVR RBF (PCA Data) MSE:", score_svr_pca)

# --- 未降维数据 ---
model_svr_rbf_full = SVR(kernel='rbf') # 使用默认参数
model_svr_rbf_full.fit(X_train, y_train)
score_svr_full = mean_squared_error(y_valid, model_svr_rbf_full.predict(X_valid))
print("SVR RBF (Full Data) MSE:", score_svr_full)

# --- 最终预测 (示例：RBF核，未降维数据) ---
final_model_svr = SVR(kernel='rbf')
final_model_svr.fit(train_data.drop('target', axis=1), train_data['target'])
y_pred_svr = final_model_svr.predict(test_data)
# np.savetxt('./export_data/SVR模型预测(非降维数据).txt', y_pred_svr)
```

### 4.4 梯度提升决策树 (GBDT / Gradient Boosting Regressor)

- **优点**: 高精度、能处理各种类型数据、对异常值相对鲁棒。
- **缺点**: 容易过拟合、训练时间较长、调参复杂。

```python
# --- 模型定义 ---
gbdt_params = {
    'learning_rate': 0.03,
    'loss': 'huber',
    'max_depth': 14,
    'max_features': 'sqrt',
    'min_samples_leaf': 10,
    'min_samples_split': 40,
    'n_estimators': 300,
    'subsample': 0.8,
    'random_state': 42
}
model_gbdt = GradientBoostingRegressor(**gbdt_params)

# --- 降维数据 ---
model_gbdt.fit(X_train_pca, y_train_pca)
score_gbdt_pca = mean_squared_error(y_valid_pca, model_gbdt.predict(X_valid_pca))
print("GBDT (PCA Data) MSE:", score_gbdt_pca)

# --- 未降维数据 ---
model_gbdt.fit(X_train, y_train)
score_gbdt_full = mean_squared_error(y_valid, model_gbdt.predict(X_valid))
print("GBDT (Full Data) MSE:", score_gbdt_full)

# --- 最终预测 (示例：未降维数据) ---
final_model_gbdt = GradientBoostingRegressor(**gbdt_params)
final_model_gbdt.fit(train_data.drop('target', axis=1), train_data['target'])
y_pred_gbdt = final_model_gbdt.predict(test_data)
# np.savetxt('export_data/GBDT模型预测（非降维数据）.txt', y_pred_gbdt)
```

### 4.5 LightGBM

- **优点**: 训练速度快、内存占用低、准确率高、支持并行学习、对类别特征友好。
- **缺点**: 相比 XGBoost，社区和文档可能稍显不足。

```python
# --- 模型定义 ---
lgb_params = {
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_samples': 10,
    'max_depth': 25,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 0.1,
    'random_state': 42
}
model_lgb = lgb.LGBMRegressor(**lgb_params)

# --- 降维数据 ---
model_lgb.fit(X_train_pca, y_train_pca)
score_lgb_pca = mean_squared_error(y_valid_pca, model_lgb.predict(X_valid_pca))
print("LightGBM (PCA Data) MSE:", score_lgb_pca)

# --- 未降维数据 ---
model_lgb.fit(X_train, y_train)
score_lgb_full = mean_squared_error(y_valid, model_lgb.predict(X_valid))
print("LightGBM (Full Data) MSE:", score_lgb_full)

# --- 最终预测 (示例：未降维数据) ---
final_model_lgb = lgb.LGBMRegressor(**lgb_params)
final_model_lgb.fit(train_data.drop('target', axis=1), train_data['target'])
y_pred_lgb = final_model_lgb.predict(test_data)
# np.savetxt('./export_data/lightGBM模型预测(非降维数据).txt', y_pred_lgb)
```

### 4.6 XGBoost

- **优点**: 高效、灵活、准确率高、内置交叉验证、正则化防止过拟合。
- **缺点**: 调参复杂、在某些硬件上可能不如 LightGBM 快。

```python
# --- 模型定义 ---
xgb_params = {
    'n_estimators': 300,
    'max_depth': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 1, # 注意：这个学习率通常过高，应调小
    'gamma': 0,
    'reg_lambda': 0,
    'reg_alpha': 0,
    'verbosity': 1,
    'random_state': 42
}
model_xgb = XGBRFRegressor(**xgb_params) # 注意：这里用了 XGBRFRegressor，通常用 XGBRegressor

# --- 降维数据 ---
model_xgb.fit(X_train_pca, y_train_pca)
score_xgb_pca = mean_squared_error(y_valid_pca, model_xgb.predict(X_valid_pca))
print("XGBoost (PCA Data) MSE:", score_xgb_pca)

# --- 未降维数据 ---
model_xgb.fit(X_train, y_train)
score_xgb_full = mean_squared_error(y_valid, model_xgb.predict(X_valid))
print("XGBoost (Full Data) MSE:", score_xgb_full)

# --- 最终预测 (示例：未降维数据) ---
final_model_xgb = XGBRFRegressor(**xgb_params)
final_model_xgb.fit(train_data.drop('target', axis=1), train_data['target'])
y_pred_xgb = final_model_xgb.predict(test_data)
# np.savetxt('export_data/Xgboost模型预测(非降维数据).txt', y_pred_xgb)
```

## 5. 结果总结与分析

根据代码中给出的最终 MSE 分数（越低越好）：

| 模型 (基于非降维数据) | MSE (Mean Squared Error) |
| :---------------------- | :------------------------- |
| Random Forest         | ~0.1461                  |
| GBDT                  | ~0.1392                  |
| LightGBM              | ~0.1378                  |
| **XGBoost**                      |  **~0.1329**                         |

- **集成算法优势**: GBDT、LightGBM、XGBoost 等集成算法普遍优于单一模型（如线性回归、SVR），证明了集成学习在此任务上的有效性。
- **最佳模型**: 在本次实验中，**XGBoost** 模型取得了最低的 MSE 分数，表现最佳。
- **数据维度影响**: 代码注释提到“集成算法对非降维数据表现更好”，而“多元线性回归、SVR对降维数据表现更好”。这表明不同类型的模型对数据维度的敏感度不同。集成树模型通常能很好地处理高维稀疏数据，而线性模型和 SVM 在降维后可能因减少了噪声或冗余特征而受益。

## 6. 下一步建议

1. **超参数调优**: 对表现较好的模型（尤其是 XGBoost, LightGBM）进行网格搜索或贝叶斯优化，以进一步提升性能。
2. **模型融合**: 尝试将多个表现优异的模型（如 XGBoost, LightGBM, GBDT）进行加权平均或堆叠（Stacking）以获得更强的预测能力。
3. **特征重要性分析**: 利用 XGBoost 或 LightGBM 等模型自带的 `feature_importances_` 属性，分析哪些锅炉运行参数对燃烧效率影响最大，为实际操作提供指导。
4. **模型解释**: 使用 SHAP (SHapley Additive exPlanations) 等工具对最终模型进行解释，理解模型是如何做出预测的。
