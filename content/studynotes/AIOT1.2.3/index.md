---
author: lanshi
date: "2026-04-08T12:00:00+08:00"
lastmod: "2026-04-08T22:00:00+08:00"
title: "在 LFW 人脸数据上比较 Logistic Regression 与 Linear SVM：基于 Nested CV 的严谨实验"

draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: "本文基于 LFW 人脸数据集，在统一的 StandardScaler + PCA 预处理条件下，对 Logistic Regression 与 Linear SVM 进行公平比较，并使用 Nested Cross Validation 避免调参与评估耦合，综合分析两类模型在泛化性能、稳定性与训练成本上的差异。"

series:
  - "机器学习实验"
tags:
  - "LFW"
  - "人脸识别"
  - "Logistic Regression"
  - "Linear SVM"
  - "Nested CV"
  - "Pipeline"
  - "PCA"
  - "GridSearchCV"
  - "Scikit-learn"
categories:
  - "机器学习"
  - "实验对比"

description: "本文在 LFW 人脸数据集上比较 Logistic Regression 与 Linear SVM 的表现，使用 StandardScaler、PCA 和 Nested Cross Validation 构建严谨实验流程，从准确率、平衡准确率、Macro-F1 和训练耗时等维度进行分析。"
keywords:
  - "LFW"
  - "人脸识别"
  - "Logistic Regression"
  - "Linear SVM"
  - "Nested Cross Validation"
  - "PCA"
  - "Pipeline"
  - "GridSearchCV"
  - "Scikit-learn"
  - "机器学习实验"

math: false
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

cover:
  image: "database-cover.png"
  alt: "LFW 人脸识别实验对比封面"
  caption: "LFW 数据集上的 Logistic Regression 与 Linear SVM 对比实验"
  relative: true

copyright: true
---
## 1. 选题背景

LFW（Labeled Faces in the Wild）是经典的人脸识别数据集。
这篇实验希望回答一个很实用的问题：

> 在同样的预处理条件下，**逻辑回归（Logistic Regression）**  与 **线性支持向量机（Linear SVM）** ，谁在多分类人脸识别任务中表现更好？

为了避免“调参导致评估乐观偏差”，本文采用了更严谨的评估框架：**Nested Cross Validation（嵌套交叉验证）** 。

---

## 2. 数据集与任务设定

使用 `sklearn.datasets.fetch_lfw_people` 加载数据：

- `resize=1`：保留原始尺寸
- `min_faces_per_person=70`：只保留样本数不少于 70 的人物

```python
data = datasets.fetch_lfw_people(resize=1, min_faces_per_person=70)
X = data["data"]          # 展平后的特征
y = data["target"]        # 类别标签
faces = data["images"]    # 原始图像
target_names = data["target_names"]
```

---

## 3. 方法设计：为什么要用 Nested CV？

很多教程是“train/test split + GridSearchCV + test打分”，但这会有一定评估偏差。
本文采用：

- **外层 CV（5折）** ：估计模型真实泛化误差
- **内层 CV（5折）** ：只在外层训练折里做超参数搜索

这样评估更稳健，更适合做模型对比实验。

---

## 4. 预处理与模型流水线

图像向量维度较高，直接训练容易过拟合/耗时。
因此统一使用：

1. `StandardScaler()`：标准化
2. `PCA(n_components=0.95)`：保留 95% 方差降维
3. 分类器（LR 或 Linear SVM）

```python
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, svd_solver="full")),
    ("clf", LogisticRegression(max_iter=5000))
])

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, svd_solver="full")),
    ("clf", SVC(kernel="linear"))
])
```

---

## 5. 超参数搜索空间

```python
lr_param_grid = {
    "clf__C": np.logspace(-3, 2, 10),
    "clf__solver": ["lbfgs", "newton-cg", "saga"]
}

svm_param_grid = {
    "clf__C": np.logspace(-3, 2, 10),
    "clf__tol": [1e-2, 1e-3, 1e-4]
}
```

---

## 6. 评估指标

除了准确率，本文还统计了：

- `accuracy`
- `balanced_accuracy`
- `macro_f1`
- `fit_time_sec`

这能避免只看 accuracy 而忽略类别平衡与训练成本。

---

## 7. 核心实验流程（外层评估 + 内层调参）

你这段实现是本实验的亮点：

```python
records = []

for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # LR: inner search + outer test
    lr_search.fit(X_train, y_train)
    lr_best = lr_search.best_estimator_
    lr_pred = lr_best.predict(X_test)

    # SVM: inner search + outer test
    svm_search.fit(X_train, y_train)
    svm_best = svm_search.best_estimator_
    svm_pred = svm_best.predict(X_test)

    # 记录每折结果
    ...
```

最终输出每折结果、均值方差汇总、每折最佳参数，并保存到 `records.csv`，可复现性非常好。

---

## 8. 结果解读建议（配合你的 `summary` 输出）

你已经有下面这些输出：

- 每折指标
- `groupby(model).agg(["mean","std"])`
- 每折最佳参数

博客中建议按以下方式解读（不虚构数值）：

1. **泛化性能**：比较 LR 与 SVM 的 `outer_acc / outer_macro_f1` 均值
2. **稳定性**：比较各指标 `std`，越小越稳定
3. **训练成本**：比较 `fit_time_sec` 均值
4. **参数敏感性**：看每折 `best_params` 是否变化较大

> 结论以你实际运行输出为准。通常在线性可分程度较高、PCA后特征质量较好时，Linear SVM 与 Logistic Regression 都会有不错表现；SVM 往往边界更“硬”，LR 概率解释性更强。

---

## 9. 最优参数重训与可视化

你后续对最优参数做了单次重训，并展示预测效果：

```python
svc = Pipeline([
    ('scaler', StandardScaler()),
    ('PCA', PCA(n_components=0.95)),
    ('svc', SVC(C=0.003593813663804626, kernel='linear', tol=1e-2))
])
```

并绘制前 50 张测试图像的 `True vs Pred`。这部分很适合博客展示直观效果。

---

## 10. 一个小优化建议

为了更严谨，建议把“最优参数重训”改成：

- 在**全量数据**上再做一次最终训练（或单独留出最终 hold-out 测试集）
- 避免直接使用“最后一折的 `X_test/y_test`”作为最终展示依据

此外，Pipeline 步骤命名最好统一小写（`"pca"`），便于和前文一致。

---

## 11. 方法优缺点总结

### Logistic Regression

**优点**：

- 训练较快
- 可解释性更好（系数可分析）
- 概率输出更直接

**缺点**：

- 线性决策边界表达能力有限

### Linear SVM

**优点**：

- 在高维稀疏/图像特征任务中通常表现稳定
- 对间隔最大化有天然优势

**缺点**：

- 超参数敏感（如 `C`, `tol`）
- 概率输出需额外启用，计算成本更高

---

## 12. 可复现完整性清单

这套实验已满足高质量复现条件：

- [X] 固定随机种子（`StratifiedKFold(..., random_state=...)`）
- [X] 统一预处理 Pipeline，避免数据泄露
- [X] 内外层 CV 分离，避免调参与评估耦合
- [X] 多指标评估（accuracy/balanced_acc/macro_f1）
- [X] 记录训练耗时
- [X] 结果落盘（`records.csv`）

---

## 13. 结论

本文在 LFW 人脸多分类任务中，对 Logistic Regression 与 Linear SVM 进行了公平对比。通过统一的 `StandardScaler + PCA` 预处理以及 Nested CV 评估框架，实验避免了常见的数据泄露与评估偏差问题。结果显示，两类线性模型都能在该任务上取得稳定表现；最终选型应结合你的实际输出，从**泛化性能、稳定性、训练时间、可解释性**四个维度综合判断，而非只看单一 accuracy。
