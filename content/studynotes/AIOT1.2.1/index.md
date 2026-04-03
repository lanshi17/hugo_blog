---
# 核心元数据
author: lanshi
date: "2026-04-02T12:00:00+08:00"
lastmod: "2025-04-02T22:00:00+08:00"
title: "通过逻辑斯蒂回归实现手写数字多分类（0~9）"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: "从数据读取、标准化、逻辑斯蒂回归建模到混淆矩阵、错误样本分析、Softmax 概率推导与 GridSearchCV 调参，系统完成手写数字 0~9 多分类实战。"

# 内容分类
series:
  - 机器学习实战
tags:
  - 逻辑斯蒂回归
  - LogisticRegression
  - 多分类
  - 手写数字识别
  - sklearn
  - StandardScaler
  - 混淆矩阵
  - GridSearchCV
categories:
  - 机器学习

# SEO优化
description: "本文基于 sklearn 的 LogisticRegression 实现手写数字 0~9 多分类任务，包含数据处理、标准化、模型训练、分类评估、混淆矩阵、错误样本分析、Softmax 概率推导与超参数调优。"
keywords:
  - 逻辑斯蒂回归
  - LogisticRegression
  - 手写数字识别
  - 多分类
  - sklearn
  - StandardScaler
  - Softmax
  - 混淆矩阵
  - GridSearchCV
  - digits.csv

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "logistic-regression-digits-cover.png"
  alt: "逻辑斯蒂回归手写数字多分类封面"
  caption: "逻辑斯蒂回归实现 0~9 手写数字识别"
  relative: true

# 版权声明
copyright: true
---
# 一、项目目标

本实验使用 **逻辑斯蒂回归（Logistic Regression）** 完成手写数字图片的多分类任务，类别为 **0~9**。
数据集保存在 `digits.csv` 中，使用 `pandas` 加载，借助 `scikit-learn` 完成：

- 数据读取与分析
- 训练集 / 测试集划分
- 标准化 + 逻辑斯蒂回归建模
- 模型评估
- 混淆矩阵可视化
- 错误样本分析
- 模型保存
- 手动推导概率计算
- 网格搜索调参

---

# 二、逻辑斯蒂回归为什么能做多分类？

很多同学一看到“逻辑斯蒂回归”，第一反应是：
> 这不是二分类模型吗？

其实在 `sklearn` 中，逻辑斯蒂回归不仅可以做二分类，也可以做多分类。

对于本题的 10 个类别（0~9），模型本质上学习的是：

- 每个类别对应一组参数 `W_k` 和偏置 `b_k`
- 对输入样本先计算每一类的线性得分
- 再通过 **Softmax** 转成概率分布
- 最后取概率最大的类别作为预测结果

数学形式如下：

## 1. 线性打分

对于一个样本 `x`：


$$
z = xW^T + b
$$


其中：

- `x`：输入特征
- `W`：模型权重
- `b`：偏置
- `z`：每个类别的得分

## 2. Softmax 概率


$$
P(y=k|x)=\frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$


这样就能把每个类别的得分转换成概率。

---

# 三、导入依赖库

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
```

---

# 四、日志配置

为了更方便地记录实验过程，可以使用 `loguru` 输出日志。

```python
logger.add(
    "logs/digits_logistic.log",
    rotation="1 MB",
    encoding="utf-8",
    level="INFO"
)
```

说明：

- `rotation="1 MB"`：日志文件达到 1MB 后自动切分
- `encoding="utf-8"`：避免中文乱码
- `level="INFO"`：记录 `INFO` 及以上级别日志

---

# 五、读取数据集

```python
raw_data = pd.read_csv("data/digits.csv")

logger.info("数据形状: {}", raw_data.shape)
display(raw_data.head())
```

这里默认：

- `label` 列为标签
- 其余列为像素特征

---

# 六、数据初步分析

```python
logger.info("列信息:\n{}", raw_data.info())
logger.info("缺失值统计:\n{}", raw_data.isnull().sum().head())
logger.info("标签分布:\n{}", raw_data["label"].value_counts().sort_index())
```

这里重点关注三件事：

## 1. 数据形状

查看样本数与特征维度。

## 2. 是否有缺失值

如果有缺失值，需要先补全或删除，否则影响建模。

## 3. 标签是否平衡

如果某些数字样本特别少，可能会影响模型对该类别的识别能力。
本实验通过查看 `label` 分布确认各类别情况。

---

# 七、划分特征和标签

```python
X = raw_data.iloc[:, 1:]   # 假设第一列是 label 后面的像素列
y = raw_data["label"]

logger.info("X shape: {}", X.shape)
logger.info("y shape: {}", y.shape)
```

这里：

- `X`：像素特征
- `y`：数字标签（0~9）

> 注意：这段代码写法和注释略有出入。
> 如果 `label` 列确实叫 `"label"`，更稳妥的写法应是：

```python
X = raw_data.drop(columns=["label"])
y = raw_data["label"]
```

这样不会因为列顺序变化而出错。

---

# 八、划分训练集与测试集

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

logger.info("X_train: {}", X_train.shape)
logger.info("X_test: {}", X_test.shape)
logger.info("y_train: {}", y_train.shape)
logger.info("y_test: {}", y_test.shape)
```

参数说明：

- `test_size=0.2`：测试集占 20%
- `random_state=42`：保证实验可复现
- `stratify=y`：按标签分层抽样，保证训练集和测试集类别比例一致

> 在分类任务中，`stratify=y` 是一个非常重要的细节，尤其是类别不均衡时。

---

# 九、可视化一个训练样本

在训练前，先看看数据到底长什么样。

```python
sample_idx = 1
plt.figure(figsize=(4, 4))
plt.imshow(X_train.iloc[sample_idx].values.reshape(28, 28), cmap="gray")
plt.title(f"label = {y_train.iloc[sample_idx]}")
plt.axis("off")
plt.show()
```

说明：

- 每个样本有 `28 × 28 = 784` 个像素特征
- 将一维向量 `reshape(28, 28)` 后即可还原为灰度图像

---

# 十、构建机器学习流水线 Pipeline

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=42
    ))
])
```

这一步非常关键。

## 为什么要用 `Pipeline`？

因为完整训练流程通常包含两步：

1. **标准化**
2. **模型训练**

`Pipeline` 的优点：

- 避免数据泄漏
- 流程统一
- 便于调参
- 保存模型更方便

## 为什么要做标准化？

逻辑斯蒂回归对特征尺度比较敏感。
像素特征虽然都在相同区间，但标准化后通常仍能帮助优化器更快收敛。

## 为什么设置 `max_iter=5000`？

多分类任务、特征维度较高时，默认迭代次数可能不够，容易出现不收敛警告。
增大到 `5000` 更稳妥。

---

# 十一、训练模型并预测

```python
# 训练模型
pipeline.fit(X_train, y_train)

# 预测类别
y_pred = pipeline.predict(X_test)

# 预测概率
y_prob = pipeline.predict_proba(X_test)

logger.info("预测类别前10个: {}", y_pred[:10])
logger.info("预测概率前3个样本:\n{}", y_prob[:3])
```

这里得到两个重要结果：

- `y_pred`：预测类别
- `y_prob`：每个样本对 10 个类别的预测概率

---

# 十二、模型评估

```python
acc = accuracy_score(y_test, y_pred)
logger.info("测试集准确率: {:.6f}", acc)
print("accuracy =", acc)

print("\n分类报告：")
print(classification_report(y_test, y_pred))
```

---

## 1. 准确率 Accuracy

$$
Accuracy = \frac{\text{预测正确样本数}}{\text{总样本数}}
$$

它表示整体分类效果。

---

## 2. 分类报告 `classification_report`

分类报告中常见指标：

- **precision（精确率）**
- **recall（召回率）**
- **f1-score**
- **support（样本数）**

对于每个类别，报告都能告诉我们模型识别得如何。

### 这些指标怎么理解？

#### 精确率 Precision

预测成某类的样本中，有多少是真的该类。

#### 召回率 Recall

真实属于某类的样本中，有多少被成功识别出来。

#### F1-score

精确率与召回率的调和平均，更适合综合衡量模型性能。

---

# 十三、混淆矩阵分析

```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.show()
```

## 混淆矩阵怎么看？

- **行**：真实类别
- **列**：预测类别
- 对角线：预测正确
- 非对角线：预测错误

如果某些数字经常被混淆，比如：

- 3 和 5
- 4 和 9
- 7 和 1

就会在矩阵相应位置看到较大的值。

> 混淆矩阵是分析分类模型错误模式的利器。

---

# 十四、查看预测错误的样本

```python
wrong_idx = np.where(y_pred != y_test.values)[0]
logger.info("预测错误样本数: {}", len(wrong_idx))

if len(wrong_idx) > 0:
    n_show = min(9, len(wrong_idx))
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(wrong_idx[:n_show]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test.iloc[idx].values.reshape(28, 28), cmap="gray")
        plt.title(f"true={y_test.iloc[idx]}, pred={y_pred[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
```

这一步很有价值，因为它能帮助我们理解：

- 是模型能力不够？
- 还是样本本身写得太模糊？
- 哪些数字更容易“长得像”？

## 错误样本分析建议

当看到错误图片时，可以重点观察：

1. **笔画是否模糊**
2. **数字是否倾斜**
3. **某些类别之间形态是否接近**
4. **数据是否存在噪声或标注问题**

---

# 十五、保存训练好的模型

```python
joblib.dump(pipeline, "models/digits_logistic_pipeline.pkl")
logger.info("模型已保存到 models/digits_logistic_pipeline.pkl")
```

模型保存后，后续可以直接加载使用，无需重复训练。

## 加载模型示例

```python
model = joblib.load("models/digits_logistic_pipeline.pkl")
pred = model.predict(X_test)
```

---

# 十六、手动计算类别概率

这是本实验最“学霸”的部分：
我们不只会调用 `predict_proba()`，还要自己把概率算出来。

---

## 1. 取出标准化器与模型参数

```python
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

# 标准化后的测试集
X_test_scaled = scaler.transform(X_test)

# 取出参数
W = model.coef_         # shape: (10, 784)
b = model.intercept_    # shape: (10,)

print("W shape:", W.shape)
print("b shape:", b.shape)
```

### 参数含义

- `W`：每个类别对应一组权重，共 10 组，每组 784 维
- `b`：每个类别一个偏置，共 10 个

---

## 2. 计算线性得分

```python
z = X_test_scaled @ W.T + b   # shape: (n_samples, 10)
```

这一步对应数学公式：

$$
z = xW^T + b
$$

每个样本会得到 10 个得分。

---

## 3. 手写稳定版 Softmax

```python
def softmax_stable(z):
    z = np.asarray(z)
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

manual_prob = softmax_stable(z)
```

为什么要写成“稳定版”？

因为如果 `z` 中某些值很大，直接 `np.exp(z)` 容易数值溢出。
因此先减去每行最大值 `z_max`，不会改变 Softmax 结果，但能提升数值稳定性。

---

## 4. 与 sklearn 概率对比

```python
sklearn_prob = pipeline.predict_proba(X_test)

print("手动计算概率前3行：")
print(manual_prob[:3])

print("\nsklearn predict_proba 前3行：")
print(sklearn_prob[:3])

print("\n两者最大绝对误差：", np.max(np.abs(manual_prob - sklearn_prob)))
```

如果一切正确，两者误差应非常小，接近浮点计算误差范围。

这说明：

- `predict_proba()` 的底层逻辑，本质上就是：

  - 线性得分
  - 再做 Softmax

---

# 十七、查看某个样本的类别概率分布

```python
sample_id = 0
plt.figure(figsize=(4, 4))
plt.imshow(X_test.iloc[sample_id].values.reshape(28, 28), cmap="gray")
plt.title(f"True={y_test.iloc[sample_id]}, Pred={y_pred[sample_id]}")
plt.axis("off")
plt.show()

prob_df = pd.DataFrame({
    "class": np.arange(10),
    "probability": y_prob[sample_id]
}).sort_values("probability", ascending=False)

display(prob_df)
```

这一步的意义在于：

- 不只看最终预测类别
- 还看模型“有多自信”

例如：

| class | probability |
| ------- | ------------- |
| 8     | 0.92        |
| 3     | 0.05        |
| 9     | 0.02        |

说明模型认为该样本大概率属于 8。

如果前两名概率很接近，例如：

- 4：0.41
- 9：0.39

说明样本具有较强迷惑性。

---

# 十八、查看模型参数规模

```python
logger.info("coef_ shape: {}", model.coef_.shape)
logger.info("intercept_ shape: {}", model.intercept_.shape)

print("coef_ shape:", model.coef_.shape)
print("intercept_ shape:", model.intercept_.shape)
```

输出通常为：

- `coef_ shape: (10, 784)`
- `intercept_ shape: (10,)`

解释：

- 10 个类别
- 每个类别一组长度为 784 的权重向量
- 每类一个偏置

这也说明，逻辑斯蒂回归虽然简单，但参数量并不算特别小。

---

# 十九、超参数调优：GridSearchCV

逻辑斯蒂回归中一个重要超参数是 `C`。

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("best params:", grid.best_params_)
print("best score:", grid.best_score_)
```

---

## 1. `C` 是什么？

`C` 是正则化强度的倒数。

- `C` 越小：正则化越强，模型更简单，防止过拟合
- `C` 越大：正则化越弱，模型更容易拟合训练数据

---

## 2. 为什么要做交叉验证？

`cv=5` 表示 5 折交叉验证，能更稳定地评估参数好坏，避免只看某一次划分的偶然结果。

---

## 3. 为什么参数名前面要写 `model__`？

因为这里调的是 `Pipeline` 中 `model` 这一步的参数。
语法规则是：

```python
步骤名__参数名
```

所以这里写成：

```python
"model__C"
```

---

# 二十、保存最优模型

```python
best_model = grid.best_estimator_
import os
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_digits_logistic.pkl")
```

这里保存的是经过网格搜索后得到的最佳模型，通常比初始模型更值得部署使用。

---

# 二十一、完整实验流程总结

整个项目可以概括为下面 9 步：

1. **读取数据**
2. **数据检查（形状、缺失值、标签分布）**
3. **划分训练集 / 测试集**
4. **可视化样本**
5. **构建 Pipeline：标准化 + 逻辑斯蒂回归**
6. **训练模型并评估**
7. **分析混淆矩阵与错误样本**
8. **手动验证 Softmax 概率**
9. **调参与模型保存**

---

# 二十二、本实验的核心知识点

## 1. 为什么逻辑斯蒂回归适合做这个任务？

虽然它是经典线性模型，但对于像素分类这种基础任务依然很有效，尤其适合作为：

- 入门基线模型
- 可解释模型
- 训练速度快的对照模型

---

## 2. 为什么标准化很重要？

标准化能帮助优化器更稳定地训练，减少不同特征量纲对模型造成的影响。

---

## 3. 为什么要分析错误样本？

准确率只是一个总指标，真正决定你对模型理解深度的，是：

- 哪些样本错了
- 为什么错
- 有没有规律

---

## 4. 为什么要手动算概率？

因为这能帮助你真正理解：

- 逻辑斯蒂回归不是“黑盒”
- `predict_proba()` 不是魔法
- 多分类的本质就是 **线性打分 + Softmax**

---

# 二十三、代码中可进一步优化的细节

下面是几个值得改进的小点。

---

## 优化 1：更稳妥地选择特征列

原写法：

```python
X = raw_data.iloc[:, 1:]
y = raw_data["label"]
```

更推荐：

```python
X = raw_data.drop(columns=["label"])
y = raw_data["label"]
```

原因：

- 不依赖列顺序
- 可读性更好
- 不容易因数据表结构变化而出错

---

## 优化 2：训练前确保目录存在

例如日志目录、模型目录应提前创建：

```python
import os
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
```

否则在某些环境下可能报路径不存在错误。

---

## 优化 3：增加收敛信息检查

训练逻辑斯蒂回归时，如果迭代次数不足，可能会出现收敛警告。
可以：

- 提高 `max_iter`
- 调整 `solver`
- 检查数据是否标准化

---

## 优化 4：增加测试集最优模型评估

网格搜索得到 `best_model` 后，最好补一段：

```python
best_pred = best_model.predict(X_test)
print("test accuracy:", accuracy_score(y_test, best_pred))
```

这样能更直观看出调参后的提升效果。

---

# 二十四、适合考试 / 面试的高频问答

## Q1：逻辑斯蒂回归是回归还是分类？

它名字里虽然有“回归”，但主要用于**分类任务**。

---

## Q2：多分类逻辑斯蒂回归的输出是什么？

输出每个类别的概率分布，通常通过 **Softmax** 获得。

---

## Q3：为什么 `predict_proba()` 的概率之和等于 1？

因为 Softmax 会把所有类别得分归一化成一个概率分布。

---

## Q4：为什么要使用 `Pipeline`？

为了把预处理和模型训练串联起来，避免数据泄漏，并方便保存与调参。

---

## Q5：`C` 变大代表什么？

代表正则化减弱，模型更容易拟合训练数据，也更可能过拟合。

---

# 二十五、一份可直接复现的核心代码模板

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from loguru import logger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logger.add(
    "logs/digits_logistic.log",
    rotation="1 MB",
    encoding="utf-8",
    level="INFO"
)

# 读取数据
raw_data = pd.read_csv("data/digits.csv")

# 特征与标签
X = raw_data.drop(columns=["label"])
y = raw_data["label"]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 建立流水线
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=42
    ))
])

# 训练
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

# 评估
acc = accuracy_score(y_test, y_pred)
print("accuracy =", acc)
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.show()

# 保存模型
joblib.dump(pipeline, "models/digits_logistic_pipeline.pkl")

# 网格搜索调参
param_grid = {
    "model__C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("best params:", grid.best_params_)
print("best score:", grid.best_score_)

best_model = grid.best_estimator_
joblib.dump(best_model, "models/best_digits_logistic.pkl")
```

---

# 二十六、最终结论

本实验说明：

- **逻辑斯蒂回归不仅能做二分类，也能高效完成 0~9 手写数字多分类**
- 使用 `Pipeline` 将 **标准化 + 模型训练** 串联，是更规范的机器学习实践
- 通过 **混淆矩阵** 和 **错误样本分析**，可以更深入理解模型表现
- 通过手动实现 **Softmax 概率计算**，可以彻底搞懂 `predict_proba()` 的原理
- 使用 `GridSearchCV` 对 `C` 调参，可以进一步提升模型效果

---

# 二十七、一句话记忆版

> **手写数字多分类 = 像素特征输入 → 标准化 → 逻辑斯蒂回归线性打分 → Softmax 输出 10 类概率 → 取最大概率类别。**
