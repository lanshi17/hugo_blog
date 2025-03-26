---
# 核心元数据
author: lanshi
date: "2025-03-20T18:30:00+08:00"
lastmod:
title: "西瓜书学习笔记(一)--绪论"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: 本文将介绍机器学习的基本概念包括样本、示例、训练集、测试集等,并讨论机器学习的发展历程和现状,同时引用了“没有免费午餐”定理来讨论归纳偏好问题。

# 内容分类
series:
tags: ["机器学习"]
categories: ["机器学习", "AI学习"]

# SEO优化
description: "机器学习的基本概念包括样本、示例、训练集、测试集等，对于机器学习的发展历程和现状进行分析，并引用了“没有免费午餐”定理来讨论归纳偏好问题。"
keywords: ["机器学习", "样本", "示例", "训练集", "测试集", "没有免费午餐定理", "归纳偏好"]

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "database-cover.png"
  alt: "机器学习的基本概念与发展历程"
  caption: "机器学习的基本概念与发展历程"
  relative: true

# 版权声明
copyright: true
---

## 引言
略

## 基本术语
1. 样本/示例 - sample/instance  
2. 训练集 - training data  
3. 测试集 - testing data  
4. 标记 - label  
5. 样例 - example: 拥有label的instance  
6. 泛化 - generalization  

## 假设空间

### 科学推理
- **归纳** - induction  
  从具体的事实归结出一般性规律  
- **演绎** - deduction  
  从基础原理推演出具体状况  

### 归纳学习 - inductive learning
- **广义归纳学习**
- **狭义归纳学习** - 概念学习  
  例如：布尔概念学习  

#### 版本空间 - version space
即存在着一个与训练集一致的“假设集合”

## 归纳偏好
有多个与训练集一致的假设，但测试新样本时有不同的输出结果，那么采用哪种模型（假设）？

### “奥卡姆剃刀”原则
若有多个假设与观察一致，则选最简单的那个。  
利用什么原则，取决于算法能否获得更好的性能，泛化能力是否更强  

### NFL(No Free Lunch Theorem)定理 - "没有免费的午餐"定理

#### 定理 [No Free Lunch 定理]  
对于所有学习算法 $\mathcal{L}_a$ 和 $\mathcal{L}_b$，在均匀分布的目标函数空间下，它们的训练外误差满足：
$$
\sum_f E_{ote}(\mathcal{L}_a|X,f) = \sum_f E_{ote}(\mathcal{L}_b|X,f)
$$  

#### 证明  
1. **定义与假设**  
   假设样本空间 $\mathcal{X}$ 和假设空间 $\mathcal{H}$ 是离散的。定义：
   $$
   E_{ote}(\mathcal{L}_a|X,f) = \sum_{h \in \mathcal{H}} \sum_{x \in \mathcal{X}\setminus X} P(x) \cdot \mathbb{I}(h(x) \neq f(x)) \cdot P(h|X,\mathcal{L}_a)
   $$
   其中 $\mathbb{I}(\cdot)$ 为指示函数。

2. **总误差求和**  
   对所有目标函数求和：
   $$
   \sum_f E_{ote}(\mathcal{L}_a|X,f) = \sum_f \sum_h \sum_{x \in \mathcal{X}\setminus X} P(x)\mathbb{I}(h(x)\neq f(x))P(h|X,\mathcal{L}_a)
   $$

3. **交换求和顺序**  
   将 $\sum_f$ 移至内部：
   $$
   = \sum_{x \in \mathcal{X}\setminus X} P(x) \sum_h P(h|X,\mathcal{L}_a) \nonumber \\
   \quad \times \underbrace{\sum_f \mathbb{I}(h(x)\neq f(x))}_{\text{\highlight{\text{关键项}}}}
   $$

4. **计算关键项**  
   对于二分类问题，每个 $x$ 处的 $f(x)$ 有等概率取 0 或 1：
   $$
   \sum_f \mathbb{I}(h(x)\neq f(x)) = \frac{1}{2} \cdot 2^{|\mathcal{X}|} = 2^{|\mathcal{X}|-1}
   $$

5. **最终化简**  
   代入关键项并利用 $\sum_h P(h|X,\mathcal{L}_a) = 1$：
   $$
   \text{原式} = 2^{|\mathcal{X}|-1} \sum_{x \in \mathcal{X}\setminus X} P(x)
   $$
   该结果与算法 $\mathcal{L}_a$ 无关，故对任意 $\mathcal{L}_a, \mathcal{L}_b$：
   $$
   \sum_f E_{ote}(\mathcal{L}_a|X,f) = \sum_f E_{ote}(\mathcal{L}_b|X,f)
   $$

由上述定理可知，脱离具体问题，空谈"什么学习算法最好"是毫无意义的。

## 发展历程
推理期: 1950s-1970s -- 符号知识, 演绎推理  

知识期: 1970s中期 -- 符号知识, 领域知识  

学习期: 1980s -- 机器学习, 归纳逻辑程序设计(Inductive Logic Programming)  

统计学习: 1990s中期 -- 向量机(Support Vector Machine), 核方法(Kernel Methods)  

深度学习: 2000s -- 神经网络  

## 应用现状
信息科学, 自然科学\dots  

## 阅读材料
### 机器学习
- 国际会议: ICML, NIPS，COLT, ECML(Europe), ACML(Asia)  
- 国际期刊: JMLR, ML  

### 人工智能
- 国际会议: IJCAI, AAAI  
- 国际期刊: AI, JAIR  

### 数据挖掘
- 国际会议: KDD, ICDM  
- 国际期刊: ACM-TKDD, DMKD  

### 计算机视觉
- CVPR(会议), IEEE-TPAMI(期刊)  

### 神经网络
- 期刊: NC, IEEE-TNNLS  

### 统计学
- 期刊: AS  

## 习题
