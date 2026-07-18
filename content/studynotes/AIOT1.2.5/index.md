---
# 核心元数据
author: lanshi
date: "2026-04-28T16:37:52+08:00"
lastmod: "2026-04-28T16:37:52+08:00"
title: "结合 PCA 降维与 LangGraph 智能体的机器学习自动化建模实践"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: "以手写数字分类任务为例，对比逻辑回归与 SVM 在使用和不使用 PCA 降维时的效果，并结合 LangGraph 构建能够自动规划、执行和汇总机器学习建模任务的智能体工作流。"

# 内容分类
series:
  - "机器学习自动化实践"
tags:
  - "机器学习"
  - "PCA"
  - "LangGraph"
  - "LangChain"
  - "SVM"
  - "逻辑回归"
  - "AutoML"
  - "智能体"
categories:
  - "机器学习"
  - "AI应用"
  - "实战教程"

# SEO优化
description: "本文基于手写数字分类任务，系统对比逻辑回归与 SVM 在 PCA 降维前后的性能差异，并结合 LangGraph 构建一个可通过自然语言自动调度模型训练与结果汇总的机器学习智能体工作流。"
keywords:
  - "PCA降维"
  - "LangGraph"
  - "LangChain"
  - "机器学习自动化建模"
  - "SVM分类"
  - "逻辑回归"
  - "Pipeline"
  - "GridSearchCV"
  - "手写数字识别"
  - "AutoML"

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "database-cover.png"
  alt: "结合 PCA 与 LangGraph 的机器学习自动化建模实践"
  caption: "PCA 降维、分类模型与智能体调度工作流"
  relative: true

# 版权声明
copyright: true
---
在日常的机器学习实验中，我们经常需要对比不同模型、不同预处理手段（如降维）以及不同超参数对最终结果的影响。传统方法往往需要编写大量用于循环、网格搜索和结果解析的基础代码。

本文将以**手写数字集分类**任务为例，对比**逻辑回归（Logistic Regression）** 和**支持向量机（SVM）** 在使用和不使用 **PCA 降维** 下的性能表现。更重要的是，我们将引入大模型开发框架 **LangGraph**，构建一个能够理解人类自然语言指令、自动规划并调度执行这些机器学习任务的智能体（Agent）工作流。

---

## 1. 数据管理：面向对象的数据预处理

在处理数据时，为了提高代码的复用性和可读性，我们通常会将数据的加载、特征清洗与数据集拆分封装成独立的类。

```python
class Data:
    """数据加载、预处理与拆分。"""
    
    def __init__(self, file_path: Optional[Path | str]) -> None:
        self._data = self._load_data(file_path)
        self._X = self._data.drop(columns=["label"])
        self._y = self._data["label"]
        
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X, self._y, test_size=test_size, random_state=random_state, stratify=self._y
        )
```

通过这种方式，我们在后续向各种算法函数传递数据时，只需传递一个 `Data` 实例，代码结构更加清晰。

---

## 2. 核心建模：逻辑回归与 SVM 的实现

我们实现了两组核心函数：一组直接使用原始高维数据进行训练，另一组则使用 PCA 降维后的数据进行训练。

### 非降维建模

利用 `GridSearchCV` 和交叉验证对模型进行超参数搜索。

- **逻辑回归 (****`LogisticRegressionCV`** **)** ：支持遍历一系列 `C` 值和求解器（如 `lbfgs`, `newton-cg`），返回具有最高准确率的参数组合。
- **SVM (****`SVC`** **)** ：通过自定义超参数网格（`kernel`, `C`, `gamma`），寻找能够获得最高准确率的模型。

### 结合 PCA 的建模流

在高维空间（比如图像像素特征）下，SVM 和逻辑回归可能会遭遇“维度灾难”导致计算缓慢。我们将 `PCA` 与算法结合：

```python
pipeline = Pipeline([
    ("pca", PCA(n_components=0.95)), # 保留 95% 的方差
    ("svm", SVC(random_state=42)),
])
```

使用 `Pipeline` 能够有效防止数据泄露（Data Leakage），即在交叉验证的每一折中，PCA 都只在训练集上 `fit`，进而提升评估结果的客观性。

---

## 3. 引领范式转变：使用 LangGraph 构建 ML Agent

本文最核心的亮点在于：**我们不再手动调用这些训练函数，而是让大语言模型（LLM）基于 LangGraph 作为调度中枢，自动完成这项工作。**

通过将上述模型训练方法封装为大模型的环境**工具 (Tools)** ，我们能依靠大模型的思考能力来决定应该运行哪些训练、如何输入参数，并汇总结果。

### 3.1 状态与计划定义 (State & Plan)

通过 Pydantic 定义清晰的数据结构，约束 LLM 输出可解析的“计划”。

```python
class Action(BaseModel):
    action_type: Literal["get_logistic_regression", "get_svm"]
    arguments: dict

class Plan(BaseModel):
    plan_summary: str
    actions: List[Action]

class State(BaseModel):
    user_query: str
    plan: Optional[Plan] = None
    results: List[str] = Field(default_factory=list)
    messages: list
```

### 3.2 封装工具 (Tools)

大模型无法直接运行 Python 代码中的函数，因此需要使用 `@tool` 装饰器将刚才写好的逻辑回归和 SVM 执行函数转换为大模型可以识别的工具接口。

```python
@tool
def get_svm(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """选择 SVM 训练器，根据 enable_pca 决定是否先做 PCA 降维。"""
    # 提取参数...
    if enable_pca:
        return pca_gridcv_svm(data, **kwargs)
    return gridcv_svm(data, **kwargs)
```

### 3.3 构建控制流图 (StateGraph)

我们设计了一个经典的“规划 - 执行 - 响应”工作流：

1. **Planner Node**：LLM 接收到诸如“请帮我找出最佳算法及其最佳参数和准确率”的请求，输出一份包含确切行动指令（即是否调用 SVM 或 逻辑回归、是否调用 PCA 的 JSON 列表）的执行计划。
2. **Executor Node**：解析并实际执行这些计算密集型任务。由于底层是多进程模型训练，这一步可能极为耗时。Agent 通过 `ToolNode` 并发调用底层代码。
3. **Responder Node**：整合所有工具返回的准确率和延迟等结果，LLM 生成一份对人类可读的自然语言最终反馈。

```python
workflow = StateGraph(State)

workflow.add_node("planner", plan_node)
workflow.add_node("executor", execute_tools_node)
workflow.add_node("responder", respond_node)

workflow.add_edge(START, "planner")
workflow.add_conditional_edges(
    "planner",
    should_execute_tools, # 条件路由
    {"execute_tools": "executor", "respond": "responder"}
)
workflow.add_edge("executor", "responder")
workflow.add_edge("responder", END)
```

---

## 4. 结语

借助 Langchain / LangGraph，我们成功实现了**从“命令式编程实现调参”到“声明式自然语言调度”的转变**。开发者只需提供“探索最优降维及模型组合”的一句话需求，AI Agent 就能自动规划：

1. 测试启用 PCA 降维的 SVM 和逻辑回归；
2. 测试直接训练的 SVM 和逻辑回归；
3. 对比并汇报时间及准确率差异。

在数据科学和算法工程面临越来越广泛工程量挑战的今天，将传统的数据驱动算法操作与 Agent 驱动的调度工作流结合，将是大规模自动化机器学习（AutoML）甚至 AI 数据科学家的一个清晰的未来趋势。
