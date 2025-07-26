---
# 核心元数据
author: lanshi
date: "2025-07-26T18:06:26+08:00"
lastmod:
title: 给AI模型用工具的能力

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: 本文介绍了如何给AI模型赋予使用工具的能力，包括自定义工具、使用现成的AI工具运行代码和分析数据表格，以及多个工具组成AI工具箱。详细展示了baseTool、hub、create_structured_chat_agent和AgentExecutor的代码示例，并提供了具体的输出结果。

# 内容分类
series:
tags: ["AI", "Python", "LangChain", "ChatOpenAI", "工具", "数据分析", "代码执行"]
categories: ["编程"]

# SEO优化
description: 本文介绍了如何给AI模型赋予使用工具的能力，包括自定义工具、使用现成的AI工具运行代码和分析数据表格，以及多个工具组成AI工具箱。详细展示了baseTool、hub、create_structured_chat_agent和AgentExecutor的代码示例，并提供了具体的输出结果。
keywords: ["AI", "Python", "LangChain", "ChatOpenAI", "工具", "数据分析", "代码执行", "baseTool", "hub", "create_structured_chat_agent", "AgentExecutor", "PythonREPLTool"]

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "ai-tools-cover.png"
  alt: "给AI模型用工具的能力封面"
  caption: "给AI模型用工具的能力"
  relative: true

# 版权声明
copyright: true
---
## 自定义你的AI工具
- baseTool、hub、create_structured_chat_agent和AgentExecutor代码示例
    ```python
    from langchain.agents import AgentExecutor, create_structured_chat_agent
    from langchain.tools import BaseTool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.utilities import SerpAPIWrapper
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field, SecretStr
    from typing import Type, Any, List
    import os
    from dotenv import load_dotenv

    # 加载 .env 文件
    load_dotenv()

    # 定义一个自定义工具类（继承BaseTool）
    class CustomSearchTool(BaseTool):
        name: str = "custom_search"
        description: str = "用于搜索最新信息的工具"
        
        def _run(self, query: str) -> str:
            # 实现具体的搜索逻辑
            search = SerpAPIWrapper()
            return search.run(query)
        
        async def _arun(self, query: str) -> str:
            raise NotImplementedError("该工具暂不支持异步调用")

    # 定义工具输入模型（用于结构化输出）
    class SearchInput(BaseModel):
        query: str = Field(description="要搜索的问题")

    # 将工具与输入模型绑定
    custom_search_tool = CustomSearchTool()
    custom_search_tool.args_schema = SearchInput

    # 初始化通义千问大模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = ChatOpenAI(
        model="qwen-turbo",
        api_key=SecretStr(api_key) if api_key else None,
        base_url=os.getenv("BASE_URL")
    )

    # 构建结构化聊天代理提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个能够使用工具解决问题的助手。请根据用户问题选择合适的工具进行处理。"
                "你可以使用以下工具:\n{tools}\n工具名称: {tool_names}"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")  # 确保此占位符被正确处理
    ])

    # 创建结构化聊天代理
    agent = create_structured_chat_agent(
        llm=model,
        tools=[custom_search_tool],
        prompt=prompt
    )

    # 使用AgentExecutor包装代理以支持工具调用和迭代执行
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[custom_search_tool],
        handle_parsing_errors=True,
        verbose=True
    )

    # 示例：运行代理执行器
    if __name__ == "__main__":
        result = agent_executor.invoke({
            "input": "今天北京天气怎么样？",
            "agent_scratchpad": []  # 显式初始化为空列表
        })
        print(result)
    ```

## 用现成的AI工具运行代码
- LangChain——experimental和PythonREPLTool代码示例
    ```python
    from langchain_experimental.tools import PythonREPLTool
    from langchain.agents import AgentExecutor, create_structured_chat_agent
    from langchain import hub
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field, SecretStr
    import os
    from dotenv import load_dotenv

    # 加载环境变量
    load_dotenv()

    # 定义计算商品总价的工具输入模型
    class PriceCalculationInput(BaseModel):
        product_prices: list = Field(description="商品单价列表")
        quantities: list = Field(description="商品数量列表")

    # 初始化Python REPL工具，用于执行价格计算
    python_repl_tool = PythonREPLTool()
    python_repl_tool.name = "price_calculator"
    python_repl_tool.description = "用于计算商品总价的工具，接收商品单价和数量列表"

    # 初始化通义千问大模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = ChatOpenAI(
        model="qwen-turbo",
        api_key=SecretStr(api_key) if api_key else None,
        base_url=os.getenv("BASE_URL")
    )

    # 使用官方的结构化聊天代理提示模板
    prompt = hub.pull("hwchase17/structured-chat-agent")

    # 创建结构化聊天代理
    agent = create_structured_chat_agent(
        llm=model,
        tools=[python_repl_tool],
        prompt=prompt
    )

    # 使用AgentExecutor包装代理
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[python_repl_tool],
        handle_parsing_errors=True,
        verbose=True
    )

    # 示例：运行代理执行器计算商品总价
    if __name__ == "__main__":
        result = agent_executor.invoke({
            "input": "我买了3件商品，单价分别是100元、200元和150元，数量分别是2件、1件和3件，请计算总价。"
        })
        print(result)
    ```
    输出如下：

    > Entering new AgentExecutor chain...
    Action:
    ```
    {
    "action": "price_calculator",
    "action_input": {
        "query": "我买了3件商品，单价分别是100元、200元和150元，数量分别是2件、1件和3件，请计算总价。"
    }
    }
    ```Python REPL can execute arbitrary code. Use with caution.
    SyntaxError("invalid character '，' (U+FF0C)", ('<string>', 1, 8, '我买了3件商品，单价分别是100元、200元和150元，数量分别是2件、1件和3件，请计算总价。', 1, 8)){
    "action": "Final Answer",
    "action_input": "您购买的商品总价为：(100元 * 2) + (200元 * 1) + (150元 * 3) = 200元 + 200元 + 450元 = 850元。"
    }

    > Finished chain.
    {'input': '我买了3件商品，单价分别是100元、200元和150元，数量分别是2件、1件和3件，请计算总价。', 'output': '您购买的商品总价为：(100元 * 2) + (200元 * 1) + (150元 * 3) = 200元 + 200元 + 450元 = 850元。'}

## 用现成的AI工具分析数据表格
- 示例
    ```python
        """
    数据分析代理程序
    该程序使用通义千问大模型和Python REPL工具来分析销售数据。
    """
    from langchain import hub
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI
    from langchain_experimental.tools import PythonREPLTool
    from langchain.agents import AgentExecutor, create_structured_chat_agent
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from pydantic import SecretStr
    import os
    import sys
    from dotenv import load_dotenv
    import pandas as pd
    import io

    # 加载环境变量
    load_dotenv()

    # 初始化Python REPL工具用于数据分析
    python_repl_tool = PythonREPLTool()
    python_repl_tool.name = "python_repl"
    python_repl_tool.description = "执行Python代码进行数据分析。输入应该是有效的Python代码。"

    # 初始化通义千问大模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：未设置DASHSCOPE_API_KEY环境变量")
        sys.exit(1)

    model = ChatOpenAI(
        model="qwen-turbo",
        api_key=SecretStr(api_key) if api_key else None,
        base_url=os.getenv("BASE_URL")
    )

    # 使用标准的结构化聊天代理提示词模板
    prompt = hub.pull("hwchase17/structured-chat-agent")

    # 创建结构化聊天代理
    agent = create_structured_chat_agent(
        llm=model,
        tools=[python_repl_tool],
        prompt=prompt
    )

    # 使用AgentExecutor包装代理
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[python_repl_tool],
        handle_parsing_errors=True,
        verbose=True
    )

    # 示例CSV数据
    try:
        sample_csv_data = pd.read_csv("code/AI模型工具应用/data.csv")
    except FileNotFoundError:
        print("错误：找不到data.csv文件，请确保文件存在于当前目录中")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取data.csv文件时发生错误: {e}")
        sys.exit(1)

    if __name__ == "__main__":
        # 先让agent了解数据结构并进行分析
        query = f"""
        我有一个销售数据集，需要你帮我分析。请使用Python代码来处理。
        首先加载数据：
        ```python
        import pandas as pd
        df = pd.read_csv("code/AI模型工具应用/data.csv")
        print("数据形状:", df.shape)
        print("列名:", df.columns.tolist())
        print("前5行数据:")
        print(df.head())然后计算：
        总销售额（quantity * price 的总和）
        找出销售额最高的产品
        请用中文回复分析结果。 """
        
        # 调用代理执行器处理数据分析请求
        result = agent_executor.invoke({
            "input": query
        })
        
        # 打印分析结果
        print("\n=== 结果 ===")
        print(result)
    ```
    输出如下：

    > Entering new AgentExecutor chain...
    {
    "action": "python_repl",
    "action_input": "import pandas as pd\n\ndf = pd.read_csv(\"code/AI模型工具应用/data.csv\")\nprint(\"数据形状:\", df.shape)\nprint(\"列名:\", df.columns.tolist())\nprint(\"前5行数据:\")\nprint(df.head())\n\n# 计算总销售额\ntotal_sales = (df['quantity'] * df['price']).sum()\n\n# 找出销售额最高的产品\nmax_sales_product = df.loc[(df['quantity'] * df['price']).idxmax()]\n\n(total_sales, max_sales_product)"
    }Python REPL can execute arbitrary code. Use with caution.
    数据形状: (20, 10)
    列名: ['order_id', 'customer_id', 'product_id', 'product_name', 'category', 'quantity', 'price', 'order_date', 'ship_date', 'order_status']
    前5行数据:
    order_id  customer_id  product_id product_name category  quantity   price  order_date   ship_date order_status
    0      1001         5001        2001       无线机械键盘     电子产品         2  129.99  2023-07-01  2023-07-02          已完成
    1      1002         5002        2002     智能手环 Pro     电子产品         1   79.50  2023-07-01  2023-07-03          已发货
    2      1003         5003        2003       降噪蓝牙耳机     电子产品         1   59.99  2023-07-02         NaN        "待付款"
    3      1004         5001        2004    24英寸曲面显示器     电子产品         1  199.99  2023-07-02  2023-07-05          已完成
    4      1005         5004        2005      办公人体工学椅     家居用品         1  149.95  2023-07-03  2023-07-06          已取消
    {
    "action": "Final Answer",
    "action_input": "总销售额为: 1279.83 元。销售额最高的产品是: 无线机械键盘，其销售额为: 259.98 元。"
    }

    > Finished chain.

    === 结果 ===
    {'input': '\n    我有一个销售数据集，需要你帮我分析。请使用Python代码来处理。\n    首先加载数据：\n    ```python\n    import pandas as pd\n    df = pd.read_csv("code/AI模型工具应用/data.csv")\n    print("数据形状:", df.shape)\n    print("列名:", df.columns.tolist())\n    print("前5行数据:")\n    print(df.head())然后计算：\n    总销售额（quantity * price 的总和）\n    找出销售额最高的产品\n    请用中文回复分析结果。 ', 'output': '总销售额为: 1279.83 元。销售额最高的产品是: 无线机械键盘，其销售额为: 259.98 元。'}

## 多个工具组成AI工具箱
- 示例
    ```python
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI
    from langchain_experimental.tools import PythonREPLTool
    from langchain.agents import AgentExecutor, create_structured_chat_agent
    from langchain import hub
    from pydantic import SecretStr
    import os
    from dotenv import load_dotenv
    import pandas as pd
    import io

    # 加载环境变量
    load_dotenv()

    # 定义Python代码执行工具输入模型
    class PythonCodeInput(BaseModel):
        code: str = Field(description="要执行的Python代码")

    # 定义CSV分析工具输入模型
    class CSVAnalysisInput(BaseModel):
        csv_content: str = Field(description="CSV文件内容")
        query: str = Field(description="用户查询问题")

    # 定义文本计算工具输入模型
    class TextCalculationInput(BaseModel):
        expression: str = Field(description="数学表达式")

    # 初始化Python REPL工具用于代码执行
    python_code_tool = PythonREPLTool()
    python_code_tool.name = "python_executor"
    python_code_tool.description = "用于执行Python代码的工具，可以进行复杂计算和数据处理"

    # 初始化Python REPL工具用于CSV分析
    csv_analysis_tool = PythonREPLTool()
    csv_analysis_tool.name = "csv_analyzer"
    csv_analysis_tool.description = "用于分析CSV数据的工具，可以执行数据统计、计算等操作"

    # 初始化Python REPL工具用于文本计算
    text_calculation_tool = PythonREPLTool()
    text_calculation_tool.name = "text_calculator"
    text_calculation_tool.description = "用于计算数学表达式的工具，支持基本和复杂数学运算"

    # 初始化通义千问大模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = ChatOpenAI(
        model="qwen-turbo",
        api_key=SecretStr(api_key) if api_key else None,
        base_url=os.getenv("BASE_URL")
    )

    # 从Hub拉取结构化聊天代理提示词模板
    prompt = hub.pull("hwchase17/structured-chat-agent")

    # 创建结构化聊天代理
    agent = create_structured_chat_agent(
        llm=model,
        tools=[python_code_tool, csv_analysis_tool, text_calculation_tool],
        prompt=prompt
    )

    # 使用AgentExecutor包装代理
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[python_code_tool, csv_analysis_tool, text_calculation_tool],
        handle_parsing_errors=True,
        verbose=True
    )

    # 示例CSV数据
    sample_csv_data =pd.read_csv("code/AI模型工具应用/data.csv") 
    # 示例：运行代理执行器使用不同工具
    if __name__ == "__main__":
        # 测试CSV分析工具
        print("=== 测试CSV分析工具 ===")
        result1 = agent_executor.invoke({
            "input": f"""
            我有一个销售数据集，需要你帮我分析。请使用Python代码来处理。
            首先加载数据：
            ```python
            import pandas as pd
            df = pd.read_csv("code/AI模型工具应用/data.csv")
            print("数据形状:", df.shape)
            print("列名:", df.columns.tolist())
            print("前5行数据:")
            print(df.head())然后计算：
            总销售额（quantity * price 的总和）
            找出销售额最高的产品
            请用中文回复分析结果。 """
        })
        print(result1)
        
        print("\n=== 测试文本计算工具 ===")
        # 测试文本计算工具
        result2 = agent_executor.invoke({
            "input": "请计算 (150 * 3 + 200) / 5 - 30 的结果"
        })
        print(result2)
        
        print("\n=== 测试Python代码执行工具 ===")
        # 测试Python代码执行工具
        result3 = agent_executor.invoke({
            "input": "请用Python生成一个包含10个随机数的列表，并计算它们的平均值"
        })
        print(result3)
    ```
    输出如下：

    === 测试CSV分析工具 ===

    > Entering new AgentExecutor chain...
    {
    "action": "python_executor",
    "action_input": "import pandas as pd\n\ndf = pd.read_csv(\"code/AI模型工具应用/data.csv\")\nprint(\"数据形状:\", df.shape)\nprint(\"列名:\", df.columns.tolist())\nprint(\"前5行数据:\")\nprint(df.head())\n\n# 计算总销售额\ntotal_sales = (df['quantity'] * df['price']).sum()\n\n# 找出销售额最高的产品\nmax_sales_product = df.loc[(df['quantity'] * df['price']).idxmax()]\n\n(total_sales, max_sales_product)"
    }Python REPL can execute arbitrary code. Use with caution.
    数据形状: (20, 10)
    列名: ['order_id', 'customer_id', 'product_id', 'product_name', 'category', 'quantity', 'price', 'order_date', 'ship_date', 'order_status']
    前5行数据:
    order_id  customer_id  product_id product_name category  quantity   price  order_date   ship_date order_status
    0      1001         5001        2001       无线机械键盘     电子产品         2  129.99  2023-07-01  2023-07-02          已完成
    1      1002         5002        2002     智能手环 Pro     电子产品         1   79.50  2023-07-01  2023-07-03          已发货
    2      1003         5003        2003       降噪蓝牙耳机     电子产品         1   59.99  2023-07-02         NaN        "待付款"
    3      1004         5001        2004    24英寸曲面显示器     电子产品         1  199.99  2023-07-02  2023-07-05          已完成
    4      1005         5004        2005      办公人体工学椅     家居用品         1  149.95  2023-07-03  2023-07-06          已取消
    {
    "action": "Final Answer",
    "action_input": "数据形状为 (20, 10)，列名包括：order_id、customer_id、product_id、product_name、category、quantity、price、order_date、ship_date、order_status。前5行数据如上所示。\n\n总销售额为：3798.46 元。\n\n销售额最高的产品是：\norder_id                    1001\ncustomer_id                 5001\nproduct_id                  2001\nproduct_name           无线机械键盘\ncategory               电子产品\nquantity                        2\nprice                       129.99\norder_date             2023-07-01\nship_date              2023-07-02\norder_status             已完成\nName: 0, dtype: object"
    }

    > Finished chain.
    {'input': '\n        我有一个销售数据集，需要你帮我分析。请使用Python代码来处理。\n        首先加载数据：\n        ```python\n        import pandas as pd\n        df = pd.read_csv("code/AI模型工具应用/data.csv")\n        print("数据形状:", df.shape)\n        print("列名:", df.columns.tolist())\n        print("前5行数据:")\n        print(df.head())然后计算：\n        总销售额（quantity * price 的总和）\n        找出销售额最高的产品\n        请用中文回复分析结果。 ', 'output': '数据形状为 (20, 10)，列名包括：order_id、customer_id、product_id、product_name、category、quantity、price、order_date、ship_date、order_status。前5行数据如上所示。\n\n总销售额为：3798.46 元。\n\n销售额最高的产品是：\norder_id                    1001\ncustomer_id                 5001\nproduct_id                  2001\nproduct_name           无线机械键盘\ncategory               电子产品\nquantity                        2\nprice                       129.99\norder_date             2023-07-01\nship_date              2023-07-02\norder_status             已完成\nName: 0, dtype: object'}

    === 测试文本计算工具 ===


    > Entering new AgentExecutor chain...
    Action:
    ```
    {
    "action": "text_calculator",
    "action_input": "(150 * 3 + 200) / 5 - 30"
    }
    ```{
    "action": "Final Answer",
    "action_input": "计算结果为 80"
    }

    > Finished chain.
    {'input': '请计算 (150 * 3 + 200) / 5 - 30 的结果', 'output': '计算结果为 80'}

    === 测试Python代码执行工具 ===


    > Entering new AgentExecutor chain...
    Action:
    ```
    {
    "action": "python_executor",
    "action_input": "import random\nrandom_numbers = [random.uniform(0, 100) for _ in range(10)]\naverage = sum(random_numbers) / len(random_numbers)\nrandom_numbers, average"
    }
    ```{
    "action": "Final Answer",
    "action_input": "生成的随机数列表为：[70.3412562862942, 75.52908411412517, 57.66020188242748, 67.71097724877533, 63.10141440195013, 75.77007066814763, 62.61627979214029, 59.78708280212198, 78.30742645772916, 52.24122079792126]，它们的平均值为：66.90343934588929"
    }

    > Finished chain.
    {'input': '请用Python生成一个包含10个随机数的列表，并计算它们的平均值', 'output': '生成的随机数列表为：[70.3412562862942, 75.52908411412517, 57.66020188242748, 67.71097724877533, 63.10141440195013, 75.77007066814763, 62.61627979214029, 59.78708280212198, 78.30742645772916, 52.24122079792126]，它们的平均值为：66.90343934588929'}
