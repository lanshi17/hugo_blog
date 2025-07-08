---
# 核心元数据
author: lanshi
date: "2025-07-08T23:11:16+08:00"
lastmod:
title: 给AI模型添加记忆

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: 本文介绍了如何给AI模型添加记忆，包括手动存储记忆、自动带记忆的对话链以及不同类型的记忆（如ConversationBufferMemory、ConversationBufferWindowMemory、ConversationSummaryMemory、ConversationSummaryBufferMemory和ConversationTokenBufferMemory）的详细对比和使用示例。

# 内容分类
series:
tags: ["AI", "Python", "LangChain", "ChatOpenAI", "记忆"]
categories: ["编程"]

# SEO优化
description: 本文介绍了如何给AI模型添加记忆，包括手动存储记忆、自动带记忆的对话链以及不同类型的记忆（如ConversationBufferMemory、ConversationBufferWindowMemory、ConversationSummaryMemory、ConversationSummaryBufferMemory和ConversationTokenBufferMemory）的详细对比和使用示例。
keywords: ["AI", "Python", "LangChain", "ChatOpenAI", "记忆", "对话链", "ConversationBufferMemory", "ConversationBufferWindowMemory", "ConversationSummaryMemory", "ConversationSummaryBufferMemory", "ConversationTokenBufferMemory"]

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false


# 版权声明
copyright: true
---

## 手动存储记忆

```Python
import os
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate

# 设置你的OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# 创建一个ConversationBufferMemory实例，并设置return_messages=True
memory = ConversationBufferMemory(return_messages=True)

# 初始化ChatOpenAI模型
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")

# 定义一个ChatPromptTemplate，包含对话历史和当前用户输入
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手，可以帮助用户回答问题和进行对话。"),
    ("history", "{history}"),
    ("human", "{input}"),
])

# 函数来生成回复
def generate_response(user_input):
    # 获取对话历史
    history = memory.load_memory_variables({})["history"]
    
    # 创建完整的提示
    prompt = prompt_template.format_prompt(history=history, input=HumanMessage(content=user_input)).to_messages()
    
    # 生成AI回复
    ai_response = chat_model(prompt)
    
    # 提取AI回复的内容
    ai_content = ai_response.content
    
    # 保存当前对话到内存
    memory.save_context({"input": HumanMessage(content=user_input)}, {"output": AIMessage(content=ai_content)})
    
    return ai_content

# 示例对话
user_input_1 = "你好！"
response_1 = generate_response(user_input_1)
print(f"AI: {response_1}")

user_input_2 = "你今天过得怎么样？"
response_2 = generate_response(user_input_2)
print(f"AI: {response_2}")

user_input_3 = "我很好，谢谢！你呢？"
response_3 = generate_response(user_input_3)
print(f"AI: {response_3}")


```

## 自动带记忆的对话链

```Python
import os
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# 设置你的OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# 创建一个ConversationBufferMemory实例，并设置return_messages=True
memory = ConversationBufferMemory(return_messages=True)

# 初始化ChatOpenAI模型
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")

# 创建一个ConversationChain实例
conversation_chain = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=True  # 设置为True以便查看详细的对话过程
)

# 示例对话
user_input_1 = {"input": "你好！"}
response_1 = conversation_chain.invoke(user_input_1)
print(f"AI: {response_1['response']}")

user_input_2 = {"input": "你今天过得怎么样？"}
response_2 = conversation_chain.invoke(user_input_2)
print(f"AI: {response_2['response']}")

user_input_3 = {"input": "我很好，谢谢！你呢？"}
response_3 = conversation_chain.invoke(user_input_3)
print(f"AI: {response_3['response']}")


```

## 不同类型的记忆

- 当然，我们可以详细对比不同类型的内存，包括它们的区别、用法、优劣和使用场景。以下是表格形式的对比：

  | 记忆类型                            | 描述                                                         | 用法                                                         | 优劣                                                         | 使用场景                                                     |
  | ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | **ConversationBufferMemory**        | 历史对话全盘记忆                                             | `memory = ConversationBufferMemory()`                        | - **优点**: 保留完整对话历史，便于上下文理解。<br>- **缺点**: 消耗大量 tokens，可能导致成本增加。 | 需要完整上下文的历史记录，且对话长度较短的情况。<br>适用于短对话或需要完整对话记录的应用。 |
  | **ConversationBufferWindowMemory**  | 只记忆 k 轮对话的历史对话                                    | `memory = ConversationBufferWindowMemory(k=5)`               | - **优点**: 控制对话长度，减少 token 消耗。<br>- **缺点**: 可能丢失部分上下文信息。 | 对话长度较长，但不需要完整历史记录的情况。<br>适用于长时间对话，但不需要保持全部历史记录的应用。 |
  | **ConversationSummaryMemory**       | 将历史对话进行总结再存储（需要大模型辅助总结）               | `memory = ConversationSummaryMemory(llm=llm)`                | - **优点**: 保留关键信息，减少 token 消耗。<br>- **缺点**: 需要大模型进行总结，可能引入误差。 | 需要保留关键信息，但对话长度较长的情况。<br>适用于需要保留关键上下文但对话较长的应用。 |
  | **ConversationSummaryBufferMemory** | 参数限制 tokens 上限，超过阈值时，将进行历史总结（需要大模型辅助） | `memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4096)` | - **优点**: 控制 token 消耗，同时保留关键信息。<br>- **缺点**: 需要大模型进行总结，可能引入误差。 | 对话长度较长，且需要控制 token 消耗的情况。<br>适用于长时间对话且需要控制成本的应用。 |
  | **ConversationTokenBufferMemory**   | 只记忆总 tokens 上限对话的历史对话                           | `memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=4096)` | - **优点**: 控制总 token 消耗。<br>- **缺点**: 可能丢失部分上下文信息。 | 需要控制总 token 消耗，但不需要完整历史记录的情况。<br>适用于需要严格控制 token 消耗的应用。 |

  ### 详细说明

  #### ConversationBufferMemory

  - **描述**: 记录完整的对话历史。
  - **用法**:

    ```python
    from langchain.memory import ConversationBufferMemory
    
    memory = ConversationBufferMemory()
    ```

  - **优劣**:
    - **优点**: 保留完整对话历史，便于上下文理解。
    - **缺点**: 消耗大量 tokens，可能导致成本增加。
  - **使用场景**: 需要完整上下文的历史记录，且对话长度较短的情况。适用于短对话或需要完整对话记录的应用。

  #### ConversationBufferWindowMemory

  - **描述**: 只记忆最近 k 轮对话的历史对话。
  - **用法**:

    ```python
    from langchain.memory import ConversationBufferWindowMemory
    
    memory = ConversationBufferWindowMemory(k=5)
    ```

  - **优劣**:
    - **优点**: 控制对话长度，减少 token 消耗。
    - **缺点**: 可能丢失部分上下文信息。
  - **使用场景**: 对话长度较长，但不需要完整历史记录的情况。适用于长时间对话，但不需要保持全部历史记录的应用。

  #### ConversationSummaryMemory

  - **描述**: 将历史对话进行总结再存储（需要大模型辅助总结）。
  - **用法**:

    ```python
    from langchain.memory import ConversationSummaryMemory
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    memory = ConversationSummaryMemory(llm=llm)
    ```

  - **优劣**:
    - **优点**: 保留关键信息，减少 token 消耗。
    - **缺点**: 需要大模型进行总结，可能引入误差。
  - **使用场景**: 需要保留关键信息，但对话长度较长的情况。适用于需要保留关键上下文但对话较长的应用。

  #### ConversationSummaryBufferMemory

  - **描述**: 参数限制 tokens 上限，超过阈值时，将进行历史总结（需要大模型辅助）。
  - **用法**:

    ```python
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4096)
    ```

  - **优劣**:
    - **优点**: 控制 token 消耗，同时保留关键信息。
    - **缺点**: 需要大模型进行总结，可能引入误差。
  - **使用场景**: 对话长度较长，且需要控制 token 消耗的情况。适用于长时间对话且需要控制成本的应用。

  #### ConversationTokenBufferMemory

  - **描述**: 只记忆总 tokens 上限对话的历史对话。
  - **用法**:

    ```python
    from langchain.memory import ConversationTokenBufferMemory
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=4096)
    ```

  - **优劣**:
    - **优点**: 控制总 token 消耗。
    - **缺点**: 可能丢失部分上下文信息。
  - **使用场景**: 需要控制总 token 消耗，但不需要完整历史记录的情况。适用于需要严格控制 token 消耗的应用。

  ### 示例代码

  以下是一些示例代码，展示了如何使用不同的内存类型：

  #### ConversationBufferMemory 示例

  ```python
  import os
  from langchain.memory import ConversationBufferMemory
  from langchain.chat_models import ChatOpenAI
  from langchain.chains import ConversationChain
  
  # 设置你的OpenAI API密钥
  os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
  
  # 创建一个ConversationBufferMemory实例
  memory = ConversationBufferMemory()
  
  # 初始化ChatOpenAI模型
  chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
  
  # 创建一个ConversationChain实例
  conversation_chain = ConversationChain(
      llm=chat_model,
      memory=memory,
      verbose=True
  )
  
  # 示例对话
  user_input_1 = "你好！"
  response_1 = conversation_chain.run(user_input_1)
  print(f"AI: {response_1}")
  
  user_input_2 = "你今天过得怎么样？"
  response_2 = conversation_chain.run(user_input_2)
  print(f"AI: {response_2}")
  
  user_input_3 = "我很好，谢谢！你呢？"
  response_3 = conversation_chain.run(user_input_3)
  print(f"AI: {response_3}")
  ```

  #### ConversationBufferWindowMemory 示例

  ```python
  import os
  from langchain.memory import ConversationBufferWindowMemory
  from langchain.chat_models import ChatOpenAI
  from langchain.chains import ConversationChain
  
  # 设置你的OpenAI API密钥
  os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
  
  # 创建一个ConversationBufferWindowMemory实例，只记忆最近5轮对话
  memory = ConversationBufferWindowMemory(k=5)
  
  # 初始化ChatOpenAI模型
  chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
  
  # 创建一个ConversationChain实例
  conversation_chain = ConversationChain(
      llm=chat_model,
      memory=memory,
      verbose=True
  )
  
  # 示例对话
  user_input_1 = "你好！"
  response_1 = conversation_chain.run(user_input_1)
  print(f"AI: {response_1}")
  
  user_input_2 = "你今天过得怎么样？"
  response_2 = conversation_chain.run(user_input_2)
  print(f"AI: {response_2}")
  
  user_input_3 = "我很好，谢谢！你呢？"
  response_3 = conversation_chain.run(user_input_3)
  print(f"AI: {response_3}")
  ```

  #### ConversationSummaryMemory 示例

  ```python
  import os
  from langchain.memory import ConversationSummaryMemory
  from langchain.chat_models import ChatOpenAI
  from langchain.chains import ConversationChain
  
  # 设置你的OpenAI API密钥
  os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
  
  # 初始化ChatOpenAI模型
  chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
  
  # 创建一个ConversationSummaryMemory实例
  memory = ConversationSummaryMemory(llm=chat_model)
  
  # 创建一个ConversationChain实例
  conversation_chain = ConversationChain(
      llm=chat_model,
      memory=memory,
      verbose=True
  )
  
  # 示例对话
  user_input_1 = "你好！"
  response_1 = conversation_chain.run(user_input_1)
  print(f"AI: {response_1}")
  
  user_input_2 = "你今天过得怎么样？"
  response_2 = conversation_chain.run(user_input_2)
  print(f"AI: {response_2}")
  
  user_input_3 = "我很好，谢谢！你呢？"
  response_3 = conversation_chain.run(user_input_3)
  print(f"AI: {response_3}")
  ```

  #### ConversationSummaryBufferMemory 示例

  ```python
  import os
  from langchain.memory import ConversationSummaryBufferMemory
  from langchain.chat_models import ChatOpenAI
  from langchain.chains import ConversationChain
  
  # 设置你的OpenAI API密钥
  os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
  
  # 初始化ChatOpenAI模型
  chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
  
  # 创建一个ConversationSummaryBufferMemory实例，设置最大token限制为4096
  memory = ConversationSummaryBufferMemory(llm=chat_model, max_token_limit=4096)
  
  # 创建一个ConversationChain实例
  conversation_chain = ConversationChain(
      llm=chat_model,
      memory=memory,
      verbose=True
  )
  
  # 示例对话
  user_input_1 = "你好！"
  response_1 = conversation_chain.run(user_input_1)
  print(f"AI: {response_1}")
  
  user_input_2 = "你今天过得怎么样？"
  response_2 = conversation_chain.run(user_input_2)
  print(f"AI: {response_2}")
  
  user_input_3 = "我很好，谢谢！你呢？"
  response_3 = conversation_chain.run(user_input_3)
  print(f"AI: {response_3}")
  ```

  #### ConversationTokenBufferMemory 示例

  ```python
  import os
  from langchain.memory import ConversationTokenBufferMemory
  from langchain.chat_models import ChatOpenAI
  from langchain.chains import ConversationChain
  
  # 设置你的OpenAI API密钥
  os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
  
  # 初始化ChatOpenAI模型
  chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
  
  # 创建一个ConversationTokenBufferMemory实例，设置最大token限制为4096
  memory = ConversationTokenBufferMemory(llm=chat_model, max_token_limit=4096)
  
  # 创建一个ConversationChain实例
  conversation_chain = ConversationChain(
      llm=chat_model,
      memory=memory,
      verbose=True
  )
  
  # 示例对话
  user_input_1 = "你好！"
  response_1 = conversation_chain.run(user_input_1)
  print(f"AI: {response_1}")
  
  user_input_2 = "你今天过得怎么样？"
  response_2 = conversation_chain.run(user_input_2)
  print(f"AI: {response_2}")
  
  user_input_3 = "我很好，谢谢！你呢？"
  response_3 = conversation_chain.run(user_input_3)
  print(f"AI: {response_3}")
  ```
