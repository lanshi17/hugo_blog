---
# 核心元数据
author: lanshi
date: "2025-07-04T23:40:24+08:00"
lastmod:
title: AI模型与输入输出

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: 本文介绍了如何使用AI模型进行基本的输入输出操作，包括基本使用方法、模板化输入、小样本示例模版化、从输出中提取列表和JSON，以及LCEL（LangChain表达式语言）的使用。

# 内容分类
series:
tags: ["AI", "Python", "LangChain", "OpenAI"]
categories: ["编程"]

# SEO优化
description: 本文介绍了如何使用AI模型进行基本的输入输出操作，包括基本使用方法、模板化输入、小样本示例模版化、从输出中提取列表和JSON，以及LCEL（LangChain表达式语言）的使用。
keywords: ["AI", "Python", "LangChain", "OpenAI", "输入输出", "模板化输入", "小样本示例", "提取列表", "提取JSON", "LCEL"]

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 版权声明
copyright: true
---
## 基本使用方法

- 示例

  ```python
  import os
  from langchain_openai import ChatOpenAI
  from IPython.display import display, Markdown
  from pydantic import SecretStr
  from langchain.schema.messages import SystemMessage, HumanMessage
  from dotenv import load_dotenv
  
  # 加载 .env 文件以获取环境变量
  load_dotenv()
  
  # 初始化 ChatOpenAI 模型，使用从环境变量中获取的 API 密钥和其他参数
  model = ChatOpenAI(
      model="qwen-turbo",
      api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
      base_url=os.getenv("BASE_URL"),
      temperature=0.3,
      frequency_penalty=1.5
  )
  
  # 定义对话消息列表，包含系统消息和人类消息
  messages = [
      SystemMessage(content="请你作为我的物理课助教,用通俗易懂的语言解释物理概念.使用markdown形式"),
      HumanMessage(content="什么是波粒二象性?")
  ]
  
  # 使用模型对对话消息列表进行处理，并获取响应
  response = model.invoke(messages)
  
  # 使用 Markdown 格式显示响应内容
  display(Markdown(response.content))
  ```
  
## 模板化输入

- 示例

  ```python
  import os
  from langchain_openai import ChatOpenAI
  from langchain.prompts import SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
  from IPython.display import display, display_markdown
  from pydantic import SecretStr
  from dotenv import load_dotenv
  
  # 加载 .env 文件以获取环境变量
  load_dotenv()
  
  # 定义系统消息模板，用于设置翻译任务和语言风格
  system_template_text = "你是一位专业翻译者,能够将{input_language}翻译成{output_language},并且输出文本会根据用户要求的任何语言风格进行调整.请只输出翻译后的文本,不要输出额外内容."
  system_prompt_template = SystemMessagePromptTemplate.from_template(system_template_text)
  
  # 显示系统消息模板及其输入变量
  display(system_prompt_template)
  display(system_prompt_template.input_variables)
  
  # 定义人类消息模板，用于提供需要翻译的文本和语言风格
  human_template_text = "文本:{text}\n语言风格:{style}"
  human_prompt_template = HumanMessagePromptTemplate.from_template(human_template_text)
  
  # 使用系统消息模板生成具体的系统消息，指定输入和输出语言
  system_prompt = system_prompt_template.format(input_language="英语", output_language="汉语")
  
  # 使用人类消息模板生成具体的人类消息，指定需要翻译的文本和语言风格
  human_prompt = human_prompt_template.format(text="I'm miss you !", style="诗词")
  
  # 初始化 ChatOpenAI 模型，使用从环境变量中获取的 API 密钥和其他参数
  model = ChatOpenAI(
      model="qwen-turbo",
      api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
      base_url=os.getenv("BASE_URL"),
      temperature=0.3,
      frequency_penalty=1.5
  )
  
  # 使用模型对生成的系统消息和人类消息进行处理，并获取响应
  response = model.invoke([
      system_prompt,
      human_prompt
  ])
  display_markdown(response.content, raw=True)
  
  # 创建一个聊天提示模板，结合系统消息和人类消息模板
  prompt_template = ChatPromptTemplate.from_messages(
      [
          ("system", f"{system_prompt_template.prompt.template}"),
          ("human", f"{human_prompt_template.prompt.template}")
      ]
  )
  
  # 使用聊天提示模板生成具体的提示值，指定输入和输出语言、文本以及语言风格
  prompt_value = prompt_template.invoke(
      {
          "input_language": "English",
          "output_language": "汉语",
          "text": "I'm do love you",
          "style": "诗词"
      }
  )
  
  # 使用模型对生成的提示值进行处理，并获取响应
  response = model.invoke(prompt_value)
  display_markdown(response.content, raw=True)
  
  ```
  
## 小样本示例模版化

- 示例

  ```python
  import os
  from langchain_openai import ChatOpenAI
  from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
  from IPython.display import display, display_markdown
  from pydantic import SecretStr
  from langchain.schema.messages import SystemMessage, HumanMessage
  from dotenv import load_dotenv
  
  # 加载 .env 文件以获取环境变量
  load_dotenv()
  
  # 初始化 ChatOpenAI 模型，使用从环境变量中获取的 API 密钥和其他参数
  model = ChatOpenAI(
      model="qwen-turbo",
      api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
      base_url=os.getenv("BASE_URL"),
      temperature=0.3,
      frequency_penalty=1.5
  )
  
  # 定义示例提示模板，用于展示如何格式化客户信息
  example_prompt = ChatPromptTemplate.from_messages(
      [
          ("human", "格式化以下客户信息:\n姓名 -> {customer_name}\n年龄 -> {customer_age}\n城市 -> {customer_city}"),
          ("ai", "##客户信息\n- 客户姓名: {formatted_name}\n- 客户年龄: {formatted_age}\n- 客户所在地: {formatted_city}")
      ]
  )
  
  # 定义一些示例数据，展示如何应用格式化规则
  examples = [
      {
          "customer_name": "张三",
          "customer_age": "27",
          "customer_city": "长沙",
          "formatted_name": "张三",
          "formatted_age": "27岁",
          "formatted_city": "湖南省长沙市"
      },
      {
          "customer_name": "李四",
          "customer_age": "42",
          "customer_city": "广州",
          "formatted_name": "李四",
          "formatted_age": "42岁",
          "formatted_city": "广东省广州市"
      },
      {
          "customer_name": "王五",
          "customer_age": "35",
          "customer_city": "长沙",
          "formatted_name": "王五",
          "formatted_age": "35岁",
          "formatted_city": "湖南省长沙市"
      }
  ]
  
  # 使用示例提示和数据创建一个 FewShotChatMessagePromptTemplate 对象
  few_shot_template = FewShotChatMessagePromptTemplate(
      example_prompt=example_prompt,
      examples=examples,
  )
  
  # 创建最终的提示模板，结合 FewShot 模板和用户输入
  final_prompt_template = ChatPromptTemplate.from_messages(
      [
          few_shot_template,
          ("human", "{input}"),
      ]
  )
  
  # 使用最终的提示模板生成具体的提示，输入为需要格式化的客户信息
  final_prompt = final_prompt_template.invoke(
      {
          "input": "格式化以下客户信息:\n姓名 -> 刘六\n年龄 -> 28\n城市 -> 南通"
      }
  )
  final_prompt.to_messages()
  
  # 使用预设的模型对生成的提示进行处理，并获取响应
  response = model.invoke(final_prompt)
  display_markdown(response.content, raw=True)
  ```

## 从输出中提取列表

- 示例

  ```python
  import os
  from langchain_openai import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate
  from langchain.output_parsers import CommaSeparatedListOutputParser
  from IPython.display import display, display_markdown
  from pydantic import SecretStr
  from dotenv import load_dotenv
  
  # 加载 .env 文件以获取环境变量
  load_dotenv()
  
  # 初始化 ChatOpenAI 模型，使用从环境变量中获取的 API 密钥和其他参数
  model = ChatOpenAI(
      model="qwen-turbo",
      api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
      base_url=os.getenv("BASE_URL"),
      temperature=0.3,
      frequency_penalty=1.5
  )
  
  # 定义聊天提示模板，包含系统消息和人类消息
  prompt = ChatPromptTemplate.from_messages([
      ("system", "{parser_instructions}"),
      ("human", "列出5个{subject}色系的hex编码")
  ])
  
  # 创建一个逗号分隔的列表输出解析器，并获取格式说明
  output_parser = CommaSeparatedListOutputParser()
  parser_instructions = output_parser.get_format_instructions()
  display(parser_instructions)
  
  # 使用提示模板生成具体的提示，指定主题和格式说明
  final_prompt = prompt.invoke(
      {
          "subject": "莫兰迪",
          "parser_instructions": parser_instructions
      }
  )
  
  # 使用模型对生成的提示进行处理，并获取响应
  response = model.invoke(final_prompt)
  display_markdown(response.content, raw=True)
  
  # 使用输出解析器解析响应内容
  parsed_response = output_parser.invoke(response.content)
  
  # 显示解析后的响应类型
  type(parsed_response)
  ```

## 从输出中提取json

- 示例

  ```python
  import os
  from langchain_openai import ChatOpenAI
  from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
  from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
  from pydantic import BaseModel, Field
  from typing import List
  from IPython.display import display, display_markdown, display_json
  from pydantic import SecretStr
  from dotenv import load_dotenv
  
  # 加载 .env 文件以获取环境变量
  load_dotenv()
  
  # 初始化 ChatOpenAI 模型，使用从环境变量中获取的 API 密钥和其他参数
  model = ChatOpenAI(
      model="qwen-turbo",
      api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
      base_url=os.getenv("BASE_URL"),
      temperature=0.3,
      frequency_penalty=1.5
  )
  
  # 定义 BookInfo 类，用于存储书籍信息
  class BookInfo(BaseModel):
      book_name: str = Field(
          description="书籍的名字",
          example="百年孤独"
      )
      author_name: str = Field(
          description="书籍的作者",
          example="加西亚-马克克斯"
      )
      genres: List[str] = Field(  # 注意这里应该是 genres 而不是 generes
          description="书籍的体裁",
          example=["小说", "文学"]
      )
  
  # 创建 Pydantic 输出解析器，并获取格式说明
  output_parser = PydanticOutputParser(pydantic_object=BookInfo)
  parser_instructions = output_parser.get_format_instructions()
  display(output_parser.get_format_instructions())
  
  # 定义聊天提示模板，包含系统消息和人类消息
  prompt = ChatPromptTemplate.from_messages([
      ("system", "{parser_instructions} 你输出的结果请使用中文"),
      ("human", "请你帮我从书籍概述中,提取书名,作者,以及书籍的体裁.书籍概述会被三个#符号包围\n###{book_introduction}###")
  ])
  
  # 定义书籍概述
  book_introduction = """
  《明朝那些事儿》百科介绍
  基本信息
  书名：明朝那些事儿
  作者：当年明月（本名石悦）
  出版时间：2006年首次出版
  类别：历史小说、通俗历史读物
  篇幅：共九部，总计数百万字
  内容简介
  《明朝那些事儿》是一部以明朝历史为背景的长篇历史小说。它从1344年开始，一直讲述到1644年明朝灭亡，涵盖了将近三百年的时间跨度。书中以明朝十六位皇帝为核心，同时描绘了众多王公权贵、文臣武将以及普通小人物的命运，展现了明朝社会的复杂面貌。
  
  该书以严谨的史料为基础，结合生动的小说笔法，用幽默风趣的语言讲述了明朝的政治、军事、经济和文化等方面的历史事件。尤其对官场斗争、战争场面、帝王心术等内容着墨较多，使读者能够更加直观地理解历史人物的心理活动与时代背景。
  
  创作特点
  语言风格：轻松幽默，打破传统历史书籍的枯燥感，让历史变得有趣。
  叙事方式：以年代为主线，结合具体人物故事，穿插历史事件，形成全景式的历史画卷。
  史料基础：作者通过大量查阅正史资料（如《明史》《二十四史》等），确保内容在趣味性之外兼具一定的历史准确性。
  受众广泛：适合对历史感兴趣的大众读者，尤其是年轻群体，作为了解明朝历史的入门读物。
  社会影响
  《明朝那些事儿》自问世以来，广受读者欢迎，被誉为中国现代通俗历史文学的经典之作。
  它不仅在国内畅销多年，还被翻译成多种语言，在海外也拥有大量读者。
  该书的成功推动了“草根历史”写作潮流，激发了许多人对历史学习的兴趣。
  作者简介
  笔名：当年明月
  原名：石悦
  职业：作家、公务员
  代表作品：《明朝那些事儿》是其最著名的作品，也是他以业余时间创作的成果。
  相关评价
  正面评价：许多读者称赞该书将枯燥的历史变得生动有趣，易于理解，是一本“让人笑着学历史”的好书。
  批评观点：也有部分学者认为书中部分内容存在过度演绎或主观色彩较强的问题，建议将其视为“历史小说”而非完全意义上的学术著作。
  出版形式
  纸质书籍：分为多卷出版，涵盖整个明朝历史。
  电子书与有声书：由于市场需求大，该书也被制作成电子书及有声读物，方便不同人群阅读/收听。
  改编作品：已被改编为电视剧、广播剧等多种形式。
  总结
  《明朝那些事儿》以其独特的写作风格和深入浅出的历史解读方式，成为当代中国最受欢迎的历史类畅销书之一。它不仅让广大读者爱上历史，也为历史普及工作做出了重要贡献。
  """
  
  # 使用提示模板生成具体的提示，指定书籍概述和格式说明
  final_prompt = prompt.invoke({
      "book_introduction": book_introduction,
      "parser_instructions": parser_instructions
  })
  
  # 使用模型对生成的提示进行处理，并获取响应
  response = model.invoke(final_prompt)
  display_json(response.content, raw=True)
  
  # 使用输出解析器解析响应内容
  result = output_parser.invoke(response)
  display(result)
  
  ```

## LCEL(langchain表达式语言)

- 示例

  ```python
  import os
  from langchain_openai import ChatOpenAI  # 导入 ChatOpenAI 类，用于与 OpenAI 的对话模型交互
  from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate  # 导入聊天消息提示模板类
  from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser  # 导入输出解析器类
  from pydantic import BaseModel, Field  # 导入 Pydantic 的 BaseModel 和 Field 类
  from typing import List  # 导入 List 类型注解
  from IPython.display import display, display_markdown, display_json  # 导入 IPython 的显示函数
  from pydantic import SecretStr  # 导入 Pydantic 的 SecretStr 类，用于处理敏感信息
  from dotenv import load_dotenv  # 导入 load_dotenv 函数，用于加载 .env 文件中的环境变量
  
  # 加载 .env 文件以获取环境变量
  load_dotenv()
  
  # 初始化 ChatOpenAI 模型，使用从环境变量中获取的 API 密钥和其他参数
  model = ChatOpenAI(
      model="qwen-turbo",  # 指定使用的模型名称
      api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),  # 使用 SecretStr 包装 API 密钥以提高安全性
      base_url=os.getenv("BASE_URL"),  # 设置 API 的基础 URL
      temperature=0.3,  # 设置生成文本的随机性程度
      frequency_penalty=1.5  # 设置重复惩罚因子
  )
  
  # 定义聊天提示模板，包含系统消息和人类消息
  prompt = ChatPromptTemplate.from_messages([
      ("system", "{parser_instructions}"),  # 系统消息部分，将使用输出解析器的格式说明
      ("human", "列出5个{subject}色系的hex编码")  # 人类消息部分，用户请求的内容
  ])
  
  # 创建逗号分隔列表输出解析器，并获取格式说明
  output_parser = CommaSeparatedListOutputParser()
  parser_instructions = output_parser.get_format_instructions()
  
  # 使用管道操作符连接提示模板、模型和输出解析器，并调用 invoke 方法
  # 这一步会将提示、模型和解析器组合在一起，然后执行模型调用
  # 最终输出的结果是根据提示生成的文本经过解析器解析后的结果
  result = (prompt | model | output_parser).invoke(
      {"subject": "莫兰迪",  # 替换 {subject} 为具体的色系名称
       "parser_instructions": parser_instructions}  # 将格式说明传递给系统消息
  )
  
  # 打印或展示结果（如果是在 Jupyter Notebook 中运行）
  print(result)
  ```
