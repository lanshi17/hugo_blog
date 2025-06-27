---
# 核心元数据
author: lanshi
date: "2025-06-27T23:11:22+08:00"
lastmod:
title: API用法示例

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: 本文介绍了如何使用API进行文本总结、文本撰写、文本分类和文本翻译。

# 内容分类
series:
tags: ["API", "Python", "OpenAI"]
categories: ["编程"]

# SEO优化
description: 本文介绍了如何使用API进行文本总结、文本撰写、文本分类和文本翻译。
keywords: ["API", "Python", "OpenAI", "文本总结", "文本撰写", "文本分类", "文本翻译"]

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false



# 版权声明
copyright: true
---
## 文本总结

```python
# %%
import os
from openai import OpenAI
from dotenv import load_dotenv
# 加载 .env 文件
load_dotenv()

# %%
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# %%
def get_openai_response(client,prompt,model="qwen-turbo"):
    response=client.chat.completions.create(
        model=model,
        messages=[
            {"role":"user","content":prompt}
        ],
    )
    return response.choices[0].message.content

# %%
product_review="""
    这款手机整体表现非常不错，值得推荐！外观设计简约时尚，尤其是背面的渐变色处理，拿在手里特别有质感。屏幕显示效果也很出色，色彩鲜艳且清晰，看视频或玩游戏时沉浸感很强。性能方面，搭载了最新的处理器，运行流畅，多任务切换毫无压力，即使是大型游戏也能轻松应对。

不过，有一点需要改进的是，电池续航能力在重度使用时略显不足，如果全天候使用社交媒体、拍照和看视频，可能需要随身携带充电宝。另外，虽然快充速度很快，但充电器需要额外购买，这一点稍显不便。

相机表现令人惊喜，尤其是在光线充足的环境下拍摄的照片细节丰富，色彩还原真实。夜拍模式也有不错的表现，但偶尔会出现噪点问题。总体来说，这是一款性价比很高的手机，适合追求性能与颜值的用户。
"""

# %%
product_review_prompt=f"""
请根据以下用户评价，提取关键信息并进行简洁、客观的总结。总结应包括以下内容：
用户对手机整体的满意度；
优点（如外观、性能、拍照等）；
不足之处（如续航、配件等）；
最终推荐意见。
用户的评价内容会以三个#符号进行包围.

###
{product_review}
###
"""

# %%
print(get_openai_response(client=client,prompt=product_review_prompt))

# %%
```

## 文本撰写

```python
# %%
import os
from openai import OpenAI
from dotenv import load_dotenv
# 加载 .env 文件
load_dotenv()

# %%
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# %%
def get_openai_response(client,system_prompt,user_prompt,model="qwen-turbo"):
    response=client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system","content":system_prompt},
            {"role": "user","content": user_prompt}
        ],
    )
    return response.choices[0].message.content

# %%
system_prompt="""
你是小红书爆款写作专家，请你用以下步骤来进行创作，首先产出5个标题（含适当的emoji表情），其次产出1个正文（每一个段落含有适当的emoji表情，文末有合适的tag标签） 
 一、在小红书标题方面，你会以下技能： 
 1. 采用二极管标题法进行创作 
 2. 你善于使用标题吸引人的特点 
 3. 你使用爆款关键词，写标题时，从这个列表中随机选1-2个 
 4. 你了解小红书平台的标题特性 
 5. 你懂得创作的规则 
 二、在小红书正文方面，你会以下技能： 
 1. 写作风格 
 2. 写作开篇方法 
 3. 文本结构 
 4. 互动引导方法 
 5. 一些小技巧 
 6. 爆炸词 
 7. 从你生成的稿子中，抽取3-6个seo关键词，生成#标签并放在文章最后 
 8. 文章的每句话都尽量口语化、简短 
 9. 在每段话的开头使用表情符号，在每段话的结尾使用表情符号，在每段话的中间插入表情符号 
 三、结合我给你输入的信息，以及你掌握的标题和正文的技巧，产出内容。请按照如下格式输出内容，只需要格式描述的部分，如果产生其他内容则不输出： 
 一. 标题 
 [标题1到标题5] 
 [换行] 
 二. 正文 
 [正文] 
 标签：[标签]
"""

# %%
print(get_openai_response(client=client,system_prompt=system_prompt,user_prompt="学英语"))

# %%
```

## 文本分类

```python
# %%
import os
from openai import OpenAI
from dotenv import load_dotenv
# 加载 .env 文件
load_dotenv()

# %%
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# %%
def get_openai_response(client,prompt,model="qwen-turbo"):
    response=client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user","content": prompt}
        ],
    )
    return response.choices[0].message.content

# %%
q1="我刚买的xyz智能手表无法同步我的日历,我应该怎么办?"
q2="xyz智能手表的电池续航时间多久?每天使用会掉电快吗?"
q3="如何将手机上的音乐同步到xyz智能手表上播放?"
q4="xyz智能手表支持第三方应用吗?有哪些常用应用可以安装?"
q5="xyz智能手表的防水等级是多少?可以在游泳时佩戴吗?"
q6="如何更新xyz智能手表的系统固件?"
q7="xyz智能手表的血氧检测功能准确吗?有没有校准方法?"
q_list=[q1,q2,q3,q4,q4,q5,q6,q7]

# %%
category_list=["产品规格","使用咨询","功能比较","用户反馈","价格查询","故障问题","其他"]

# %%
classify_prompt_template="""
你的任务是为用户对产品的疑问进行分类,类别应该是这些里面的其中一个:{categories},
直接输出所属类别,不要有任何额外的描述或补充内容,
用户的问题内容会以三个#符号进行包围.

###
{question}
###
"""

# %%
for q in q_list:
    formatted_prompt=classify_prompt_template.format(categories=",".join(category_list),question=q)
    response=get_openai_response(client,formatted_prompt)
    print(response)

# %%
```

## 文本翻译

````python
# %%
import os
from openai import OpenAI
from dotenv import load_dotenv
# 加载 .env 文件
load_dotenv()

# %%
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# %%
def get_openai_response(client,prompt,model="qwen-turbo"):
    response=client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user","content": prompt}
        ],
    )
    return response.choices[0].message.content

# %%
translate_prompt="""
请将以下内容翻译成中文，要求翻译准确、自然、符合中文的表达习惯。

输出格式为:
```
============
原始消息(<文本的语言>):
<原始消息>
------------
翻译消息:
<翻译后的文本内容>
============
```
来自用户的消息内容会以三个#符号进行包围.
###
{message}
###
"""

# %%
message=input()
print(get_openai_response(client,translate_prompt.format(message=message)))
````
