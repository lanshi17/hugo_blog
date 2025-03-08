---
# 基础元数据
title: "{{ replace .Name "-" " " | title }}"  # 自动生成标题（根据文件名）
date: {{ .Date.Format "2006-01-02T15:04:05-07:00" }}  # 自动生成创建时间
lastmod: {{ now.Format "2006-01-02T15:04:05-07:00" }}  # 最后修改时间（Git提交时自动更新）
draft: false  # 非草稿模式（与全局配置一致）

# 内容分类
tags: []
categories: []

# SEO优化
description: "文章描述（建议50-160字符，用于SEO）"  
keywords: []  # 文章专属关键词（会继承全局关键词）

# 主题功能配置
math: true  # 启用数学公式（与全局配置一致）
comment: true  # 开启评论（使用Giscus）
showToc: true  # 显示目录
tocOpen: false  # 默认折叠目录（与全局配置一致）
hiddenFromSearch: false  # 是否隐藏于搜索
hiddenFromHomePage: false  # 是否在首页隐藏

# 封面图配置（建议尺寸：1200x628像素）
cover:
  image: "/images/posts/default-cover.jpg"  # 默认封面图路径（存放于static目录）
  alt: "文章封面图"  # 无障碍文本
  caption: "封面图说明文字（可选）"  # 图片下方说明
  relative: false  # 不使用相对路径

# 高级配置（可选）
resources:  # 文章相关资源文件
  - name: "featured-image"
    src: "featured-image.jpg"
    title: "特色图片标题"
  - name: "other-resource"
    src: "document.pdf"

# 自定义字段（根据需求添加）
series: []  # 系列文章
copyright: true  # 是否显示版权声明
showFullContent: false  # 是否在首页显示全文
---

<!-- 文章摘要分隔符（摘要上方的部分会显示在列表页） -->
{{ "## 摘要" | emojify }}

这里是文章摘要，建议150字左右...

<!--more-->

{{ "## 正文内容" | emojify }}

从这里开始撰写您的文章...

{{ "### 二级标题" | emojify }}
{{</* notice tip "技巧提示" */>}}
这里是自定义提示框内容
{{</* /notice */>}}

{{</< admonition type="warning" title="注意事项" */>}}
这里是警告内容
{{</* /admonition */>}}
段落内容...