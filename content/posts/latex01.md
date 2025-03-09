---
title: "ctex的字体设置"
weight: 1
date: 2025-03-07T00:00:00+08:00
aliases: ["/first"]
tags: ["latex","notebook"]
categories : ["latex"]
pubulisdata: 2025-03-07T20:11:00+08:00
author: "lanshi"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
description: "latex的字体设置"
editPost:
    URL: "https://github.com/lanshi47/hugo_blog/tree/master/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
## latex字体设置

### 需要去官网下载相关离线字体,保存到本地再进行应用,因为很多字体并没有版权

- [宋体字体下载](https://github.com/flyskywhy/react-native-font-sim)
- [新罗马字体下载](https://github.com/weiweihuanghuang/Times-New-Bastard)

---

```latex
\documentclass[a4paper,12pt,UTF8,fontset=none]{ctexart}
\usepackage{geometry} % 页面设置
\usepackage{xcolor} % Color support for listings
\usepackage{graphicx}
\usepackage{amssymb} % For join symbol
\usepackage{amsmath}
\usepackage{listings}
\usepackage{longtable}
\usepackage{booktabs} % 添加booktabs宏包以支持三线表

\usepackage{newtxtext,newtxmath}
\usepackage{fontspec}
\usepackage{titlesec}

% 设置英文主字体为 Times New Roman
\setmainfont{Times New Roman}[Path=D:/Program Files/MiKTeX/fonts/custom/, Extension=.otf]

% 设置英文粗体字体为 Times New Roman Bold
\newfontfamily\enboldfont{Times New Roman Bold}[Path=D:/Program Files/MiKTeX/fonts/custom/, Extension=.otf]

% 设置中文正文字体为宋体
\setCJKmainfont{SimSun}[Path=D:/Program Files/MiKTeX/fonts/custom/, Extension=.ttf]

% 设置中文黑体字体
\setCJKfamilyfont{zhhei}{SimHei}[Path=D:/Program Files/MiKTeX/fonts/custom/, Extension=.ttf]
\newcommand{\heiti}{\CJKfamily{zhhei}}



% 设置标题格式
\titleformat{\section}
  {\heiti\zihao{-3}\enboldfont}{\thesection}{1em}{}
\titleformat{\subsection}
  {\heiti\zihao{4}\enboldfont}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\heiti\zihao{-4}\enboldfont}{\thesubsubsection}{1em}{}



% 设置代码块样式
\lstset{
    basicstyle=\ttfamily,
    columns=fullflexible,
    frame=single,
    breaklines=true,
    postbreak=\mbox{\textcolor{red}{$\hookrightarrow$}\space},
    language=SQL,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},    
    rulecolor=\color{black!30},%边框颜色
    stringstyle=\color{red},
    escapeinside={\%*}{*)},
    showstringspaces=false,
    captionpos=b % 设置标题位置, b表示在底部
}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm} % 页边距

```
