---
# 核心元数据
author: lanshi  # 修正为标量值
date: 2025-03-09T16:00:00+08:00 # 添加引号包裹
lastmod:   # 添加引号包裹
title: SQL数据库系统课后练习(一)  # 添加引号包裹"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true

# 内容分类
series: ["SQL", "数据库"] 
tags: ["SQL", "E-R图"]
categories: ["数据库系统"]

# SEO优化
description: "数据库系统核心知识点练习题集，包含SQL查询、关系代数、范式分解和E-R图设计等实践内容"
keywords: ["关系代数", "范式分解", "SQL练习"]

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "cover.png"
  alt: "数据库系统概念图示"
  caption: "关系模型与SQL实践"
  relative: true

# 版权声明
copyright: true
---


## ER图相关习题

## 关系代数运算相关习题

例题:现有关系S(S##,SNAME,AGE,SEX)),C(C##,CNAME,TEACHER)和SC(S##,C##,GRADE),试用表达式表示以下查询语句:
第一个问题:查询至少选修\"程军\"老师所授全部课程的学生姓名(SNAME);
解析:有三个部分,要查询\"程军\"老师的全部课程;要查询学生的选课记录,包括学号和课程号;要查询学生的姓名.
至少表示要查询选修了全部课程的学生,即选修了\"程军\"老师的全部课程的学生.
所以,首先要找到\"程军\"老师的全部课程,然后找到选修了这些课程的学生,最后找到这些学生的姓名.
这个查询可以分为三个部分: 1.找到\"程军\"老师的全部课程:\

```latex
$$\pi_{(C\##(\sigma_(TEACHER='\text{程军}')(C)))}$$ 2.学生选课记录:\
$$\pi_{S\## C\##(SC)}$$ 3.筛选学生:
$${\pi_{S\##,C\##(SC)}}\div{\pi_{(C\##(\sigma_(TEACHER='\text{程军}')(C)}}$$
综合以上三个部分,可以得到整个查询的表达式:\
$$\pi_{\text{SNAME}} \Big( S \Join \big( \pi_{\text{S\##, C\##}}(SC) \div \pi_{\text{C\##}}( \sigma_{\text{TEACHER='程军'}}(C) ) \big) \Big)$$
```

## SQL语句

#### 题1

1\. 设学生课程数据库中有三个关系：

学生关系 S (S##, SNAME, AGE, SEX) 学习关系 SC (S##, C##, GRADE) 课程关系 C
(C##, CNAME)

其中 S##, C##, SNAME, AGE, SEX, GRADE, CNAME
分别表示学号、课程号、姓名、年龄、性别、成绩和课程名。

用 SQL 语句表达以下操作：

1\. 检索选修课程名称为 \"MATHS\" 的学生学号与姓名。 答:

        SELECT DISTINCT S.S##, S.SNAME
        FROM S
        JOIN SC ON S.S## = SC.S##
        JOIN C ON SC.C## = C.C##
        WHERE C.CNAME = 'MATHS';

2\. 检索至少学习了课程号为 \"C1\" 和 \"C2\" 的学生的学号。

        SELECT S## FROM SC
        WHERE C## IN("C1","C2")
        GROUP BY S##
        HAVING COUNT (DISTINCT C##)=2

3\. 检索年龄在 18 到 20 之间（含 18 和
20）的女性学生的学号、姓名和年龄。

        SELECT S##, SNAME, AGE
        FROM S
        WHERE AGE BETWEEN 18 AND 20
          AND SEX = '女'; 

4\. 检索平均成绩达到 80 的学生学号和平均成绩。

        SELECT S##,AVG(GRADE) AS AVG_GRADE
        FROM SC
        GROUP BY S##
        HAVING AVG(GRADE)>=80;

5\. 检索选修了全部课程的学生姓名。

        SELECT S.SNAME
        FROM S 
        WHERE NOT EXISTS(
            SELECT C.C##
            FROM C 
            WHERE S## NOT IN(
                SELECT *
                FROM SC
                WHERE S.S##=SC.S##
                AND C.C##=SC.C##
            )
        )

6\. 检索选修了三个课程以上的学生的学号。

        SELECT S##
        FROM SC
        GROUP BY S##
        HAVING COUNT(DISTINCT C##) > 3 

#### 题2:学生-课程数据库中包括三个表

- 学生表：**Student** (`Sno`, `Sname`, `Sex`, `Sage`, `Sdept`)

- 课程表：**Course** (`Cno`, `Cname`, `Ccredit`)

- 学生选课表：**SC** (`Sno`, `Cno`, `Grade`)

其中
`Sno`、`Sname`、`Sex`、`Sage`、`Sdept`、`Cno`、`Cname`、`Ccredit`、`Grade`
分别表示学号、姓名、性别、年龄、所在系名、课程号、课程名、学分和成绩。

**试用 SQL 语言完成下列操作：**

1. 查询选修课程包括 "1042" 号学生所学的课程的学生学号。\

                SELECT DISTINCT Sno
                FROM SC AS X
                WHERE NOT EXISTS (
                    SELECT Cno
                    FROM SC
                    WHERE Sno = '1042'    -- 获取1042学生的所有课程
                    AND Cno NOT IN (       -- 检查是否存在1042选修的课程未被当前学生选修
                        SELECT Cno
                        FROM SC AS Y
                        WHERE Y.Sno = X.Sno
                    )
                );

2. 创建一个计算系学生信息视图 **CS_VIEW**，包括 `Sno` 学号、`Sname`
    姓名、`Sex` 性别。

                CREATE VIEW CS_VIEW AS
                SELECT Sno,Sname,Sex
                FROM Student
                WHERE Sdept='计算系'

3. 通过上面第 2 题创建的视图修改数据，把王平的名字改为王慧平。

                UPDATE CS_VIEW
                SET Sname='王慧平'
                WHERE Sname='王平'

4. 创建一选修数据库课程信息的视图，视图名称为
    **datascore_view**，包含学号、姓名、成绩。

                CREATE VIEW datascore_view AS
                SELECT Student.Sno, Sname, Grade
                FROM Student
                JOIN SC ON Student.Sno = SC.Sno
                JOIN Course ON SC.Cno = Course.Cno
                WHERE Course.Cname = '数据库';

## 关系模式

#### 题1:已知学生关系模式如下: {##题1已知学生关系模式如下 .unnumbered}

关系模式 $S(Sno, Sname, SD, Sdname, Course, Grade)$

其中：

$Sno$ 学号，

$Sname$ 姓名，

$SD$ 系名，

$Sdname$ 系主任名，

$Course$ 课程，

$Grade$ 成绩。

1. 写出关系模式 $S$ 的基本函数依赖和主码。

    $Sno \rightarrow Sname$\
    $Sno \rightarrow Course$\
    $(Sno,Course) \rightarrow Grade$\
    $Sno \rightarrow SD$\
    $SD \rightarrow Sdname$

    主码为(Sno,Course)

2. 原关系模式 $S$ 为几范式？为什么？分解成高一级范式，并说明为什么？
    原关系模式为1NF,Grade对主码存在完全依赖;
    其他非主码候选健对主码存在部分依赖. 2NF如下所示:

    $S1(Sno,Sname,SD,Sdname)$\
    $S2(Sno,Course,Grade)$\

3. 将关系模式分解成 3NF，并说明为什么。 S1存在传递依赖,还可以继续分解
    3NF如下所示:

    $S11(Sno,Sname,SD)$\
    $S12(SD,Sdname)$\
    $S2(Sno,Course,Grade)$

#### 题2:设有如下关系 R {##题2设有如下关系-r .unnumbered}

| 课程名 | 教师名 | 教师地址 |
|--------|--------|----------|
| C1     | 马千里 | D1       |
| C2     | 于得水 | D1       |
| C3     | 余快   | D2       |
| C4     | 于得水 | D1       |

1. 它为几范式？为什么？

    2NF,存在传递依赖,\
    $\text{课程名} \rightarrow \text{教师名}$\
    $\text{教师名} \rightarrow \text{教师地址}$\
    存在$\text{课程名} \rightarrow \text{教师地址}$的传递依赖;但又不存在部分函数依赖.
    属于2NF

2. 是否存在删除操作异常？若存在，则说明是在什么情况下发生的？

    存在,当删除课程名时,可能某些信息会全部丢失

3. 将它分解为高一级范式，分解后的关系是如何解决分解前可能存在的删除操作问题的？

    转为3NF,消除传递依赖,分解原表为R1,R2

| 课程名 | 教师名 |
|--------|--------|
| C1     | 马千里 |
| C2     | 于得水 |
| C3     | 余快   |
| C4     | 于得水 |

: 关系 R1

| 教师名 | 教师地址 |
|--------|----------|
| 马千里 | D1       |
| 于得水 | D1       |
| 余快   | D2       |

: 关系 R2

#### 题3. 设某商业集团数据库中有一关系模式 R 如下： {##题3.-设某商业集团数据库中有一关系模式-r-如下 .unnumbered}

R (商店编号, 商品编号, 数量, 部门编号, 负责人)

如果规定：(1) 每个商店的每种商品只能在一个部门销售；(2)
每个商店的每个部门只有一个负责人；(3)
每个商店的每种商品只有一个库存数量。

试回答下列问题：

\(1\) 根据上述规定，写出关系模式 R 的基本函数依赖；

::: center
$\text{部门编号} \rightarrow \text{负责人}$\
$\text{商品编号} \rightarrow \text{数量}$\
$(\text{商店编号,商品编号}) \rightarrow \text{部门编号}$\
:::

\(2\) 找出关系模式 R 的候选码；

(商店编号,商品编号)

\(3\) 试问关系模式 R 最高已经达到第几范式？为什么？

2NF,存在$\mbox{(商店编号,商品编号)} \rightarrow \mbox{负责人}$的传递依赖.

\(4\) 如果 R 不属于 3NF，请将 R 分解成 3NF 模式集。

R1(商店编号,商品编号,部门编号,数量)

R2(商品编号,商店编号,负责人)

## 实体关系模型设计

#### 题1:设有如下实体 {##题1设有如下实体 .unnumbered}

- 学生：学号、单位、姓名、性别、年龄、选修课程名；

- 教师：教师号、姓名、性别、职称、讲授课程编号；

- 课程：编号、课程名、开课单位、任课教师号；

- 单位：单位名称、电话、教师号、教师名。

上述实体中存在如下关联：

1. 一个学生可选择多门课程，一门课程可由多个学生选择；

2. 一个教师可讲授多门课程，一门课程可由多个教师讲授；

3. 一个单位有多个教师，一个教师只能属于一个单位。

试完成如下任务：

1. 分别设计学生信息和教师信息的 E-R 图；

2. 将上述设计完成的 E-R 图合并成一个全局 E-R 图；

    题1,题2如[1](##图5-1){reference-type="ref+label" reference="图5-1"}所示
    {{< figure align=center src="题1-E-R图.jpg" width=80% >}}

3\. 将该全局 E-R 图转换为对应的数据库逻辑结构：  
    单位(单位名,电话)  
    课程(课程编号,课程名,单位名)  
    学生(学号,单位名.姓名,性别,年龄)  
    教师(教师号,姓名,性别,职称,单位名)  
    选修(学号,课程编号)  
    讲授(教师号,课程编号)
