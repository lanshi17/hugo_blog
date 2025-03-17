---
title: "FishStone Cloud V1.0 云存储系统"
summary: "基于 Flask 框架开发的文件管理系统"
anthor: me
date: 2025-03-08T14:15:45+08:00
draft: false
tags: ["项目","云盘系统","Flask","Python"]
categories: ["作品集","云盘系统","Flask"]
comments: true
showToc: true ## 显示目录
TocOpen: false ## 自动展开目录
hidemeta: false ## 是否隐藏文章的元信息，如发布日期、作者等
disableShare: false ## 底部不显示分享栏
showbreadcrumbs: true ##顶部显示当前路径
# 视觉配置
cover:
  image: "网站主页已登录.png"
  caption: "主页页面"
  relative: true
---
## FishStone-Cloud
[Github项目链接](https://github.com/lanshi47/FishStone-Cloud)
## 文件管理系统

这是一个基于 Flask 框架开发的文件管理系统，提供了用户认证、文件上传、下载、删除、分享等功能。项目支持多种账户操作，包括用户注册、登录、修改邮箱、修改密码、删除账户等。
{{< figure align=center src="网站主页未登录.png" width=80% >}}
![网站主页已登录](网站主页已登录.png)

#### 功能

1. **用户管理**
   - 注册、登录、邮箱验证、密码重置
   - 支持验证码登录和密码登录
   - 修改账户信息，如邮箱和密码
   - 删除账户功能，包括删除账户下所有文件
![密码登录](密码登录.png)
![验证码登录](验证码登录.png)
![邮箱注册](邮箱注册.png)
![邮箱注册2](邮箱注册2.png)
![重置密码](重置密码.png)
![注销账户](注销账户.png)
![修改密码](修改密码.png)
![修改邮件](修改邮件.png)
2. **文件管理**
   - 文件上传、下载、删除
   - 支持批量操作，如批量下载、分享、删除
   - 文件分享功能：生成带有验证码的分享链接
   - 显示文件统计信息：文件总数、下载量等
![文件列表](文件列表.png)
![分享文件](分享文件.png)
![上传文件](上传文件.png)
![上传成功](上传文件成功.png)

3. **数据展示**
   - 通过图表展示用户文件上传情况
   - 支持生成词云展示上传文件的关键词
![文件统计](文件统计.png)
![上传统计](上传统计.png)
![词云统计](词云统计.png)
4. **安全**
   - 验证码冷却机制，防止恶意频繁发送验证码
   - 使用 CSRF 保护表单安全
   - 密码加密存储

#### 安装

###### 克隆仓库

```bash
git clone https://github.com/yourusername/file-management-system.git
cd file-management-system
```

###### 创建虚拟环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate  ## Windows 环境下使用 venv\Scripts\activate
pip install -r requirements.txt
```

###### 设置环境变量

在根目录下创建一个 `.env` 文件，并添加以下内容：

```bash
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your_secret_key
SQLALCHEMY_DATABASE_URI=mysql+pymysql://username:password@localhost/db_name
MAIL_USERNAME=your_email@example.com
MAIL_PASSWORD=your_email_password
MAIL_SERVER=smtp.example.com
MAIL_PORT=587
MAIL_USE_TLS=1
```

###### 数据库迁移

项目使用了 Flask-Migrate 来管理数据库迁移。首先初始化数据库并应用迁移：

```bash
flask db init
flask db migrate
flask db upgrade
```

#### 配置

根据需求修改 `.env` 文件中的配置，如：

- 数据库连接字符串 `SQLALCHEMY_DATABASE_URI`
- 邮件服务器 `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USERNAME`, `MAIL_PASSWORD`
- `SECRET_KEY` 用于加密敏感信息，如用户的会话信息

#### 运行

###### 启动开发服务器

```bash
flask run
```

项目将会运行在 `http://127.0.0.1:5000`。可以通过浏览器访问来测试功能。

#### API 端点

###### 用户相关

- **POST** `/register` - 注册用户
- **POST** `/login` - 用户登录
- **POST** `/email_login` - 使用验证码登录
- **POST** `/reset_password_request` - 请求密码重置
- **POST** `/reset_password/<token>` - 重置密码

###### 文件管理

- **GET** `/files` - 查看用户文件列表
- **POST** `/files/upload` - 上传文件
- **POST** `/files/delete` - 删除文件
- **POST** `/files/share` - 分享文件
- **GET** `/files/download/<file_id>` - 下载文件

###### 图表和统计

- **GET** `/upload_stats` - 获取文件上传的统计数据
- **GET** `/wordcloud` - 生成文件名词云

#### 技术栈

- **前端**
  - HTML5, CSS3, JavaScript (使用 ECharts 生成图表)
  - Bootstrap 5 用于响应式布局
- **后端**
  - Flask 框架
  - Flask-Login (用户认证)
  - Flask-Migrate (数据库迁移)
  - Flask-Mail (邮件发送)
  - Flask-WTF (表单处理)
- **数据库**
  - MySQL (或其他支持的关系型数据库)
- **其他**
  - SQLAlchemy (ORM)
  - Werkzeug (密码加密)

#### 问题排查

1. **无法发送邮件？**
   - 检查 `.env` 文件中的邮件服务器配置是否正确。
   - 可能需要在邮件服务器中启用 "允许不安全应用" 或生成应用密码。

2. **数据库问题？**
   - 确保 MySQL 服务正在运行。
   - 确保 `.env` 文件中的 `SQLALCHEMY_DATABASE_URI` 配置正确。

3. **验证码冷却机制不起作用？**
   - 检查是否正确使用了 `cooldown_required` 装饰器，确保冷却时间设置合理。

#### 贡献

欢迎贡献代码！请按照以下步骤进行贡献：

1. Fork 此仓库
2. 创建你的功能分支 (`git checkout -b feature/new-feature`)
3. 提交你的更改 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/new-feature`)
5. 打开 Pull Request
