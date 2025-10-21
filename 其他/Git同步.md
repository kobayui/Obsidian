# 下载Git

在命令行窗口安装

~~~
winget install --id Git.Git -e --source winget
~~~

# GitHub创建仓库

cd到仓库储存文件夹执行命令

使用 `git init` 在vault文件夹中初始化一个Git仓库。

~~~
git init
~~~


添加远端存储库：

~~~
git remote add origin https://github.com/kobayui/Obsidian.git
~~~

所有文件使用 **LF****(Line Feed)**换行符（适用于跨平台项目）

~~~
git config --global core.eol lf
git add --renormalize .
~~~

添加所有文件到仓库

~~~
git add .
git branch -M main
~~~

执行以下命令来设置全局的用户名和电子邮件地址。这将应用于所有的 Git 仓库

~~~
git config --global user.name "zw"
git config user.email "zhouwen640@outlook.com"
~~~

提交

~~~
git commit -m "Initial commit"
~~~

将文件推送到远程仓库：

~~~
git push -u origin main
~~~

每次更改文件后，提交并推送到远程仓库：

~~~
git add .
git commit -m "Update notes"
git push
~~~

在其他设备上同步

在其他设备上，使用相同的 Git 仓库，克隆远程仓库到新设备

~~~
git clone https://github.com/yourusername/yourrepository.git
~~~

在其他设备上，使用 `git pull` 来获取最新的文件：

~~~
git pull
~~~
