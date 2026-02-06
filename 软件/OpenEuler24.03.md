
# 安装中文字体

sudo dnf search wqy

查找相关字体随后进行安装

# 安装中文输入法

sudo dnf search fcitx

只有 **Fcitx (即 Fcitx 4)**

~~~
sudo dnf install fcitx fcitx-libpinyin fcitx-cloudpinyin fcitx-configtool fcitx-qt5
~~~

配置文件

# 安装基础图形库依赖 (Avalonia 在 Linux 运行通常需要这些)

sudo dnf install fontconfig libX11