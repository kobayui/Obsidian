# 1.安装Influxdb数据库

参考链接:[centos7安装influxdb2-阿里云开发者社区](https://developer.aliyun.com/article/1579245)

对于rpm安装包，使用rpm，默认安装在系统中

~~~
rpm -ivh package
~~~

安装后使用systemctl启动

~~~
systemctl start influxdb
~~~

目前尚未安装客户端Influx

# 2.安装 .Net8.0 SDK

- 安装包：dotnet-sdk-8.0.415-linux-x64.tar.gz

~~~
tar -xvf dotnet-sdk-8.0.415-linux-x64.tar.gz
~~~

- 可选：移动到合适的位置

~~~
sudo mkdir -p /opt/dotnet
sudo mv dotnet-sdk-8.0.xxxx /opt/dotnet
~~~

- 为了能够在任何目录下使用 `dotnet` 命令，你需要将 `.NET` 的安装路径添加到 `PATH` 环境变量中。
  编辑 `~/.bashrc` 文件，添加路径，并使其生效：

~~~
vi ~/.bashrc
export DOTNET_ROOT=/opt/dotnet
export PATH=$PATH:/opt/dotnet
source ~/.bashrc
~~~

- 验证安装

~~~
dotnet --version
~~~

出现版本号即安装成功

# 3.部署Avalonia桌面应用程序

[[Avalonia发布windows和Linux项目]]

[[EPICS 7.0.8在linux下采用ChannelAccess]]