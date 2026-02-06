# Windows


安装

~~~
dotnet tool install --global dotnet-counters
~~~


使用如下命令每隔1s监测指定软件运行指标

~~~
 dotnet-counters collect -p 26216 --counters System.Runtime[cpu-usage,working-set,gc-heap-size] --format csv -o test_report_data
~~~

注意：

不能开两个终端，一个monitor ，一个 collect


# Linux

安装

~~~
vi ~/.bashrc
export PATH=$PATH:$HOME/.dotnet/tools
source ~/.bashrc
dotnet tool install --global dotnet-counters
dotnet-counters --version
~~~

监测/收集

~~~
dotnet-counters ps
~~~

