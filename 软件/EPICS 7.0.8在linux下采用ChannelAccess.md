
clientChannels中原先的写法：windows可以正常使用，但是linux不行

~~~
foreach (string name in channelNames)
{
    var client = new CAClient();
    EpicsSharp.ChannelAccess.Client.Channel channel = client.CreateChannel<double>(name);
    channels.Add(channel);
}
~~~

修改：foreach循环中去掉新建CAClient，在类的构造函数中声明一个CAClient

~~~
public ClientChannel(CAClient client)
~~~

在EpicsService中构造函数建立CaClient和ClientChannel

~~~
CaClient = new CAClient();
clientChannel = new ClientChannel(CaClient);
~~~

