关键点（为何离线失败）

- 你的 HTML 很可能引用了外部 CDN（例如 vis-network、d3、jquery、css 等）。离线时这些资源无法加载，页面空白或 JS 报错。
- CEF 对 file:// 的跨域 / XHR / fetch/模块加载可能有限制（浏览器安全策略），需要在初始化时打开 allow-file-access-from-files / allow-universal-access-from-files 或禁用 web-security，或改用本地 http 服务。
- 如果 HTML 用 fetch/XHR 加载本地相对文件，某些 CEF 版本会阻止，需要上述命令行开关或使用 HTTP。

改正：

在应用内启动一个本地 HTTP 服务器并通过 [http://localhost:port/your.html](http://localhost:port/your.html) 加载 优点：

- 与浏览器行为一致（file:// 的各种限制都会消失）。
- 更可靠地支持 XHR/fetch/ESM模块/动态加载资源。
- 可在不修改原 HTML 的情况下工作（只要资源在同一目录结构）。

LocalHttpServer.cs（一个简单的本地静态文件服务器，用于开发/部署阶段）
~~~
using System;
using System.IO;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace HtmlViewerApp;

public class LocalHttpServer : IDisposable
{
    private readonly HttpListener _listener;
    private readonly string _root;
    private readonly CancellationTokenSource _cts = new();

    public int Port { get; }

    public LocalHttpServer(string rootPath, int port = 0)
    {
        _root = rootPath ?? throw new ArgumentNullException(nameof(rootPath));
        _listener = new HttpListener();
        // 使用随机端口（0 表示系统分配）
        Port = port == 0 ? GetRandomAvailablePort() : port;
        _listener.Prefixes.Add($"http://localhost:{Port}/");
    }

    public void Start()
    {
        _listener.Start();
        Task.Run(() => HandleLoopAsync(_cts.Token));
    }

    private async Task HandleLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var ctx = await _listener.GetContextAsync().ConfigureAwait(false);
                _ = Task.Run(() => HandleRequestAsync(ctx), ct);
            }
            catch (Exception)
            {
                if (ct.IsCancellationRequested) break;
            }
        }
    }

    private static string GetMimeType(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext switch
        {
            ".html" or ".htm" => "text/html",
            ".js" => "application/javascript",
            ".css" => "text/css",
            ".json" => "application/json",
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            ".svg" => "image/svg+xml",
            ".wasm" => "application/wasm",
            _ => "application/octet-stream"
        };
    }

    private async Task HandleRequestAsync(HttpListenerContext ctx)
    {
        try
        {
            var urlPath = ctx.Request.Url.LocalPath.TrimStart('/');
            if (string.IsNullOrEmpty(urlPath)) urlPath = "index.html";

            var file = Path.Combine(_root, urlPath.Replace('/', Path.DirectorySeparatorChar));
            if (!File.Exists(file))
            {
                ctx.Response.StatusCode = 404;
                await using var writer = new StreamWriter(ctx.Response.OutputStream);
                await writer.WriteAsync("Not found");
                ctx.Response.Close();
                return;
            }

            var bytes = await File.ReadAllBytesAsync(file);
            ctx.Response.ContentType = GetMimeType(file);
            ctx.Response.ContentLength64 = bytes.Length;
            await ctx.Response.OutputStream.WriteAsync(bytes, 0, bytes.Length);
            ctx.Response.Close();
        }
        catch
        {
            try { ctx.Response.StatusCode = 500; ctx.Response.Close(); } catch { }
        }
    }

    public void Dispose()
    {
        _cts.Cancel();
        try { _listener.Stop(); } catch { }
        _listener.Close();
    }

    private static int GetRandomAvailablePort()
    {
        var listener = new TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}
~~~

使用 LocalHttpServer（启动并导航）

~~~
// 假定你已有 Browser 初始化逻辑（AvaloniaCefBrowser _browser）
// 当用户选择 HTML 文件时：
private LocalHttpServer? _server;

private void LoadHtmlFileViaHttp(string htmlFilePath)
{
    // 1. 启动本地服务器，根目录为 html 文件所在目录
    var root = Path.GetDirectoryName(htmlFilePath)!;
    _server?.Dispose();
    _server = new LocalHttpServer(root); // 随机端口
    _server.Start();

    // 2. 构建到文件的 HTTP URL（保留相对路径）
    var fileName = Path.GetFileName(htmlFilePath);
    var url = $"http://localhost:{_server.Port}/{Uri.EscapeUriString(fileName)}";

    // 3. 导航 CefGlue 浏览器
    _browser!.Address = url;
}
~~~