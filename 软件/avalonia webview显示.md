[[动态因果链显示]]

 //若是在虚拟机中，这个功能可能会因显卡/GPU加速而导致崩溃
# Avalonia.HtmlRenderer

将python生成的动态因果链显示在avalonia界面当中

HtmlRenderer 不支持 JavaScript。你的 demo.html 使用 vis-network 的脚本和动画，在 HtmlRenderer 中不会运行，最终只能显示静态 HTML 结构（若有）

axaml文件

~~~
<HtmlControl Grid.Row="1"
      Name="HtmlHost"
      HorizontalAlignment="Stretch"                         
				  VerticalAlignment="Stretch"/>
~~~

xaml.cs

~~~
public partial class DataProcessView : UserControl
{
    private const string AssetRelativePath = "Assets/demo.html";
    private HtmlControl? _htmlHost;

    public DataProcessView()
    {
        InitializeComponent();
        _htmlHost = this.FindControl<HtmlControl>("HtmlHost");
        _ = LoadHtmlAsync();

    }
    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }

    private async Task LoadHtmlAsync()
    {
        if (_htmlHost is null) return;

        try
        {
            var asmName = Assembly.GetEntryAssembly()?.GetName().Name
                          ?? Assembly.GetExecutingAssembly().GetName().Name!;
            var normalized = AssetRelativePath.Replace('\\', '/').TrimStart('/');
            var avares = new Uri($"avares://{asmName}/{normalized}");
            using var stream = AssetLoader.Open(avares);
            using var reader = new StreamReader(stream);
            var html = await reader.ReadToEndAsync();

            // 设置 HtmlRenderer 属性：优先 HtmlText，其次 Text
            var prop = typeof(HtmlControl).GetProperty("HtmlText")
                       ?? typeof(HtmlControl).GetProperty("Text");
            if (prop is not null && prop.CanWrite)
            {
                prop.SetValue(_htmlHost, html);
            }
            else
            {
                Console.WriteLine("HtmlRenderer 控件缺少 HtmlText/Text 属性，请检查包版本。");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("加载 HTML 失败: " + ex);
        }
    }
}
~~~

但仍不显示html，静态也没有

# using AvaloniaWebView

显示黑屏

axaml

~~~
<WebView Grid.Row="1"
     Name="WebViewControl"
     HorizontalAlignment="Stretch"
     VerticalAlignment="Stretch"
				 Background="AliceBlue"/>
~~~

xaml.cs
~~~
public partial class DataProcessView : UserControl
{
    private const string AssetRelativePath = "Assets/demo.html";
    private WebView? _webView;
    private bool _initialized; // 新增：防止重复导航

    public DataProcessView()
    {
        InitializeComponent();
        _webView = this.FindControl<WebView>("WebViewControl");
        // 在控件加载后再导航，确保可视树与尺寸就绪
        this.AttachedToVisualTree += async (_, __) =>
        {
            if (_initialized) return;
            _initialized = true;
            await LoadHtmlAsync();
        };

    }
    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }

    private async Task LoadHtmlAsync()
    {
        if (_webView is null) return;

        // 先用内嵌 data: 测试
        var testHtml = "<!DOCTYPE html><html><body style='background:#fff'><h1>Hello WebView</h1></body></html>";
        var dataUrl = "data:text/html;base64," + Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(testHtml));
        NavigateUrl(dataUrl);

        // 再加载 Assets 的 html
        try
        {
            var asmName = Assembly.GetEntryAssembly()?.GetName().Name
                          ?? Assembly.GetExecutingAssembly().GetName().Name!;
            var normalized = AssetRelativePath.Replace('\\', '/').TrimStart('/');
            var avares = new Uri($"avares://{asmName}/{normalized}");

            using var stream = AssetLoader.Open(avares);
            using var reader = new StreamReader(stream);
            var html = await reader.ReadToEndAsync();

            var tmp = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}_{Path.GetFileName(normalized)}");
            await File.WriteAllTextAsync(tmp, html);

            var fileUri = $"file:///{tmp.Replace('\\', '/')}";
            NavigateUrl(fileUri);
        }
        catch (Exception ex)
        {
            Console.WriteLine("加载 Assets HTML 失败: " + ex);
        }
    }

    private void NavigateUrl(string url)
    {
        var navigate = typeof(WebView).GetMethod("Navigate", new[] { typeof(string) });
        if (navigate != null)
        {
            navigate.Invoke(_webView, new object[] { url });
            return;
        }
        var prop = typeof(WebView).GetProperty("Url") ?? typeof(WebView).GetProperty("Source");
        if (prop != null && prop.PropertyType == typeof(string))
        {
            prop.SetValue(_webView, url);
            return;
        }
        Console.WriteLine("AvaloniaWebView 未提供 Navigate(string)/Url/Source，请检查包版本与文档。");
    }
}
~~~

# CefNet.Avalonia目前还不支持Avalonia11.x

# 可以用默认浏览器打开，该方法不能内嵌

~~~
<Grid RowDefinitions="*,Auto">
    <!-- 主界面 -->
    <Border Background="#1e1e1e">
        <StackPanel VerticalAlignment="Center" 
                    HorizontalAlignment="Center" 
                    Spacing="20">
            <TextBlock Text="🔗" FontSize="64" HorizontalAlignment="Center"/>
            <TextBlock Text="动态因果链可视化查看器" 
                       FontSize="24" 
                       Foreground="White"
                       HorizontalAlignment="Center"/>
            <TextBlock x:Name="FilePathText"
                       Text="未选择文件" 
                       FontSize="12" 
                       Foreground="#888"
                       HorizontalAlignment="Center"/>
            <StackPanel Orientation="Horizontal" 
                        HorizontalAlignment="Center" 
                        Spacing="10"
                        Margin="0,20,0,0">
                <Button Content="📂 选择 HTML 文件" 
                        Click="OnSelectFileClick"
                        Background="#0078d4"
                        Foreground="White"
                        Padding="20,10"
                        CornerRadius="5"/>
                <Button x:Name="OpenButton"
                        Content="🌐 在浏览器中打开" 
                        Click="OnOpenInBrowserClick"
                        Background="#444"
                        Foreground="White"
                        Padding="20,10"
                        CornerRadius="5"
                        IsEnabled="False"/>
            </StackPanel>
        </StackPanel>
    </Border>
    
    <!-- 状态栏 -->
    <Border Grid.Row="1" Background="#252526" Padding="10,5">
        <TextBlock x:Name="StatusText" Text="就绪" Foreground="#888" FontSize="12"/>
    </Border>
</Grid>
~~~

~~~
public partial class MainWindow : Window
{
    private string? _selectedFilePath;
    public MainWindow()
    {
        InitializeComponent();
    }

    private async void OnSelectFileClick(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "选择 HTML 文件",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
            new FilePickerFileType("HTML 文件")
            {
                Patterns = new[] { "*.html", "*.htm" }
            }
        }
        });

        if (files.Count > 0)
        {
            _selectedFilePath = files[0].Path.LocalPath;
            FilePathText.Text = _selectedFilePath;
            OpenButton.IsEnabled = true;
            StatusText.Text = $"已选择: {Path.GetFileName(_selectedFilePath)}";
        }
    }

    private void OnOpenInBrowserClick(object? sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_selectedFilePath) || !File.Exists(_selectedFilePath))
        {
            StatusText.Text = "请先选择有效的 HTML 文件";
            return;
        }

        try
        {
            OpenInBrowser(_selectedFilePath);
            StatusText.Text = $"已在浏览器中打开: {Path.GetFileName(_selectedFilePath)}";
        }
        catch (Exception ex)
        {
            StatusText.Text = $"打开失败: {ex.Message}";
        }
    }

    private static void OpenInBrowser(string filePath)
    {
        var url = new Uri(filePath).AbsoluteUri;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            Process.Start("xdg-open", url);
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            Process.Start("open", url);
        }
    }
}
~~~

# CefGlue.Avalonia成功

D:\工作\vs_wpf\2025\Monitor_Avalonia\cefglue\HtmlViewerApp项目为成功的独立demo

以下为集成到监测系统中的代码

proggram.cs

~~~
    [STAThread]
    public static void Main(string[] args)
    {
        try
        {
            BuildAvaloniaApp()
                .AfterSetup(_ =>
                {
                    InitializeCef();
                })
                .StartWithClassicDesktopLifetime(args);
        }
        finally
        {
            // 关闭 CEF
            CefRuntime.Shutdown();
        }
    }

    private static void InitializeCef()
    {
        var cachePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "HtmlViewerApp", "cache"
        );
        Directory.CreateDirectory(cachePath);

        var settings = new CefSettings
        {
            RootCachePath = cachePath,
            CachePath = cachePath,
            LogSeverity = CefLogSeverity.Warning,
            WindowlessRenderingEnabled = false,
            NoSandbox = true
        };

        try
        {
            CefRuntimeLoader.Initialize(settings);
            Console.WriteLine("CEF 初始化成功!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"CEF 初始化失败: {ex.Message}");
            throw;
        }
    }

    public static AppBuilder BuildAvaloniaApp()
    {
        var app = AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace()
            .UseXamlDisplay();

        if (OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() || OperatingSystem.IsLinux())
        {
            //app.UseManagedSystemDialogs();
        }

        return app;
    }

~~~

DataProcessView.axaml.cs

~~~
private string? _currentFilePath;
private double _zoomLevel = 0;

// 浏览器控件（在代码中创建）
private AvaloniaCefBrowser? _browser;

// 控件引用
private readonly Grid _browserContainer;
private readonly Border _placeholderPanel;
private readonly TextBlock _addressText;
private readonly TextBlock _statusText;
private readonly TextBlock _loadingText;
private readonly TextBlock _zoomText;
private readonly Ellipse _loadingIndicator;
private readonly Button _backButton;
private readonly Button _forwardButton;

public DataProcessView()
{
    InitializeComponent();

    // 获取控件引用
    _browserContainer = this.FindControl<Grid>("BrowserContainer")!;
    _placeholderPanel = this.FindControl<Border>("PlaceholderPanel")!;
    _addressText = this.FindControl<TextBlock>("AddressText")!;
    _statusText = this.FindControl<TextBlock>("StatusText")!;
    _loadingText = this.FindControl<TextBlock>("LoadingText")!;
    _zoomText = this.FindControl<TextBlock>("ZoomText")!;
    _loadingIndicator = this.FindControl<Ellipse>("LoadingIndicator")!;
    _backButton = this.FindControl<Button>("BackButton")!;
    _forwardButton = this.FindControl<Button>("ForwardButton")!;

    // 窗口关闭时清理
    //Closed += OnWindowClosed;
}

/// <summary>
/// 初始化浏览器控件
/// </summary>
private void InitializeBrowser()
{
    if (_browser != null) return;

    _browser = new AvaloniaCefBrowser();
    _browser.IsVisible = false;

    // 注册事件
    _browser.LoadStart += OnBrowserLoadStart;
    _browser.LoadEnd += OnBrowserLoadEnd;
    _browser.LoadingStateChange += OnBrowserLoadingStateChange;
    //_browser.TitleChanged += OnBrowserTitleChanged;
    _browser.AddressChanged += OnBrowserAddressChanged;

    // 添加到容器
    _browserContainer.Children.Insert(0, _browser);
}

private void OnWindowClosed(object? sender, EventArgs e)
{
    _browser?.Dispose();
}

private async void OnOpenFileClick(object? sender, RoutedEventArgs e)
{
    var topLevel = TopLevel.GetTopLevel(this);
    var storageProvider = topLevel?.StorageProvider;
    if (storageProvider is null) return;
    var files = await storageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
    {
        Title = "选择 HTML 文件",
        AllowMultiple = false,
        FileTypeFilter = new[]
        {
            new FilePickerFileType("HTML 文件")
            {
                Patterns = new[] { "*.html", "*.htm" }
            },
            new FilePickerFileType("所有文件")
            {
                Patterns = new[] { "*.*" }
            }
        }
    });

    if (files.Count > 0)
    {
        _currentFilePath = files[0].Path.LocalPath;
        LoadHtmlFile(_currentFilePath);
    }
}

private void LoadHtmlFile(string filePath)
{
    if (!File.Exists(filePath))
    {
        _statusText.Text = $"文件不存在: {filePath}";
        return;
    }

    // 初始化浏览器（如果尚未初始化）
    InitializeBrowser();

    // 隐藏占位符，显示浏览器
    _placeholderPanel.IsVisible = false;
    _browser!.IsVisible = true;

    // 构建 file:// URL
    var fileUrl = new Uri(filePath).AbsoluteUri;

    _statusText.Text = $"正在加载: {System.IO.Path.GetFileName(filePath)}";
    SetLoading(true);

    // 导航到文件
    _browser.Address = fileUrl;
}

private void OnRefreshClick(object? sender, RoutedEventArgs e)
{
    if (_browser == null) return;
    _browser.Reload();
    _statusText.Text = "正在刷新... ";
    SetLoading(true);
}

private void OnDevToolsClick(object? sender, RoutedEventArgs e)
{
    _browser?.ShowDeveloperTools();
}

private void OnGoBackClick(object? sender, RoutedEventArgs e)
{
    if (_browser?.CanGoBack == true)
    {
        _browser.GoBack();
    }
}

private void OnGoForwardClick(object? sender, RoutedEventArgs e)
{
    if (_browser?.CanGoForward == true)
    {
        _browser.GoForward();
    }
}

private void OnZoomInClick(object? sender, RoutedEventArgs e)
{
    if (_browser == null) return;
    _zoomLevel = Math.Min(_zoomLevel + 0.5, 4.0);
    _browser.ZoomLevel = _zoomLevel;
    UpdateZoomText();
}

private void OnZoomOutClick(object? sender, RoutedEventArgs e)
{
    if (_browser == null) return;
    _zoomLevel = Math.Max(_zoomLevel - 0.5, -4.0);
    _browser.ZoomLevel = _zoomLevel;
    UpdateZoomText();
}

private void OnZoomResetClick(object? sender, RoutedEventArgs e)
{
    if (_browser == null) return;
    _zoomLevel = 0;
    _browser.ZoomLevel = _zoomLevel;
    UpdateZoomText();
}

private void UpdateZoomText()
{
    int percent = (int)(100 * Math.Pow(1.2, _zoomLevel));
    _zoomText.Text = $"{percent}%";
}

private void OnBrowserLoadStart(object? sender, LoadStartEventArgs e)
{
    Dispatcher.UIThread.Post(() =>
    {
        SetLoading(true);
        _statusText.Text = "加载中...";
    });
}

private void OnBrowserLoadEnd(object? sender, LoadEndEventArgs e)
{
    Dispatcher.UIThread.Post(() =>
    {
        SetLoading(false);
        if (!string.IsNullOrEmpty(_currentFilePath))
        {
            _statusText.Text = $"✓ 已加载: {System.IO.Path.GetFileName(_currentFilePath)}";
        }
    });
}

private void OnBrowserLoadingStateChange(object? sender, LoadingStateChangeEventArgs e)
{
    Dispatcher.UIThread.Post(() =>
    {
        _backButton.IsEnabled = e.CanGoBack;
        _forwardButton.IsEnabled = e.CanGoForward;

        if (!e.IsLoading)
        {
            SetLoading(false);
        }
    });
}

private void OnBrowserTitleChanged(object? sender, string title)
{
    Dispatcher.UIThread.Post(() =>
    {
        if (!string.IsNullOrEmpty(title))
        {
            //Title = $"{title} - 动态因果链查看器";
        }
    });
}

private void OnBrowserAddressChanged(object? sender, string address)
{
    Dispatcher.UIThread.Post(() =>
    {
        _addressText.Text = address;
    });
}

private void SetLoading(bool isLoading)
{
    _loadingIndicator.Fill = new SolidColorBrush(
        Color.Parse(isLoading ? "#0078d4" : "#444"));
    _loadingText.Text = isLoading ? "加载中..." : "";
}
~~~

DataProcessView.axaml

~~~
<suki:SukiStackPage Margin="5">
	<suki:SukiStackPage.Content>
		<SplitView Name="仿真" CompactPaneLength="50" DisplayMode="CompactInline" IsPaneOpen="False" PanePlacement="Right" OpenPaneLength="0">
            <Grid RowDefinitions="Auto,*,Auto">
                <!-- 顶部工具栏 -->
                <Border Grid.Row="0" Background="{Binding Background, RelativeSource={RelativeSource AncestorType=UserControl}}" Padding="10,8">
                    <Grid ColumnDefinitions="Auto,*,Auto">
                        <!-- 左侧按钮 -->
                        <StackPanel Grid.Column="0" Orientation="Horizontal" Spacing="6">
                            <Button Content="📂 打开 HTML" 
                                    Click="OnOpenFileClick"
                                    Background="#0078d4"
                                    Foreground="White"
                                    Padding="14,6"
                                    CornerRadius="4"/>
                            <Button Content="🔄" 
                                    Click="OnRefreshClick"
                                    Background="#444"
                                    CornerRadius="4"
                                    ToolTip.Tip="刷新"/>
                            <Button Content="🛠" 
                                    Click="OnDevToolsClick"
                                    Background="#444"
                                    Foreground="White"
                                    CornerRadius="4"
                                    ToolTip.Tip="开发者工具"/>
                
                            <Rectangle Width="1" Fill="#555" Margin="6,4"/>
                
                            <Button x:Name="BackButton"
                                    Content="◀" 
                                    Click="OnGoBackClick"
                                    Background="#444"
                                    Foreground="White"
                                    CornerRadius="4"
                                    IsEnabled="False"
                                    ToolTip.Tip="后退"/>
                            <Button x:Name="ForwardButton"
                                    Content="▶" 
                                    Click="OnGoForwardClick"
                                    Background="#444"
                                    Foreground="White"
                                    CornerRadius="4"
                                    IsEnabled="False"
                                    ToolTip.Tip="前进"/>
                        </StackPanel>
            
                        <!-- 地址栏 -->
                        <Border Grid.Column="1" 
                                Background="{Binding Background, RelativeSource={RelativeSource AncestorType=UserControl}}" 
                                CornerRadius="4" 
                                Margin="12,0"
                                Padding="10,4">
                            <TextBlock x:Name="AddressText" 
                                       Text="请选择 HTML 文件..."
                                       Foreground="#aaa"
                                       FontSize="12"
                                       VerticalAlignment="Center"
                                       TextTrimming="CharacterEllipsis"/>
                        </Border>
            
                        <!-- 右侧缩放 -->
                        <StackPanel Grid.Column="2" Orientation="Horizontal" Spacing="4">
                            <Button Content="➖" 
                                    Click="OnZoomOutClick"
                                    Background="#444"
                                    Foreground="White"
                                    CornerRadius="4"
                                    ToolTip.Tip="缩小"/>
                            <TextBlock x:Name="ZoomText" 
                                       Text="100%"
                                       Foreground="#aaa"
                                       VerticalAlignment="Center"
                                       Width="45"
                                       TextAlignment="Center"/>
                            <Button Content="➕" 
                                    Click="OnZoomInClick"
                                    Background="#444"
                                    Foreground="White"
                                    CornerRadius="4"
                                    ToolTip.Tip="放大"/>
                            <Button Content="↺" 
                                    Click="OnZoomResetClick"
                                    Background="#444"
                                    Foreground="White"
                                    CornerRadius="4"
                                    ToolTip.Tip="重置缩放"/>
                        </StackPanel>
                    </Grid>
                </Border>
    
                <!-- 浏览器区域 -->
                <Grid x:Name="BrowserContainer" Grid.Row="1">
                    <!-- 占位符面板 -->
                    <Border x:Name="PlaceholderPanel" 
                            Background="{Binding Background, RelativeSource={RelativeSource AncestorType=UserControl}}"
                            IsVisible="True"
                            ZIndex="10">
                        <StackPanel VerticalAlignment="Center" 
                                    HorizontalAlignment="Center" 
                                    Spacing="16">
                            <TextBlock Text="🔗" FontSize="72" HorizontalAlignment="Center"/>
                            <TextBlock Text="动态因果链可视化查看器" 
                                       FontSize="26" 
                                       FontWeight="SemiBold"
                                       Foreground="White"
                                       HorizontalAlignment="Center"/>
                            <TextBlock Text="支持 vis-network、D3.js、ECharts 等可视化库" 
                                       FontSize="13" 
                                       Foreground="#666"
                                       HorizontalAlignment="Center"/>
                            <Button Content="📂 选择 HTML 文件" 
                                    Click="OnOpenFileClick"
                                    Background="#0078d4"
                                    Foreground="White"
                                    Padding="24,12"
                                    FontSize="14"
                                    CornerRadius="6"
                                    Margin="0,24,0,0"
                                    HorizontalAlignment="Center"/>
                        </StackPanel>
                    </Border>
        
                    <!-- 浏览器将在代码中动态添加 -->
                </Grid>
    
                <!-- 底部状态栏 -->
                <Border Grid.Row="2" Background="#252526" Padding="12,6">
                    <Grid ColumnDefinitions="*,Auto">
                        <TextBlock x:Name="StatusText" 
                                   Text="就绪" 
                                   Foreground="#888"
                                   FontSize="12"/>
                        <StackPanel Grid.Column="1" Orientation="Horizontal" Spacing="12">
                            <Ellipse x:Name="LoadingIndicator"
                                     Width="8" Height="8"
                                     Fill="#444"/>
                            <TextBlock x:Name="LoadingText" 
                                       Text=""
                                       Foreground="#0078d4"
                                       FontSize="12"/>
                        </StackPanel>
                    </Grid>
                </Border>
            </Grid>

        </SplitView>
	</suki:SukiStackPage.Content>
</suki:SukiStackPage>
~~~
