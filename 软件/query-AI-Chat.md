
模型部署及调用见笔记[[大模型部署及调用]]

连接API

~~~
	private Kernel _kernel;
    public async Task AIresultAsync()
    {

        _kernel = Kernel.CreateBuilder()
.AddOpenAIChatCompletion(
    modelId: "gpt-5-chat",
    apiKey: "sk-rkEXqwVzqoPM2LlyAnI3PXRh8lPqIKARLQXogy",
    httpClient: new HttpClient(new OpenAIHttpClientHandler("https://api.token-ai.cn/")))
.Build();
    }
    
    private IKernelMemory _memory;

public async Task AIresultAsyncF()
{
    var httpClientHandler = new OpenAIHttpClientHandler("https://api.token-ai.cn/");
    var key = "sk-rkEXqwVzqoPM2LlyAnI3PXRh8lPqIKARLQXogy";
    //var embeddingModel = "text-embedding-3-large";
    //var embeddingModel = "text-embedding-ada-002";
    var embeddingModel = "Qwen3-30B-A3B";
    //var embeddingModel = "embedding-2";
    //var textModel = "gpt-3.5-turbo-0125";
    var textModel = "Qwen3-30B-A3B";

    _memory = new KernelMemoryBuilder()
        .WithOpenAITextGeneration(new OpenAIConfig()
        {
            APIKey = key,
            TextModel = textModel
        }, null, new HttpClient(httpClientHandler))
        .WithOpenAITextEmbeddingGeneration(new OpenAIConfig()
        {
            APIKey = key,
            EmbeddingModel = embeddingModel,
        }, null, false, new HttpClient(httpClientHandler))
        .Build<MemoryServerless>();
}
~~~

