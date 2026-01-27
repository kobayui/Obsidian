## 1.构建算法引擎 (Python 算法层)

### Eg1.PCA及贡献率分析算法

~~~
class PCADiagnosticEngine:

    def __init__(self, model_dir="models"):

        self.model_dir = model_dir

        if not os.path.exists(model_dir):

            os.makedirs(model_dir)

  

    def train_and_save(self, device_id, normal_data, feature_names):

        """训练模型并持久化保存 (含 Scaler)"""

        from sklearn.preprocessing import StandardScaler

        from sklearn.decomposition import PCA

        scaler = StandardScaler()

        pca = PCA(n_components=0.9) # 保留90%信息量

        # 训练

        X_scaled = scaler.fit_transform(normal_data)

        pca.fit(X_scaled)

        # 计算 Q 阈值 (95% 置信度)

        X_rec = pca.inverse_transform(pca.transform(X_scaled))

        errors = np.sum((X_scaled - X_rec)**2, axis=1)

        q_limit = np.percentile(errors, 95)

        # 打包保存

        model_pack = {

            "scaler": scaler,

            "pca": pca,

            "q_limit": q_limit,

            "feature_names": feature_names

        }

        joblib.dump(model_pack, os.path.join(self.model_dir, f"{device_id}.pkl"))

        print(f"[System] 模型 {device_id} 训练完成并保存。")

  

    def diagnose(self, device_id, current_data_df):

        """加载模型并诊断"""

        model_path = os.path.join(self.model_dir, f"{device_id}.pkl")

        if not os.path.exists(model_path):

            return {"error": f"未找到设备 {device_id} 的模型，请先进行基准训练。"}

        # 加载模型

        pack = joblib.load(model_path)

        scaler = pack["scaler"]

        pca = pack["pca"]

        q_limit = pack["q_limit"]

        feats = pack["feature_names"]

        # 计算

        # X = current_data_df[feats].values

        X = current_data_df[feats]

        X_scaled = scaler.transform(X)

        X_proj = pca.transform(X_scaled)

        X_rec = pca.inverse_transform(X_proj)

        # Q 统计量

        q_val = np.sum((X_scaled - X_rec)**2)

        # 结果封装

        status = "Anomaly" if q_val > q_limit else "Normal"

        # 贡献率分析 (仅在异常时计算)

        contributions = {}

        if status == "Anomaly":

            raw_contrib = (X_scaled[0] - X_rec[0])**2

            total_contrib = np.sum(raw_contrib)

            for i, name in enumerate(feats):

                contributions[name] = (raw_contrib[i] / total_contrib) * 100

            # 排序找到最大根因

            sorted_causes = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

            root_cause = sorted_causes[0]

        else:

            root_cause = (None, 0)

  

        return {

            "status": status,

            "q_value": round(q_val, 4),

            "q_limit": round(q_limit, 4),

            "root_cause_sensor": root_cause[0],

            "root_cause_confidence": round(root_cause[1], 2)

        }
~~~


## 2.知识检索服务：RAG (数据层)

~~~
class RAGService:

    def __init__(self, db_path="chroma_db", mock_mode=True):

        self.mock_mode = mock_mode

        self.db_path = db_path

        if not mock_mode:

            # 真实模式：需要下载 Embedding 模型

            self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

            self.vectordb = Chroma(persist_directory=db_path, embedding_function=self.embedding)

    def search(self, query):

        if self.mock_mode:

            # --- 模拟数据，防止无 PDF 时报错 ---

            if "振动" in query or "Pump" in query:

                return "【维修手册-第3章】泵体振动过大处理：\n1. 检查地脚螺栓是否松动。\n2. 检查联轴器对中情况（标准偏差<0.05mm）。\n3. 若轴承温度同时也升高，优先检查轴承磨损。"

            else:

                return "【通用安全规范】请在停机状态下进行检查，并佩戴防护手套。"

        else:

            # 真实检索

            docs = self.vectordb.similarity_search(query, k=2)

            return "\n".join([d.page_content for d in docs])
~~~

## 3.Agent 工具定义 (工具层)

~~~
@tool

def run_fault_diagnosis(device_id: str):

    """

    运行设备故障诊断算法。

    当用户询问设备状态、是否正常、或者要求分析数据时使用。

    参数: device_id (例如 'Pump_01')

    """

    print(f"\n[Tool Calling] 正在调用 PCA 算法分析 {device_id}...")

    # 模拟获取实时传感器数据 (实际项目中这里会读数据库)

    # 这里模拟一个故障数据：稍微偏离正常值

    mock_realtime_data = pd.DataFrame([[85.5, 0.65, 0.15, 12.0]],

                                      columns=['Temp', 'Vib_X', 'Vib_Y', 'Current'])

    result = pca_engine.diagnose(device_id, mock_realtime_data)

    if "error" in result:

        return result["error"]

    if result["status"] == "Normal":

        return f"设备 {device_id} 运行正常。Q统计量 {result['q_value']} (阈值 {result['q_limit']})。"

    else:

        return (f"⚠️ 警告：设备 {device_id} 检测到异常！\n"

                f"Q统计量: {result['q_value']} (超过阈值 {result['q_limit']})\n"

                f"主要根因: {result['root_cause_sensor']} 传感器 (贡献率 {result['root_cause_confidence']}%)。\n"

                f"建议调用知识库查询针对 '{result['root_cause_sensor']}' 异常的处理措施。")

  

@tool

def search_knowledge_base(query: str):

    """

    查询维修手册和专家知识库。

    当需要查找故障原因、维修建议、操作步骤时使用。

    """

    print(f"\n[Tool Calling] 正在检索知识库: {query}...")

    return rag_service.search(query)
~~~

## 4. LangGraph 智能编排 (逻辑层)

### 定义Agent状态

~~~
class AgentState(TypedDict):

    messages: Annotated[List[BaseMessage], add_messages]
~~~

### 设置LLM，使用本地OLLAMA

~~~
llm = ChatOllama(

    model="qwen2.5:7b",  # 确保这里是你已下载成功的模型名

    temperature=0,

    base_url="http://127.0.0.1:11434"  # 强制指定本地 IP

)
~~~

### 绑定工具

~~~
ools = [run_fault_diagnosis, search_knowledge_base]

llm_with_tools = llm.bind_tools(tools)
~~~

### 定义节点逻辑

~~~
def agent_node(state: AgentState):

    return {"messages": [llm_with_tools.invoke(state["messages"])]}

  

def tool_node(state: AgentState):

    # LangGraph 内置的 ToolNode 比较方便，但为了演示清晰，我们手动配置

    return ToolNode(tools).invoke(state)
~~~

### 定义路由逻辑

~~~
def router(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:

        return "tools"

    return END
~~~

### 构建图

~~~
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)

workflow.add_node("tools", tool_node)

  

workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", router)

workflow.add_edge("tools", "agent") # 工具执行完，回传给 Agent 生成回复

app = workflow.compile()
~~~

### 初始化全局实例

~~~
pca_engine = PCADiagnosticEngine()

# rag_service = RAGService(mock_mode=True) # 演示模式开启

rag_service = RAGService(mock_mode=False)
~~~

# 5.主程序与 Demo 演示

~~~
if __name__ == "__main__":

    # 1. 准备环境

    init_demo_data()

    # 2. 模拟用户输入

    # 场景：用户发现问题，Agent 先诊断，发现异常后自动查库，最后给出建议

    # user_input = "帮我看一下 Pump_01 现在的状态，如果有问题告诉我怎么修。"

    user_input = input("请输入用户问题: ")

    print(f"User: {user_input}")

    print("Agent is thinking... (这可能需要几秒钟)")

    # 3. 运行 Agent

    inputs = {"messages": [HumanMessage(content=user_input)]}

    # 使用 stream 可以看到中间步骤

    for output in app.stream(inputs):

        for key, value in output.items():

            # 打印中间节点的产生的信息

            if key == "agent":

                print(f"\n[Agent 思考结果]: {value['messages'][0].content}")

            elif key == "tools":

                # 工具输出通常在 message 的 artifact 里，这里简化打印

                print(f"[Tool 执行完毕]")

  

    # 4. 获取最终回复

    # (在 stream 循环结束后，最后一条 agent 消息即为最终回复)
~~~
