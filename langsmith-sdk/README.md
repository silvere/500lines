# LangSmith Agent SDK

一个强大而易用的LangSmith SDK，专为Agent的trace追踪和evaluation评估而设计。

## 特性

- 🔍 **完整的Agent追踪**: 自动记录Agent的执行过程，包括输入、输出、步骤和元数据
- 📊 **智能评估系统**: 内置多种评估器，支持自定义评估逻辑
- 🎯 **灵活的使用方式**: 支持手动管理、上下文管理器和装饰器三种使用模式
- 📈 **实时分析**: 提供详细的性能分析和统计数据
- 🔧 **易于集成**: 简单的API设计，快速集成到现有项目中
- 🎨 **美观的输出**: 使用Rich库提供彩色和格式化的控制台输出

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 初始化SDK

```python
import os
from langsmith_agent_sdk import LangSmithAgentSDK

# 设置API密钥
os.environ["LANGSMITH_API_KEY"] = "your-langsmith-api-key"

# 初始化SDK
sdk = LangSmithAgentSDK(
    project_name="my-agent-project"
)
```

### 2. 基本使用

#### 方式1: 手动管理trace

```python
# 创建trace
trace_id = sdk.create_trace(
    agent_name="my_agent",
    inputs={"question": "What is AI?"},
    metadata={"user_id": "user123"}
)

# 添加执行步骤
sdk.add_step(
    trace_id=trace_id,
    step_name="knowledge_search",
    inputs={"query": "AI definition"},
    outputs={"results": ["AI is artificial intelligence"]},
)

# 完成trace
sdk.finish_trace(
    trace_id=trace_id,
    outputs={"answer": "AI is artificial intelligence"},
    total_tokens=100,
    cost=0.001
)
```

#### 方式2: 上下文管理器（推荐）

```python
with sdk.trace_context("my_agent", {"question": "What is AI?"}) as trace_id:
    # 添加步骤
    sdk.add_step(
        trace_id=trace_id,
        step_name="processing",
        inputs={"data": "input"},
        outputs={"result": "output"}
    )
    
    # 完成trace
    sdk.finish_trace(trace_id, {"answer": "AI is artificial intelligence"})
```

#### 方式3: 装饰器

```python
@sdk.trace_decorator(
    agent_name="text_processor",
    extract_inputs=lambda text: {"text": text},
    extract_outputs=lambda result: {"result": result}
)
def process_text(text: str) -> str:
    return f"Processed: {text}"

# 使用
result = process_text("Hello world")  # 自动追踪
```

### 3. 创建数据集和评估

```python
# 创建评估数据集
examples = [
    {
        "inputs": {"question": "What is 2+2?"},
        "outputs": {"answer": "4"},
        "metadata": {"difficulty": "easy"}
    },
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"answer": "Paris"},
        "metadata": {"difficulty": "easy"}
    }
]

dataset_id = sdk.create_dataset("test_dataset", examples)

# 定义agent函数
def my_agent(inputs):
    question = inputs["question"]
    if "2+2" in question:
        return {"answer": "4"}
    elif "capital of France" in question:
        return {"answer": "Paris"}
    else:
        return {"answer": "I don't know"}

# 评估agent
import asyncio
from langsmith_agent_sdk import accuracy_evaluator

result = asyncio.run(sdk.evaluate_agent(
    agent_func=my_agent,
    dataset_name="test_dataset",
    evaluators=[accuracy_evaluator],
    agent_name="my_agent"
))

print(f"评估结果: {result.scores}")
```

## 高级功能

### 自定义评估器

```python
from langsmith_agent_sdk import custom_evaluator

@custom_evaluator
def my_custom_evaluator(inputs, actual, expected):
    # 自定义评估逻辑
    score = calculate_similarity(actual, expected)
    return score

# 使用自定义评估器
result = await sdk.evaluate_agent(
    agent_func=my_agent,
    dataset_name="test_dataset",
    evaluators=[my_custom_evaluator],
    agent_name="my_agent"
)
```

### 复杂Agent类

```python
class ComplexAgent:
    def __init__(self, name: str):
        self.name = name
        self.sdk = sdk
        
    def process_request(self, user_input: str):
        with self.sdk.trace_context(self.name, {"input": user_input}) as trace_id:
            # 步骤1: 理解输入
            understanding = self._understand_input(trace_id, user_input)
            
            # 步骤2: 生成响应
            response = self._generate_response(trace_id, understanding)
            
            # 完成trace
            self.sdk.finish_trace(trace_id, {"response": response})
            return response
            
    def _understand_input(self, trace_id: str, user_input: str):
        self.sdk.add_step(
            trace_id=trace_id,
            step_name="understand_input",
            inputs={"user_input": user_input},
            outputs={"intent": "question"}
        )
        return {"intent": "question"}
        
    def _generate_response(self, trace_id: str, understanding):
        self.sdk.add_step(
            trace_id=trace_id,
            step_name="generate_response",
            inputs=understanding,
            outputs={"response": "Generated response"}
        )
        return "Generated response"
```

### 性能分析

```python
# 获取trace分析数据
analytics = sdk.get_trace_analytics()

print(f"总运行次数: {analytics['total_runs']}")
print(f"成功率: {analytics['successful_runs']/analytics['total_runs']*100:.1f}%")
print(f"平均执行时间: {analytics['avg_duration']:.2f}秒")

# Agent性能统计
for agent_name, stats in analytics['agents'].items():
    print(f"{agent_name}: {stats['success']}/{stats['count']} 成功")
```

### 数据导出

```python
# 导出为JSON
sdk.export_traces("traces.json", format="json")

# 导出为CSV
sdk.export_traces("traces.csv", format="csv")
```

## API参考

### LangSmithAgentSDK

#### 初始化
```python
LangSmithAgentSDK(
    api_key: Optional[str] = None,
    project_name: str = "agent-traces",
    endpoint: Optional[str] = None
)
```

#### 主要方法

- `create_trace(agent_name, inputs, metadata=None)`: 创建新的trace
- `add_step(trace_id, step_name, inputs, outputs=None, metadata=None)`: 添加执行步骤
- `finish_trace(trace_id, outputs, error=None, total_tokens=None, cost=None)`: 完成trace
- `trace_context(agent_name, inputs, metadata=None)`: 上下文管理器
- `trace_decorator(agent_name=None, extract_inputs=None, extract_outputs=None)`: 装饰器
- `create_dataset(dataset_name, examples)`: 创建数据集
- `evaluate_agent(agent_func, dataset_name, evaluators, agent_name=None)`: 评估agent
- `get_trace_analytics(project_name=None, start_date=None, end_date=None)`: 获取分析数据
- `export_traces(output_file, project_name=None, format="json")`: 导出trace数据

### 数据模型

#### AgentTrace
```python
class AgentTrace(BaseModel):
    trace_id: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime]
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    steps: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: Optional[str]
    total_tokens: Optional[int]
    cost: Optional[float]
```

#### EvaluationResult
```python
class EvaluationResult(BaseModel):
    evaluation_id: str
    agent_name: str
    dataset_name: str
    scores: Dict[str, float]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    created_at: datetime
```

### 内置评估器

- `accuracy_evaluator`: 准确率评估器
- `similarity_evaluator`: 文本相似度评估器
- `custom_evaluator`: 自定义评估器装饰器

## 环境设置

确保设置了以下环境变量：

```bash
export LANGSMITH_API_KEY="your-langsmith-api-key"
```

或者在代码中直接传入：

```python
sdk = LangSmithAgentSDK(api_key="your-api-key")
```

## 示例项目

查看 `examples/` 目录中的完整示例：

- `basic_usage.py`: 基本使用示例
- `advanced_usage.py`: 高级功能示例

## 最佳实践

1. **使用上下文管理器**: 推荐使用 `trace_context()` 来自动管理trace生命周期
2. **详细的元数据**: 为trace和步骤添加有意义的元数据，便于后续分析
3. **错误处理**: 始终在可能出错的地方添加适当的错误处理
4. **性能监控**: 定期检查trace分析数据，监控agent性能
5. **数据导出**: 定期导出trace数据进行离线分析

## 故障排除

### 常见问题

1. **API密钥错误**: 确保设置了正确的LANGSMITH_API_KEY
2. **网络连接问题**: 检查网络连接和LangSmith服务状态
3. **权限问题**: 确保API密钥具有足够的权限

### 调试技巧

1. 启用详细日志输出
2. 使用小数据集测试
3. 检查trace数据的完整性

## 贡献

欢迎贡献代码、报告bug或提出新功能建议！

## 许可证

MIT License

---

更多信息请查看 [LangSmith官方文档](https://docs.langchain.com/langsmith)