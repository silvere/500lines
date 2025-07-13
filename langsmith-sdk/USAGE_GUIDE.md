# LangSmith Agent SDK 使用指南

这个详细的使用指南将帮助你充分利用LangSmith Agent SDK的所有功能。

## 目录

1. [环境准备](#环境准备)
2. [基本概念](#基本概念)
3. [快速开始](#快速开始)
4. [核心功能](#核心功能)
5. [高级用法](#高级用法)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)
8. [API参考](#api参考)

## 环境准备

### 1. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd langsmith-sdk

# 安装依赖
pip install -r requirements.txt

# 或者使用开发模式安装
pip install -e .
```

### 2. 设置API密钥

你需要从[LangSmith](https://smith.langchain.com/)获取API密钥：

```bash
# 方式1: 环境变量
export LANGSMITH_API_KEY="your-api-key"

# 方式2: .env文件
echo "LANGSMITH_API_KEY=your-api-key" > .env

# 方式3: 直接在代码中设置
sdk = LangSmithAgentSDK(api_key="your-api-key")
```

### 3. 验证安装

```bash
# 运行快速开始脚本
python quick_start.py

# 或者使用CLI工具
python -m langsmith_agent_sdk.cli --help
```

## 基本概念

### Trace（追踪）

Trace是对Agent执行过程的完整记录，包含：

- **输入**: Agent接收的数据
- **输出**: Agent产生的结果
- **步骤**: 执行过程中的各个阶段
- **元数据**: 额外的上下文信息
- **性能指标**: 执行时间、token使用量、成本等

### Evaluation（评估）

Evaluation是对Agent性能的系统性测试，包含：

- **数据集**: 测试用例集合
- **评估器**: 评分函数
- **结果**: 性能指标和详细分析

### Agent

Agent是执行特定任务的智能体，可以是：

- 函数
- 类方法
- 复杂的多步骤流程

## 快速开始

### 1. 初始化SDK

```python
from langsmith_agent_sdk import LangSmithAgentSDK

# 基本初始化
sdk = LangSmithAgentSDK(project_name="my-project")

# 完整初始化
sdk = LangSmithAgentSDK(
    api_key="your-api-key",
    project_name="my-project",
    endpoint="https://api.smith.langchain.com"
)
```

### 2. 创建你的第一个trace

```python
# 手动管理
trace_id = sdk.create_trace(
    agent_name="my_agent",
    inputs={"query": "Hello world"},
    metadata={"user_id": "123"}
)

sdk.add_step(
    trace_id=trace_id,
    step_name="process",
    inputs={"query": "Hello world"},
    outputs={"result": "Hello from agent"}
)

sdk.finish_trace(
    trace_id=trace_id,
    outputs={"response": "Hello from agent"}
)
```

### 3. 使用上下文管理器（推荐）

```python
with sdk.trace_context("my_agent", {"query": "Hello world"}) as trace_id:
    # 你的agent逻辑
    result = process_query("Hello world")
    
    # 添加步骤
    sdk.add_step(
        trace_id=trace_id,
        step_name="process",
        inputs={"query": "Hello world"},
        outputs={"result": result}
    )
    
    # 完成trace
    sdk.finish_trace(trace_id, {"response": result})
```

## 核心功能

### 1. Trace管理

#### 创建Trace

```python
trace_id = sdk.create_trace(
    agent_name="my_agent",
    inputs={"query": "user input"},
    metadata={
        "user_id": "user123",
        "session_id": "session456",
        "timestamp": "2024-01-01T00:00:00Z"
    }
)
```

#### 添加步骤

```python
sdk.add_step(
    trace_id=trace_id,
    step_name="data_preprocessing",
    inputs={"raw_data": "..."},
    outputs={"processed_data": "..."},
    metadata={
        "processing_time": 0.5,
        "method": "normalization"
    }
)
```

#### 完成Trace

```python
sdk.finish_trace(
    trace_id=trace_id,
    outputs={"final_result": "..."},
    error=None,  # 或者错误信息
    total_tokens=150,
    cost=0.003
)
```

### 2. 上下文管理器

上下文管理器自动处理trace的生命周期，推荐使用：

```python
with sdk.trace_context(
    agent_name="my_agent",
    inputs={"query": "user input"},
    metadata={"context": "web_chat"}
) as trace_id:
    try:
        # 你的agent逻辑
        result = my_agent_function(inputs)
        
        # 完成trace
        sdk.finish_trace(trace_id, {"result": result})
        
    except Exception as e:
        # 错误会自动记录
        raise
```

### 3. 装饰器

装饰器提供最简单的使用方式：

```python
@sdk.trace_decorator(
    agent_name="text_processor",
    extract_inputs=lambda text, **kwargs: {"text": text, "options": kwargs},
    extract_outputs=lambda result: {"processed_text": result}
)
def process_text(text: str, uppercase: bool = False) -> str:
    result = text.upper() if uppercase else text.lower()
    return f"Processed: {result}"

# 使用
result = process_text("Hello World", uppercase=True)
```

### 4. 创建数据集

```python
examples = [
    {
        "inputs": {"question": "What is AI?"},
        "outputs": {"answer": "AI is artificial intelligence"},
        "metadata": {"category": "definition", "difficulty": "easy"}
    },
    {
        "inputs": {"question": "How does ML work?"},
        "outputs": {"answer": "ML uses algorithms to learn patterns"},
        "metadata": {"category": "explanation", "difficulty": "medium"}
    }
]

dataset_id = sdk.create_dataset("my_dataset", examples)
```

### 5. Agent评估

```python
import asyncio
from langsmith_agent_sdk import accuracy_evaluator

def my_agent(inputs):
    question = inputs["question"]
    # 你的agent逻辑
    return {"answer": "Generated answer"}

# 运行评估
result = asyncio.run(sdk.evaluate_agent(
    agent_func=my_agent,
    dataset_name="my_dataset",
    evaluators=[accuracy_evaluator],
    agent_name="my_agent"
))

print(f"准确率: {result.scores['accuracy_evaluator']}")
```

## 高级用法

### 1. 自定义评估器

```python
from langsmith_agent_sdk import custom_evaluator

@custom_evaluator
def semantic_similarity_evaluator(inputs, actual, expected):
    """语义相似度评估器"""
    # 使用embedding计算相似度
    actual_text = actual.get("answer", "")
    expected_text = expected.get("answer", "")
    
    # 你的相似度计算逻辑
    similarity = calculate_similarity(actual_text, expected_text)
    return similarity

@custom_evaluator
def response_length_evaluator(inputs, actual, expected):
    """响应长度评估器"""
    actual_length = len(actual.get("answer", ""))
    expected_length = len(expected.get("answer", ""))
    
    # 长度差异越小，分数越高
    length_diff = abs(actual_length - expected_length)
    max_length = max(actual_length, expected_length)
    
    if max_length == 0:
        return 1.0
    
    return max(0.0, 1.0 - length_diff / max_length)
```

### 2. 复杂Agent类

```python
class ComplexAgent:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.sdk = sdk
        
    def process_request(self, user_input: str) -> dict:
        """处理用户请求"""
        
        with self.sdk.trace_context(
            self.name,
            {"user_input": user_input},
            metadata={"config": self.config}
        ) as trace_id:
            
            # 步骤1: 预处理
            preprocessed = self._preprocess(trace_id, user_input)
            
            # 步骤2: 核心处理
            processed = self._process(trace_id, preprocessed)
            
            # 步骤3: 后处理
            result = self._postprocess(trace_id, processed)
            
            # 完成trace
            self.sdk.finish_trace(trace_id, result)
            
            return result
    
    def _preprocess(self, trace_id: str, input_data: str) -> dict:
        """预处理步骤"""
        # 清理和标准化输入
        cleaned = input_data.strip().lower()
        
        result = {
            "cleaned_input": cleaned,
            "input_length": len(cleaned),
            "language": "en"  # 简化的语言检测
        }
        
        self.sdk.add_step(
            trace_id=trace_id,
            step_name="preprocess",
            inputs={"raw_input": input_data},
            outputs=result,
            metadata={"preprocessing_config": self.config.get("preprocess", {})}
        )
        
        return result
    
    def _process(self, trace_id: str, preprocessed: dict) -> dict:
        """核心处理步骤"""
        # 模拟复杂的处理逻辑
        input_text = preprocessed["cleaned_input"]
        
        # 模拟多个子步骤
        steps = ["analysis", "reasoning", "generation"]
        results = {}
        
        for step in steps:
            step_result = self._execute_substep(trace_id, step, input_text)
            results[step] = step_result
        
        final_result = {
            "analysis": results["analysis"],
            "reasoning": results["reasoning"],
            "generated_response": results["generation"]
        }
        
        self.sdk.add_step(
            trace_id=trace_id,
            step_name="core_processing",
            inputs=preprocessed,
            outputs=final_result,
            metadata={"processing_steps": steps}
        )
        
        return final_result
    
    def _execute_substep(self, trace_id: str, step_name: str, input_text: str) -> str:
        """执行子步骤"""
        # 模拟子步骤处理
        result = f"Result of {step_name} for: {input_text}"
        
        self.sdk.add_step(
            trace_id=trace_id,
            step_name=f"substep_{step_name}",
            inputs={"input": input_text},
            outputs={"result": result},
            metadata={"substep_type": step_name}
        )
        
        return result
    
    def _postprocess(self, trace_id: str, processed: dict) -> dict:
        """后处理步骤"""
        # 格式化最终输出
        final_response = {
            "response": processed["generated_response"],
            "confidence": 0.85,
            "metadata": {
                "analysis_summary": processed["analysis"],
                "reasoning_path": processed["reasoning"]
            }
        }
        
        self.sdk.add_step(
            trace_id=trace_id,
            step_name="postprocess",
            inputs=processed,
            outputs=final_response,
            metadata={"postprocess_config": self.config.get("postprocess", {})}
        )
        
        return final_response
```

### 3. 批量评估

```python
async def batch_evaluation():
    """批量评估多个agent"""
    
    # 定义多个agent
    agents = {
        "simple_agent": SimpleAgent(),
        "complex_agent": ComplexAgent("ComplexAgent", {"mode": "advanced"}),
        "baseline_agent": BaselineAgent()
    }
    
    # 定义评估器
    evaluators = [
        accuracy_evaluator,
        semantic_similarity_evaluator,
        response_length_evaluator
    ]
    
    # 批量评估
    results = {}
    for agent_name, agent in agents.items():
        print(f"评估 {agent_name}...")
        
        def agent_wrapper(inputs):
            return agent.process_request(inputs["user_input"])
        
        result = await sdk.evaluate_agent(
            agent_func=agent_wrapper,
            dataset_name="comprehensive_dataset",
            evaluators=evaluators,
            agent_name=agent_name
        )
        
        results[agent_name] = result
    
    # 比较结果
    comparison = compare_evaluation_results(results)
    return comparison

def compare_evaluation_results(results):
    """比较评估结果"""
    comparison = {}
    
    # 获取所有指标
    all_metrics = set()
    for result in results.values():
        all_metrics.update(result.scores.keys())
    
    # 为每个指标创建排名
    for metric in all_metrics:
        metric_scores = []
        for agent_name, result in results.items():
            score = result.scores.get(metric, 0.0)
            metric_scores.append((agent_name, score))
        
        # 按分数排序
        metric_scores.sort(key=lambda x: x[1], reverse=True)
        comparison[metric] = metric_scores
    
    return comparison
```

### 4. 实时监控

```python
class AgentMonitor:
    def __init__(self, sdk: LangSmithAgentSDK):
        self.sdk = sdk
        self.alerts = []
        
    def monitor_agent_performance(self, 
                                 agent_name: str, 
                                 check_interval: int = 60):
        """监控agent性能"""
        import time
        
        while True:
            try:
                # 获取最新分析数据
                analytics = self.sdk.get_trace_analytics()
                
                # 检查性能指标
                self._check_performance_alerts(analytics, agent_name)
                
                # 打印状态
                self._print_status(analytics, agent_name)
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("监控停止")
                break
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(check_interval)
    
    def _check_performance_alerts(self, analytics: dict, agent_name: str):
        """检查性能告警"""
        if agent_name in analytics["agents"]:
            stats = analytics["agents"][agent_name]
            
            # 检查成功率
            if stats["count"] > 0:
                success_rate = stats["success"] / stats["count"]
                if success_rate < 0.8:  # 成功率低于80%
                    alert = f"⚠️  {agent_name} 成功率过低: {success_rate:.1%}"
                    if alert not in self.alerts:
                        self.alerts.append(alert)
                        print(alert)
            
            # 检查错误率
            if stats["failed"] > 5:  # 失败次数超过5次
                alert = f"⚠️  {agent_name} 错误次数过多: {stats['failed']}"
                if alert not in self.alerts:
                    self.alerts.append(alert)
                    print(alert)
    
    def _print_status(self, analytics: dict, agent_name: str):
        """打印状态信息"""
        if agent_name in analytics["agents"]:
            stats = analytics["agents"][agent_name]
            success_rate = stats["success"] / max(stats["count"], 1)
            
            print(f"📊 {agent_name} 状态:")
            print(f"   运行次数: {stats['count']}")
            print(f"   成功率: {success_rate:.1%}")
            print(f"   平均执行时间: {analytics['avg_duration']:.2f}s")
            print("-" * 40)
```

## 最佳实践

### 1. Trace设计原则

- **细粒度步骤**: 将复杂流程分解为多个小步骤
- **有意义的名称**: 使用清晰的step名称
- **丰富的元数据**: 添加有助于分析的元数据
- **错误处理**: 确保错误信息被正确记录

### 2. 性能优化

- **批量操作**: 尽量使用批量API
- **异步处理**: 使用异步评估提高效率
- **数据清理**: 定期清理过期的trace数据
- **缓存策略**: 对频繁访问的数据使用缓存

### 3. 评估策略

- **多维度评估**: 使用多个评估器获得全面评估
- **分层数据集**: 创建不同难度级别的数据集
- **持续评估**: 定期重新评估agent性能
- **A/B测试**: 比较不同版本的agent

### 4. 错误处理

```python
def robust_agent_execution(sdk, agent_func, inputs):
    """健壮的agent执行"""
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with sdk.trace_context(
                "robust_agent",
                inputs,
                metadata={"retry_count": retry_count}
            ) as trace_id:
                
                result = agent_func(inputs)
                
                # 验证结果
                if validate_result(result):
                    sdk.finish_trace(trace_id, {"result": result})
                    return result
                else:
                    raise ValueError("Invalid result")
                    
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Agent执行失败，已重试{max_retries}次: {e}")
                raise
            else:
                print(f"Agent执行失败，第{retry_count}次重试: {e}")
                time.sleep(1)  # 等待1秒后重试

def validate_result(result):
    """验证结果有效性"""
    # 实现你的验证逻辑
    return result is not None and "error" not in result
```

## 故障排除

### 常见问题

1. **API密钥错误**
   ```
   错误: 需要提供LangSmith API密钥
   解决: 设置LANGSMITH_API_KEY环境变量
   ```

2. **网络连接问题**
   ```
   错误: Connection failed
   解决: 检查网络连接，确认LangSmith服务状态
   ```

3. **项目不存在**
   ```
   错误: Project not found
   解决: 检查项目名称，或者让SDK自动创建项目
   ```

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **使用小数据集测试**
   ```python
   # 先用小数据集测试
   small_examples = examples[:5]
   test_result = await sdk.evaluate_agent(...)
   ```

3. **检查trace完整性**
   ```python
   # 验证trace数据
   analytics = sdk.get_trace_analytics()
   if analytics["total_runs"] == 0:
       print("没有trace数据，请检查agent执行")
   ```

## API参考

详细的API文档请参考README.md文件。

## 结论

LangSmith Agent SDK提供了完整的agent追踪和评估解决方案。通过合理使用本指南中的功能和最佳实践，你可以：

- 深入了解agent的执行过程
- 系统性地评估agent性能
- 持续优化agent的表现
- 建立可靠的agent监控体系

如果你有任何问题或建议，请参考项目的GitHub仓库或联系维护团队。