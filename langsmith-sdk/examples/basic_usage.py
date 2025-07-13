"""
LangSmith Agent SDK 基本使用示例
"""

import os
import time
from langsmith_agent_sdk import LangSmithAgentSDK, accuracy_evaluator, similarity_evaluator

# 设置环境变量
os.environ["LANGSMITH_API_KEY"] = "your-langsmith-api-key"

# 初始化SDK
sdk = LangSmithAgentSDK(
    project_name="my-agent-project",
    # api_key="your-api-key"  # 或者直接传入API密钥
)

# 示例1: 手动创建和管理trace
def example_manual_trace():
    print("=== 手动trace示例 ===")
    
    # 创建trace
    trace_id = sdk.create_trace(
        agent_name="math_solver",
        inputs={"problem": "What is 2+2?"},
        metadata={"user_id": "user123", "session_id": "session456"}
    )
    
    # 添加执行步骤
    sdk.add_step(
        trace_id=trace_id,
        step_name="parse_problem",
        inputs={"problem": "What is 2+2?"},
        outputs={"parsed": "2+2"},
        metadata={"step_type": "parsing"}
    )
    
    # 模拟处理时间
    time.sleep(0.5)
    
    sdk.add_step(
        trace_id=trace_id,
        step_name="calculate",
        inputs={"expression": "2+2"},
        outputs={"result": 4},
        metadata={"step_type": "calculation"}
    )
    
    # 完成trace
    sdk.finish_trace(
        trace_id=trace_id,
        outputs={"answer": "4", "confidence": 0.99},
        total_tokens=50,
        cost=0.001
    )

# 示例2: 使用上下文管理器
def example_context_manager():
    print("\n=== 上下文管理器示例 ===")
    
    with sdk.trace_context("question_answerer", {"question": "What is the capital of France?"}) as trace_id:
        # 模拟agent处理
        sdk.add_step(
            trace_id=trace_id,
            step_name="knowledge_retrieval",
            inputs={"query": "capital of France"},
            outputs={"results": ["Paris is the capital of France"]},
        )
        
        time.sleep(0.3)
        
        sdk.add_step(
            trace_id=trace_id,
            step_name="answer_generation",
            inputs={"retrieved_info": "Paris is the capital of France"},
            outputs={"answer": "Paris"}
        )
        
        # 完成trace
        sdk.finish_trace(trace_id, {"answer": "Paris"})

# 示例3: 使用装饰器
@sdk.trace_decorator(
    agent_name="text_summarizer",
    extract_inputs=lambda text: {"text": text},
    extract_outputs=lambda result: {"summary": result}
)
def summarize_text(text: str) -> str:
    """模拟文本总结功能"""
    time.sleep(0.2)  # 模拟处理时间
    return f"Summary: {text[:50]}..."

def example_decorator():
    print("\n=== 装饰器示例 ===")
    
    text = "This is a long text that needs to be summarized. It contains multiple sentences and ideas."
    summary = summarize_text(text)
    print(f"原文: {text}")
    print(f"摘要: {summary}")

# 示例4: 创建数据集和评估
def example_evaluation():
    print("\n=== 评估示例 ===")
    
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
        },
        {
            "inputs": {"question": "Explain quantum computing"},
            "outputs": {"answer": "Quantum computing is a type of computation that uses quantum mechanical phenomena..."},
            "metadata": {"difficulty": "hard"}
        }
    ]
    
    # 创建数据集
    dataset_id = sdk.create_dataset("test_dataset", examples)
    
    # 定义要评估的agent函数
    def simple_qa_agent(inputs):
        question = inputs["question"]
        # 简单的QA逻辑
        if "2+2" in question:
            return {"answer": "4"}
        elif "capital of France" in question:
            return {"answer": "Paris"}
        else:
            return {"answer": "I don't know"}
    
    # 定义评估函数
    def qa_accuracy_evaluator(inputs, actual, expected):
        return 1.0 if actual.get("answer") == expected.get("answer") else 0.0
    
    # 运行评估
    import asyncio
    result = asyncio.run(sdk.evaluate_agent(
        agent_func=simple_qa_agent,
        dataset_name="test_dataset",
        evaluators=[qa_accuracy_evaluator],
        agent_name="simple_qa_agent"
    ))
    
    print(f"评估结果: {result.scores}")

# 示例5: 获取分析数据
def example_analytics():
    print("\n=== 分析示例 ===")
    
    # 获取trace分析数据
    analytics = sdk.get_trace_analytics()
    
    print("Trace分析数据:")
    print(f"总运行次数: {analytics['total_runs']}")
    print(f"成功运行: {analytics['successful_runs']}")
    print(f"失败运行: {analytics['failed_runs']}")
    print(f"平均持续时间: {analytics['avg_duration']:.2f}秒")
    
    print("\nAgent统计:")
    for agent_name, stats in analytics['agents'].items():
        print(f"  {agent_name}: {stats['count']}次运行 ({stats['success']}成功, {stats['failed']}失败)")

# 示例6: 导出trace数据
def example_export():
    print("\n=== 导出示例 ===")
    
    # 导出为JSON
    sdk.export_traces("traces.json", format="json")
    
    # 导出为CSV
    sdk.export_traces("traces.csv", format="csv")
    
    print("数据导出完成!")

if __name__ == "__main__":
    # 运行所有示例
    example_manual_trace()
    example_context_manager()
    example_decorator()
    example_evaluation()
    example_analytics()
    example_export()
    
    print("\n=== 所有示例运行完成 ===")