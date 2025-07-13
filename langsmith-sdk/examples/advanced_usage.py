"""
LangSmith Agent SDK 高级使用示例
"""

import os
import time
import asyncio
from typing import Dict, Any, List
from langsmith_agent_sdk import LangSmithAgentSDK, custom_evaluator

# 设置环境变量
os.environ["LANGSMITH_API_KEY"] = "your-langsmith-api-key"

# 初始化SDK
sdk = LangSmithAgentSDK(
    project_name="advanced-agent-project"
)

# 高级示例1: 复杂的多步骤agent
class ComplexAgent:
    def __init__(self, name: str):
        self.name = name
        self.sdk = sdk
        
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求的完整流程"""
        
        with self.sdk.trace_context(
            self.name, 
            {"user_input": user_input},
            metadata={"agent_type": "complex", "version": "1.0"}
        ) as trace_id:
            
            # 步骤1: 输入理解
            understanding = self._understand_input(trace_id, user_input)
            
            # 步骤2: 计划生成
            plan = self._generate_plan(trace_id, understanding)
            
            # 步骤3: 执行计划
            results = self._execute_plan(trace_id, plan)
            
            # 步骤4: 结果整合
            final_result = self._integrate_results(trace_id, results)
            
            # 完成trace
            self.sdk.finish_trace(
                trace_id, 
                final_result,
                total_tokens=200,
                cost=0.05
            )
            
            return final_result
            
    def _understand_input(self, trace_id: str, user_input: str) -> Dict[str, Any]:
        """理解用户输入"""
        self.sdk.add_step(
            trace_id=trace_id,
            step_name="input_understanding",
            inputs={"user_input": user_input},
            outputs={"intent": "question", "entities": []},
            metadata={"confidence": 0.95}
        )
        
        time.sleep(0.1)  # 模拟处理时间
        return {"intent": "question", "entities": []}
        
    def _generate_plan(self, trace_id: str, understanding: Dict[str, Any]) -> List[str]:
        """生成执行计划"""
        plan = ["search_knowledge", "analyze_results", "generate_response"]
        
        self.sdk.add_step(
            trace_id=trace_id,
            step_name="plan_generation",
            inputs=understanding,
            outputs={"plan": plan},
            metadata={"plan_type": "sequential"}
        )
        
        time.sleep(0.2)
        return plan
        
    def _execute_plan(self, trace_id: str, plan: List[str]) -> List[Dict[str, Any]]:
        """执行计划"""
        results = []
        
        for i, step in enumerate(plan):
            result = {"step": step, "status": "completed", "data": f"result_{i}"}
            results.append(result)
            
            self.sdk.add_step(
                trace_id=trace_id,
                step_name=f"execute_{step}",
                inputs={"step_name": step, "step_index": i},
                outputs=result,
                metadata={"execution_order": i}
            )
            
            time.sleep(0.1)
            
        return results
        
    def _integrate_results(self, trace_id: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """整合结果"""
        integrated = {
            "response": "This is the integrated response from all steps",
            "confidence": 0.85,
            "sources": [r["data"] for r in results]
        }
        
        self.sdk.add_step(
            trace_id=trace_id,
            step_name="result_integration",
            inputs={"results": results},
            outputs=integrated,
            metadata={"integration_method": "weighted_average"}
        )
        
        time.sleep(0.1)
        return integrated

# 高级示例2: 自定义评估器
@custom_evaluator
def response_quality_evaluator(inputs: Dict[str, Any], actual: Any, expected: Any) -> float:
    """评估响应质量"""
    actual_response = actual.get("response", "")
    expected_response = expected.get("response", "")
    
    # 简单的质量评估逻辑
    if not actual_response:
        return 0.0
    
    # 检查关键词匹配
    actual_words = set(actual_response.lower().split())
    expected_words = set(expected_response.lower().split())
    
    if not expected_words:
        return 0.5  # 默认分数
    
    intersection = actual_words.intersection(expected_words)
    return len(intersection) / len(expected_words)

@custom_evaluator
def confidence_evaluator(inputs: Dict[str, Any], actual: Any, expected: Any) -> float:
    """评估置信度"""
    confidence = actual.get("confidence", 0.0)
    return confidence

# 高级示例3: 批量评估
async def batch_evaluation():
    """批量评估多个agent"""
    
    # 创建多个agent
    agents = {
        "simple_agent": ComplexAgent("SimpleAgent"),
        "complex_agent": ComplexAgent("ComplexAgent"),
    }
    
    # 创建测试数据集
    test_examples = [
        {
            "inputs": {"user_input": "What is machine learning?"},
            "outputs": {"response": "Machine learning is a subset of artificial intelligence", "confidence": 0.9},
            "metadata": {"category": "definition"}
        },
        {
            "inputs": {"user_input": "How does neural network work?"},
            "outputs": {"response": "Neural networks work by processing data through interconnected layers", "confidence": 0.8},
            "metadata": {"category": "explanation"}
        }
    ]
    
    # 创建数据集
    dataset_id = sdk.create_dataset("advanced_test_dataset", test_examples)
    
    # 评估每个agent
    evaluators = [response_quality_evaluator, confidence_evaluator]
    
    results = {}
    for agent_name, agent in agents.items():
        print(f"\n评估 {agent_name}...")
        
        def agent_wrapper(inputs):
            return agent.process_request(inputs["user_input"])
        
        result = await sdk.evaluate_agent(
            agent_func=agent_wrapper,
            dataset_name="advanced_test_dataset",
            evaluators=evaluators,
            agent_name=agent_name
        )
        
        results[agent_name] = result
        
    # 比较结果
    print("\n=== 评估结果比较 ===")
    for agent_name, result in results.items():
        print(f"\n{agent_name}:")
        for metric, score in result.scores.items():
            print(f"  {metric}: {score:.4f}")

# 高级示例4: 实时监控
def real_time_monitoring():
    """实时监控agent性能"""
    
    agent = ComplexAgent("MonitoredAgent")
    
    # 模拟多次运行
    test_inputs = [
        "What is AI?",
        "How do computers work?",
        "Explain quantum physics",
        "What is the meaning of life?"
    ]
    
    print("=== 实时监控示例 ===")
    
    for i, input_text in enumerate(test_inputs):
        print(f"\n运行 {i+1}/{len(test_inputs)}: {input_text}")
        
        # 运行agent
        result = agent.process_request(input_text)
        
        # 获取当前分析数据
        analytics = sdk.get_trace_analytics()
        
        print(f"当前统计: {analytics['total_runs']}次运行, "
              f"成功率: {analytics['successful_runs']/analytics['total_runs']*100:.1f}%")
        
        time.sleep(1)  # 模拟间隔

# 高级示例5: 错误处理和恢复
def error_handling_example():
    """展示错误处理"""
    
    print("\n=== 错误处理示例 ===")
    
    with sdk.trace_context("error_prone_agent", {"input": "test"}) as trace_id:
        try:
            # 模拟正常步骤
            sdk.add_step(
                trace_id=trace_id,
                step_name="normal_step",
                inputs={"data": "test"},
                outputs={"result": "success"}
            )
            
            # 模拟错误步骤
            sdk.add_step(
                trace_id=trace_id,
                step_name="error_step",
                inputs={"data": "test"},
                outputs=None,
                metadata={"error": "Simulated error"}
            )
            
            # 模拟异常
            raise ValueError("This is a simulated error")
            
        except Exception as e:
            print(f"捕获到错误: {e}")
            # 错误会自动记录在trace中
            
    print("错误处理完成")

# 高级示例6: 性能分析
def performance_analysis():
    """性能分析"""
    
    print("\n=== 性能分析示例 ===")
    
    # 获取详细分析
    analytics = sdk.get_trace_analytics()
    
    print("性能分析报告:")
    print(f"总运行次数: {analytics['total_runs']}")
    print(f"平均执行时间: {analytics['avg_duration']:.3f}秒")
    print(f"成功率: {analytics['successful_runs']/max(analytics['total_runs'], 1)*100:.1f}%")
    
    # 分析各个agent的性能
    if analytics['agents']:
        print("\nAgent性能排名:")
        sorted_agents = sorted(
            analytics['agents'].items(),
            key=lambda x: x[1]['success'] / max(x[1]['count'], 1),
            reverse=True
        )
        
        for agent_name, stats in sorted_agents:
            success_rate = stats['success'] / max(stats['count'], 1) * 100
            print(f"  {agent_name}: {success_rate:.1f}% 成功率 ({stats['count']}次运行)")
    
    # 导出详细报告
    sdk.export_traces("performance_report.json", format="json")
    print("\n性能报告已导出到 performance_report.json")

if __name__ == "__main__":
    # 运行高级示例
    
    # 1. 复杂agent示例
    print("=== 复杂Agent示例 ===")
    complex_agent = ComplexAgent("DemoAgent")
    result = complex_agent.process_request("What is artificial intelligence?")
    print(f"Agent响应: {result}")
    
    # 2. 批量评估
    print("\n=== 批量评估 ===")
    asyncio.run(batch_evaluation())
    
    # 3. 实时监控
    real_time_monitoring()
    
    # 4. 错误处理
    error_handling_example()
    
    # 5. 性能分析
    performance_analysis()
    
    print("\n=== 所有高级示例运行完成 ===")