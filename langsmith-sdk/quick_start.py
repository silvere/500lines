#!/usr/bin/env python3
"""
LangSmith Agent SDK 快速开始脚本

这个脚本帮助你快速测试和使用LangSmith Agent SDK
"""

import os
import sys
from datetime import datetime

# 检查API密钥
if not os.getenv("LANGSMITH_API_KEY"):
    print("⚠️  请先设置LANGSMITH_API_KEY环境变量")
    print("   export LANGSMITH_API_KEY='your-api-key'")
    print("   或者在.env文件中添加: LANGSMITH_API_KEY=your-api-key")
    sys.exit(1)

try:
    from langsmith_agent_sdk import LangSmithAgentSDK, accuracy_evaluator
    print("✅ 成功导入LangSmith Agent SDK")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先安装依赖: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """主函数"""
    print("🚀 LangSmith Agent SDK 快速开始")
    print("=" * 50)
    
    # 初始化SDK
    print("\n1. 初始化SDK...")
    try:
        sdk = LangSmithAgentSDK(
            project_name="quick-start-demo"
        )
        print("✅ SDK初始化成功")
    except Exception as e:
        print(f"❌ SDK初始化失败: {e}")
        return
    
    # 演示基本trace功能
    print("\n2. 演示基本trace功能...")
    demo_basic_trace(sdk)
    
    # 演示上下文管理器
    print("\n3. 演示上下文管理器...")
    demo_context_manager(sdk)
    
    # 演示装饰器
    print("\n4. 演示装饰器...")
    demo_decorator(sdk)
    
    # 演示评估功能
    print("\n5. 演示评估功能...")
    demo_evaluation(sdk)
    
    # 演示分析功能
    print("\n6. 演示分析功能...")
    demo_analytics(sdk)
    
    print("\n🎉 快速开始演示完成!")
    print("📚 查看examples/目录获取更多示例")
    print("📖 阅读README.md获取详细文档")


def demo_basic_trace(sdk):
    """演示基本trace功能"""
    try:
        # 创建trace
        trace_id = sdk.create_trace(
            agent_name="demo_agent",
            inputs={"query": "Hello, LangSmith!"},
            metadata={"demo": "basic_trace"}
        )
        
        # 添加步骤
        sdk.add_step(
            trace_id=trace_id,
            step_name="process_query",
            inputs={"query": "Hello, LangSmith!"},
            outputs={"processed": "Processed query"},
        )
        
        # 完成trace
        sdk.finish_trace(
            trace_id=trace_id,
            outputs={"response": "Hello from LangSmith Agent SDK!"},
            total_tokens=25,
            cost=0.0005
        )
        
        print("✅ 基本trace演示完成")
        
    except Exception as e:
        print(f"❌ 基本trace演示失败: {e}")


def demo_context_manager(sdk):
    """演示上下文管理器"""
    try:
        with sdk.trace_context(
            "context_demo_agent", 
            {"task": "demonstrate context manager"}
        ) as trace_id:
            
            sdk.add_step(
                trace_id=trace_id,
                step_name="preparation",
                inputs={"task": "demonstrate context manager"},
                outputs={"status": "prepared"}
            )
            
            sdk.add_step(
                trace_id=trace_id,
                step_name="execution",
                inputs={"status": "prepared"},
                outputs={"result": "context manager works!"}
            )
            
            sdk.finish_trace(
                trace_id,
                {"message": "Context manager demonstration completed"}
            )
        
        print("✅ 上下文管理器演示完成")
        
    except Exception as e:
        print(f"❌ 上下文管理器演示失败: {e}")


def demo_decorator(sdk):
    """演示装饰器"""
    try:
        @sdk.trace_decorator(
            agent_name="decorator_demo_agent",
            extract_inputs=lambda x: {"input": x},
            extract_outputs=lambda result: {"output": result}
        )
        def simple_function(text):
            """简单的示例函数"""
            return f"Processed: {text}"
        
        # 调用被装饰的函数
        result = simple_function("Hello, decorator!")
        print(f"   函数结果: {result}")
        print("✅ 装饰器演示完成")
        
    except Exception as e:
        print(f"❌ 装饰器演示失败: {e}")


def demo_evaluation(sdk):
    """演示评估功能"""
    try:
        # 创建测试数据集
        examples = [
            {
                "inputs": {"question": "What is 1+1?"},
                "outputs": {"answer": "2"},
                "metadata": {"type": "math"}
            },
            {
                "inputs": {"question": "What is the capital of China?"},
                "outputs": {"answer": "Beijing"},
                "metadata": {"type": "geography"}
            }
        ]
        
        # 创建数据集
        dataset_id = sdk.create_dataset("quick_start_dataset", examples)
        
        # 定义测试agent
        def test_agent(inputs):
            question = inputs["question"]
            if "1+1" in question:
                return {"answer": "2"}
            elif "capital of China" in question:
                return {"answer": "Beijing"}
            else:
                return {"answer": "I don't know"}
        
        # 运行评估
        import asyncio
        
        result = asyncio.run(sdk.evaluate_agent(
            agent_func=test_agent,
            dataset_name="quick_start_dataset",
            evaluators=[accuracy_evaluator],
            agent_name="quick_start_agent"
        ))
        
        print(f"   评估分数: {result.scores}")
        print("✅ 评估演示完成")
        
    except Exception as e:
        print(f"❌ 评估演示失败: {e}")


def demo_analytics(sdk):
    """演示分析功能"""
    try:
        # 获取分析数据
        analytics = sdk.get_trace_analytics()
        
        print(f"   总运行次数: {analytics['total_runs']}")
        print(f"   成功运行: {analytics['successful_runs']}")
        print(f"   失败运行: {analytics['failed_runs']}")
        
        if analytics['total_runs'] > 0:
            success_rate = analytics['successful_runs'] / analytics['total_runs'] * 100
            print(f"   成功率: {success_rate:.1f}%")
        
        if analytics['agents']:
            print("   Agent统计:")
            for agent_name, stats in analytics['agents'].items():
                print(f"     {agent_name}: {stats['count']}次运行")
        
        print("✅ 分析演示完成")
        
    except Exception as e:
        print(f"❌ 分析演示失败: {e}")


if __name__ == "__main__":
    main()